import os
import time
from bisect import bisect_left
import numpy as np
import torch
import lietorch
import droid_backends
import src.geom.ba
from torch.multiprocessing import Value
from torch.multiprocessing import Lock
import torch.nn.functional as F
from typing import Any, Dict, Optional

from src.modules.droid_net import cvx_upsample
import src.geom.projective_ops as pops
from src.utils.common import align_scale_and_shift
from src.utils.Printer import FontColor
from src.utils.dyn_uncertainty import mapping_utils as map_utils

class DepthVideo:
    ''' store the estimated poses and depth maps, 
        shared between tracker and mapper '''
    def __init__(self, cfg, printer, uncer_network=None, beta_client=None):
        self.cfg =cfg
        self.output = f"{cfg['data']['output']}/{cfg['scene']}"
        ht = cfg['cam']['H_out']
        self.ht = ht
        wd = cfg['cam']['W_out']
        self.wd = wd
        self.counter = Value('i', 0) # current keyframe count
        buffer = cfg['tracking']['buffer']
        self.metric_depth_reg = cfg['tracking']['backend']['metric_depth_reg']
        if not self.metric_depth_reg:
            self.printer.print(f"Metric depth for regularization is not activated.",FontColor.INFO)
            self.printer.print(f"This should not happen for WildGS-SLAM unless you are doing ablation study",FontColor.INFO)
        self.mono_thres = cfg['tracking']['mono_thres']
        self.device = cfg['device']
        self.down_scale = 8
        self.feature_stride = 16 if "dinov3" in cfg["mono_prior"]["feature_extractor"] else 14
        self.slice_h = slice(self.down_scale // 2 - 1, ht//self.down_scale*self.down_scale+1, self.down_scale)
        self.slice_w = slice(self.down_scale // 2 - 1, wd//self.down_scale*self.down_scale+1, self.down_scale)
        ### state attributes ###
        self.timestamp = torch.zeros(buffer, device=self.device, dtype=torch.float).share_memory_()
        self.frame_ids = torch.full((buffer,), -1, device=self.device, dtype=torch.int32).share_memory_()
        # To save gpu ram, we put images to cpu as it is never used
        self.images = torch.zeros(buffer, 3, ht, wd, device='cpu', dtype=torch.float32)

        # whether the valid_depth_mask is calculated/updated, if dirty, not updated, otherwise, updated
        self.dirty = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_() 
        # whether the corresponding part of pointcloud is deformed w.r.t. the poses and depths 
        self.npc_dirty = torch.zeros(buffer, device=self.device, dtype=torch.bool).share_memory_()

        self.poses = torch.zeros(buffer, 7, device=self.device, dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.zeros = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device=self.device, dtype=torch.float).share_memory_()
        self.mono_disps = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        self.mono_disps_up = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.float).share_memory_()
        self.mono_disps_mask_up = torch.ones(buffer, ht, wd, device=self.device, dtype=torch.bool).share_memory_()
        self.depth_scale = torch.zeros(buffer,device=self.device, dtype=torch.float).share_memory_()
        self.depth_shift = torch.zeros(buffer,device=self.device, dtype=torch.float).share_memory_()
        self.valid_depth_mask = torch.zeros(buffer, ht, wd, device=self.device, dtype=torch.bool).share_memory_()
        self.valid_depth_mask_small = torch.zeros(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.bool).share_memory_()        
        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, 1, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//self.down_scale, wd//self.down_scale, dtype=torch.half, device=self.device).share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=self.device)
        self.printer = printer
        self.uncer_network = uncer_network
        
        self.uncertainty_aware = (
            cfg['tracking']["uncertainty_params"]['activate']
            or cfg['mapping']["uncertainty_params"]['activate']
        )
        self.feature_output_dir = os.path.join(self.output, "mono_priors", "features")
        os.makedirs(self.feature_output_dir, exist_ok=True)
        if self.uncertainty_aware:
            n_features = self.cfg["mapping"]["uncertainty_params"]['feature_dim']
            
            # The followings are in cpu to save memory
            self.dino_feats = torch.zeros(
                buffer,
                ht // self.feature_stride,
                wd // self.feature_stride,
                n_features,
                device='cpu',
                dtype=torch.float,
            ).share_memory_()
            self.dino_feats_resize = torch.zeros(buffer, n_features, ht//self.down_scale, wd//self.down_scale, device='cpu', dtype=torch.float).share_memory_()
            self.dino_feats_valid = torch.zeros(buffer, dtype=torch.bool, device='cpu').share_memory_()
            self.uncertainties_inv = torch.ones(buffer, ht//self.down_scale, wd//self.down_scale, device=self.device, dtype=torch.float).share_memory_()
        else:
            self.dino_feats = None
            self.dino_feats_resize = None
            self.dino_feats_valid = None

    def export_keyframe_snapshot(self, index: int) -> Dict[str, Any]:
        with self.get_lock():
            if index < 0 or index >= self.counter.value:
                raise IndexError(f"Keyframe index out of range: {index}")
            dino_feature = None
            if (
                self.dino_feats is not None
                and self.dino_feats_valid is not None
                and bool(self.dino_feats_valid[index].item())
            ):
                dino_feature = self.dino_feats[index].detach().cpu().clone()
            mono_depth = torch.where(
                self.mono_disps_up[index] > 0,
                1.0 / self.mono_disps_up[index],
                0.0,
            ).detach().cpu().clone()
            return {
                "timestamp": self.timestamp[index].detach().cpu().clone(),
                "frame_id": int(self.frame_ids[index].item()),
                "image": self.images[index].detach().cpu().clone(),
                "pose": self.poses[index].detach().cpu().clone(),
                "disp": self.disps[index].detach().cpu().clone(),
                "mono_depth": mono_depth,
                "intrinsic": self.intrinsics[index].detach().cpu().clone(),
                "fmap": self.fmaps[index].detach().cpu().clone(),
                "net": self.nets[index].detach().cpu().clone(),
                "inp": self.inps[index].detach().cpu().clone(),
                "dino_feature": dino_feature,
            }

    def _frame_id_to_index(self, frame_id: int) -> Optional[int]:
        if frame_id is None:
            return None
        frame_id = int(frame_id)
        with self.get_lock():
            if self.counter.value == 0:
                return None
            matches = torch.where(self.frame_ids[: self.counter.value] == frame_id)[0]
            if matches.numel() == 0:
                return None
            return int(matches[0].item())

    def _copy_dino_feature_if_available(
        self, index: int, frame_id: Optional[int] = None, resized: bool = True
    ) -> Optional[torch.Tensor]:
        if self.dino_feats is None or self.dino_feats_valid is None:
            return None
        if frame_id is not None:
            resolved = self._frame_id_to_index(frame_id)
            if resolved is not None:
                index = resolved
        with self.get_lock():
            if index < 0 or index >= self.counter.value:
                return None
            if not bool(self.dino_feats_valid[index].item()):
                return None
            if resized:
                return self.dino_feats_resize[index].clone()
            return self.dino_feats[index].clone()

    def wait_for_dino_feature(
        self,
        index: int,
        frame_id: Optional[int] = None,
        resized: bool = True,
        timeout_s: float = 30.0,
        poll_interval_s: float = 0.01,
    ) -> torch.Tensor:
        if self.dino_feats is None or self.dino_feats_valid is None:
            raise RuntimeError("DINO feature buffers are not enabled")
        deadline = time.time() + float(timeout_s)
        while True:
            feat = self._copy_dino_feature_if_available(
                index, frame_id=frame_id, resized=resized
            )
            if feat is not None:
                return feat.to(self.device)
            if time.time() > deadline:
                ident = f"index={index}"
                if frame_id is not None:
                    ident += f", frame_id={frame_id}"
                raise RuntimeError(f"Missing DINOv3 feature for {ident}")
            time.sleep(poll_interval_s)

    def _store_dino_feature(self, index: int, feature: torch.Tensor, frame_id: Optional[int] = None) -> None:
        if feature is None:
            return
        if self.dino_feats is None or self.dino_feats_resize is None or self.dino_feats_valid is None:
            raise RuntimeError("DINO feature buffers are not enabled")
        if torch.is_tensor(feature):
            feat = feature.detach().cpu().float()
        else:
            feat = torch.as_tensor(np.asarray(feature), dtype=torch.float32)
        if feat.dim() == 4 and feat.shape[0] == 1:
            feat = feat[0]
        if feat.dim() != 3:
            raise ValueError(f"Unsupported DINO feature shape: {feat.shape}")
        feat_dim = self.dino_feats.shape[-1]
        if feat.shape[-1] == feat_dim:
            feat_hwc = feat.contiguous()
            feat_chw = feat_hwc.permute(2, 0, 1).contiguous()
        elif feat.shape[0] == feat_dim:
            feat_chw = feat.contiguous()
            feat_hwc = feat_chw.permute(1, 2, 0).contiguous()
        else:
            raise ValueError(
                f"Feature channel mismatch: got {feat.shape}, expected channel {feat_dim}"
            )

        resized = F.interpolate(
            feat_chw.unsqueeze(0),
            self.disps_up.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        self.dino_feats[index] = feat_hwc
        self.dino_feats_resize[index] = resized.squeeze(0)[:, self.slice_h, self.slice_w].cpu()
        self.dino_feats_valid[index] = True
        if frame_id is None:
            frame_id = int(self.frame_ids[index].item())
        if frame_id >= 0:
            np.save(
                os.path.join(self.feature_output_dir, f"{int(frame_id):05d}.npy"),
                feat_hwc.numpy(),
            )

    def set_external_dino_feature(self, index: int, feature: torch.Tensor, frame_id: Optional[int] = None) -> None:
        if feature is None:
            return
        if frame_id is not None:
            resolved = self._frame_id_to_index(frame_id)
            if resolved is not None:
                index = resolved
            elif index >= self.counter.value or int(self.frame_ids[index].item()) != int(frame_id):
                return
        self._store_dino_feature(index, feature, frame_id=frame_id)

    def _compute_uncertainty_full_res(self, idxs, train_frac: float) -> torch.Tensor:
        h = self.images.shape[2]
        w = self.images.shape[3]
        data_rate = 1 + 1 * map_utils.compute_bias_factor(train_frac, 0.8)
        network_device = next(self.uncer_network.parameters()).device
        if isinstance(idxs, slice):
            start = 0 if idxs.start is None else int(idxs.start)
            stop = self.counter.value if idxs.stop is None else int(idxs.stop)
            step = 1 if idxs.step is None else int(idxs.step)
            idx_list = list(range(start, stop, step))
        elif torch.is_tensor(idxs):
            idx_list = [int(x) for x in idxs.detach().cpu().flatten().tolist()]
        elif isinstance(idxs, int):
            idx_list = [int(idxs)]
        else:
            idx_list = [int(x) for x in idxs]

        if len(idx_list) == 0:
            return torch.empty(0, h, w, device=self.device)

        feature_batch = []
        ready_idxs = []
        for idx in idx_list:
            feature = self.get_dino_feature(idx, resized=True)
            if feature is None:
                continue
            ready_idxs.append(idx)
            feature_batch.append(feature)
        if not ready_idxs:
            return torch.empty(0, h, w, device=self.device)
        with Lock():
            uncer = self.uncer_network(
                torch.stack(feature_batch, dim=0).to(network_device)
            ).to(self.device)
        uncer = torch.clip(uncer, min=0.1) + 1e-3
        uncer = uncer.unsqueeze(1)
        uncer = F.interpolate(uncer, size=(h, w), mode="bilinear").squeeze(1).detach()
        data_rate = 1 + 1 * map_utils.compute_bias_factor(train_frac, 0.8)
        uncer = uncer[:, self.slice_h, self.slice_w]
        uncer = (uncer - 0.1) * data_rate + 0.1
        return uncer

    def has_dino_feature(self, index: int) -> bool:
        if self.dino_feats_valid is None:
            return False
        return bool(self.dino_feats_valid[index].item())

    def get_dino_feature(self, index: int, resized: bool = True, frame_id: Optional[int] = None) -> Optional[torch.Tensor]:
        if frame_id is not None:
            idx = self._frame_id_to_index(frame_id)
            if idx is not None and self.dino_feats_valid is not None and self.dino_feats_valid[idx]:
                if resized:
                    return self.dino_feats_resize[idx].to(self.device).permute(1, 2, 0).contiguous()
                return self.dino_feats[idx].to(self.device)
        if self.dino_feats is None or self.dino_feats_valid is None:
            return None
        if not self.dino_feats_valid[index]:
            return None
        if resized:
            return self.dino_feats_resize[index].to(self.device).permute(1, 2, 0).contiguous()
        return self.dino_feats[index].to(self.device)

    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, slice):
            start = 0 if index.start is None else int(index.start)
            stop = self.counter.value if index.stop is None else int(index.stop)
            step = 1 if index.step is None else int(index.step)
            indices = list(range(start, stop, step))
        elif isinstance(index, torch.Tensor):
            indices = [int(x) for x in index.detach().cpu().flatten().tolist()]
        elif isinstance(index, int):
            indices = [int(index)]
        else:
            indices = [int(x) for x in index]

        def _as_list(value):
            if value is None:
                return [None] * len(indices)
            if torch.is_tensor(value):
                if value.dim() == 0:
                    return [value] * len(indices)
                if value.shape[0] == len(indices):
                    return [value[i] for i in range(len(indices))]
                return [value] * len(indices)
            if isinstance(value, (list, tuple)):
                if len(value) == len(indices):
                    return list(value)
                return [value] * len(indices)
            return [value] * len(indices)

        items = [ _as_list(part) for part in item ]

        for local_i, idx in enumerate(indices):
            if idx >= self.counter.value:
                self.counter.value = idx + 1

            # Clear stale external caches before reusing this slot.
            if self.uncertainty_aware:
                if self.dino_feats_valid is not None:
                    self.dino_feats_valid[idx] = False
                    self.dino_feats[idx].zero_()
                    self.dino_feats_resize[idx].zero_()
            timestamp = items[0][local_i]
            image = items[1][local_i]
            pose = items[2][local_i]
            disp = items[3][local_i]
            mono_depth = items[4][local_i]
            intrinsic = items[5][local_i]
            fmap = items[6][local_i] if len(items) > 6 else None
            net = items[7][local_i] if len(items) > 7 else None
            inp = items[8][local_i] if len(items) > 8 else None
            dino_feature = items[9][local_i] if len(items) > 9 else None

            def _as_tensor(value, *, dtype=None):
                if value is None:
                    return None
                if torch.is_tensor(value):
                    tensor = value
                else:
                    tensor = torch.as_tensor(value)
                if dtype is not None:
                    tensor = tensor.to(dtype=dtype)
                return tensor

            if torch.is_tensor(timestamp):
                self.timestamp[idx] = timestamp.to(self.device)
            else:
                self.timestamp[idx] = torch.as_tensor(timestamp, device=self.device)
            self.frame_ids[idx] = int(timestamp.item()) if torch.is_tensor(timestamp) else int(timestamp)
            self.images[idx] = _as_tensor(image, dtype=self.images.dtype).cpu()

            if pose is not None:
                self.poses[idx] = _as_tensor(pose, dtype=self.poses.dtype).to(self.device)

            if disp is not None:
                self.disps[idx] = _as_tensor(disp, dtype=self.disps.dtype).to(self.device)

            if mono_depth is not None:
                mono_depth = _as_tensor(mono_depth, dtype=torch.float32).to(self.device)
                mono_depth = mono_depth[self.slice_h, self.slice_w]
                self.mono_disps[idx] = torch.where(mono_depth > 0, 1.0 / mono_depth, 0)
                mono_depth_up = _as_tensor(items[4][local_i], dtype=torch.float32).to(self.device)
                self.mono_disps_up[idx] = torch.where(mono_depth_up > 0, 1.0 / mono_depth_up, 0)

            if intrinsic is not None:
                self.intrinsics[idx] = _as_tensor(intrinsic, dtype=self.intrinsics.dtype).to(self.device)

            if fmap is not None:
                self.fmaps[idx] = _as_tensor(fmap, dtype=self.fmaps.dtype).to(self.device)

            if net is not None:
                self.nets[idx] = _as_tensor(net, dtype=self.nets.dtype).to(self.device)

            if inp is not None:
                self.inps[idx] = _as_tensor(inp, dtype=self.inps.dtype).to(self.device)

            if dino_feature is not None:
                self._store_dino_feature(idx, dino_feature, frame_id=int(self.frame_ids[idx].item()))

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)


    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj, device=None):
        """ to device, long, {-1} """
        device = "cuda" if device is None else device

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device=device, dtype=torch.long).reshape(-1)
        jj = jj.to(device=device, dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value,:3] *= s
            self.set_dirty(0,self.counter.value)


    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj, device=self.device)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N),indexing="ij")
        
        ii, jj = DepthVideo.format_indicies(ii, jj, device=self.device)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d
    
    def project_images_with_mask(self, images, pixel_positions, masks=None):
        """ 
            Project images/depths from the input pixel positions using bilinear interpolation.
            This function will automatically return the mask where the given pixel positions are out of the images
        Args:
            images (torch.Tensor): A tensor of shape [B, C, H, W] representing the images/depths.
            pixel_positions (torch.Tensor): A tensor of shape [B, H, W, 2] containing float 
                                            pixel positions for interpolation. Note that [:,:,:,0]
                                            is width and [:,:,:,1] is height.
            masks (torch.Tensor, optional): A boolean tensor of shape [B, H, W]. If provided, 
                                            specifies valid pixels. Default is None, which 
                                            results in all pixels being valid at the begining.
        
        Returns:
            torch.Tensor: A tensor of shape [B, C, H, W] containing the projected images/depths, 
                        where invalid pixels are set to 0.
            torch.Tensor: The combined mask that filters out invalid positions and applies
                      the original mask.
        """
        B, C, H, W = images.shape
        device = images.device

        # If masks are not provided, create a mask of all ones (True) with the same shape as the images
        if masks is None:
            masks = torch.ones(B, H, W, dtype=torch.bool, device=device)
        
        # Normalize pixel positions to range [-1, 1]
        grid = pixel_positions.clone()
        grid[..., 0] = 2.0 * (grid[..., 0] / (W - 1)) - 1.0
        grid[..., 1] = 2.0 * (grid[..., 1] / (H - 1)) - 1.0

        projected_image = F.grid_sample(images, grid, mode='bilinear', align_corners=True)

        # Mask out invalid positions where x or y are out of bounds and combine it with the initial mask
        valid_mask = (pixel_positions[..., 0] >= 0) & (pixel_positions[..., 0] < W) & \
                    (pixel_positions[..., 1] >= 0) & (pixel_positions[..., 1] < H)
        valid_mask &= masks

        # Apply the combined mask: set to 0 where combined mask is False
        projected_image = projected_image.permute(0, 2, 3, 1)  # conver to [B, H, W, C]
        projected_image = projected_image * valid_mask.unsqueeze(-1)
        
        return projected_image.permute(0, 3, 1, 2), valid_mask  # Return to [B, C, H, W]

    @torch.no_grad()
    def filter_high_err_mono_depth(self, idx, ii, jj):
        nb_frame = self.cfg['tracking']['nb_ref_frame_metric_depth_filtering']

        jj = jj[ii==idx]
        for j in torch.arange(idx-1, max(0,idx-nb_frame)-1, -1):
            if jj.shape[0] >= nb_frame:
                break
            if j not in jj:
                torch.cat((jj, j.unsqueeze(0).to(jj.device)))

        ii = torch.tensor(idx).repeat(jj.shape[0])

        # all frames share the same intrinsics
        X0, _ = pops.iproj(self.mono_disps_up[jj].unsqueeze(0), 
                      self.intrinsics[0].unsqueeze(0).repeat(1,jj.shape[0],1)*self.down_scale, 
                      jacobian=False)
        Gs = lietorch.SE3(self.poses[None])
        Gji = Gs[:,ii] * Gs[:,jj].inv()
        X1, _ = pops.actp(Gji, X0, jacobian=False)
        x1, _ = pops.proj(X1, self.intrinsics[0].unsqueeze(0).repeat(1,jj.shape[0],1)*self.down_scale, jacobian=False, return_depth=True)

        i_disp = self.mono_disps_up[idx]
        accurate_count = torch.zeros_like(i_disp)
        inaccurate_count = torch.zeros_like(i_disp)

        x1_rounded = torch.round(x1[..., :2]).long()
        # x1 is the 3d poisition (x,y,z)
        # projected point is valid only if its inside the image range and the depth is greater than 0
        valid_mask = (x1_rounded[..., 1] >= 0) & (x1_rounded[..., 1] < x1.shape[2]) & \
                    (x1_rounded[..., 0] >= 0) & (x1_rounded[..., 0] < x1.shape[3]) & (x1[...,2]>0)
        
        i_dino = F.interpolate(self.dino_feats[idx].permute(2,0,1).unsqueeze(0),
                                self.disps_up.shape[-2:], 
                                mode='bilinear').to(self.device).squeeze()
        for j_id in range(jj.shape[0]):
            projected_j_to_i = x1[0, j_id]
            x_coords, y_coords = x1_rounded[0, j_id, ..., 0], x1_rounded[0, j_id, ..., 1]
            
            # Select valid coordinates and their Dino features
            j_dino = F.interpolate(self.dino_feats[jj[j_id]].permute(2,0,1).unsqueeze(0),
                                    self.disps_up.shape[-2:], 
                                    mode='bilinear').to(self.device).squeeze()
            valid_x, valid_y = x_coords[valid_mask[0, j_id]], y_coords[valid_mask[0, j_id]]
            j_dino_valid = j_dino[:, valid_mask[0, j_id]]
            i_dino_valid = i_dino[:, valid_y, valid_x]

            # Compute cosine similarity for each valid position
            similarity = F.normalize(j_dino_valid, p=2, dim=0).mul(F.normalize(i_dino_valid, p=2, dim=0)).sum(dim=0)
            matching_mask = similarity > 0.9

            # Update projected disparity and counts based on the similarity check
            j_projected_disp = torch.zeros_like(self.mono_disps_up[idx])
            matched_disp = projected_j_to_i[valid_mask[0, j_id]][matching_mask]
            matched_x, matched_y = valid_x[matching_mask], valid_y[matching_mask]
            j_projected_disp[matched_y, matched_x] = matched_disp[..., 2]

            # Error calculation and count updates
            error = torch.abs(1 / j_projected_disp[matched_y, matched_x] - 1 / i_disp[matched_y, matched_x]) * j_projected_disp[matched_y, matched_x]
            correct_mask = error < 0.02

            # Batch update correct and bad counts
            accurate_count[matched_y[correct_mask], matched_x[correct_mask]] += 1
            inaccurate_count[matched_y[~correct_mask], matched_x[~correct_mask]] += 1

        # Clean the gpu memory
        torch.cuda.empty_cache()

        self.mono_disps_mask_up[idx][(accurate_count<=1)&(inaccurate_count>0)&(self.mono_disps_up[idx]>0)] = False

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, iters=2, lm=1e-4, ep=0.1,
           motion_only=False):
        if self.uncertainty_aware:
            weight *= self.uncertainties_inv[ii][None, :, :, :, None]

        with self.get_lock():
            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            target = target.view(-1, self.ht//self.down_scale, self.wd//self.down_scale, 2).permute(0,3,1,2).contiguous()
            weight = weight.view(-1, self.ht//self.down_scale, self.wd//self.down_scale, 2).permute(0,3,1,2).contiguous()

            if not self.metric_depth_reg:
                droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.zeros,
                    target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only, False)
            else:
                mono_valid_mask = self.mono_disps_mask_up[:,self.slice_h,self.slice_w].clone().to(self.device)
                
                droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.mono_disps*mono_valid_mask,
                    target, weight, eta, ii, jj, t0, t1, iters, lm, ep, motion_only, False)
            
            self.disps.clamp_(min=1e-5)


    def get_depth_scale_and_shift(self,index, mono_depth:torch.Tensor, est_depth:torch.Tensor, weights:torch.Tensor):
        '''
        index: int
        mono_depth: [B,H,W]
        est_depth: [B,H,W]
        weights: [B,H,W]
        '''
        scale,shift,_ = align_scale_and_shift(mono_depth,est_depth,weights)
        self.depth_scale[index] = scale
        self.depth_shift[index] = shift
        return [self.depth_scale[index], self.depth_shift[index]]

    def get_pose(self,index,device):
        w2c = lietorch.SE3(self.poses[index].clone()).to(device) # Tw(droid)_to_c
        c2w = w2c.inv().matrix()  # [4, 4]
        return c2w

    def get_depth_and_pose(self,index,device):
        with self.get_lock():
            if self.metric_depth_reg:
                est_disp = self.mono_disps_up[index].clone().to(device)  # [h, w]
                est_depth = torch.where(est_disp>0.0, 1.0 / (est_disp), 0.0)
                depth_mask = torch.ones_like(est_disp,dtype=torch.bool).to(device)
                c2w = self.get_pose(index,device)
            else:
                est_disp = self.disps_up[index].clone().to(device)  # [h, w]
                est_depth = 1.0 / (est_disp)
                depth_mask = self.valid_depth_mask[index].clone().to(device)
                c2w = self.get_pose(index,device)
        return est_depth, depth_mask, c2w
    
    @torch.no_grad()
    def update_valid_depth_mask(self,up=True):
        '''
        For each pixel, check whether the estimated depth value is valid or not 
        by the two-view consistency check, see eq.4 ~ eq.7 in the paper for details

        up (bool): if True, check on the orignial-scale depth map
                   if False, check on the downsampled depth map
        '''
        if up:
            with self.get_lock():
                dirty_index, = torch.where(self.dirty.clone())
            if len(dirty_index) == 0:
                return
        else:
            curr_idx = self.counter.value-1
            dirty_index = torch.arange(curr_idx+1).to(self.device)
        # convert poses to 4x4 matrix
        disps = torch.index_select(self.disps_up if up else self.disps, 0, dirty_index)
        common_intrinsic_id = 0  # we assume the intrinsics are the same within one scene
        intrinsic = self.intrinsics[common_intrinsic_id].detach() * (self.down_scale if up else 1.0)
        depths = 1.0/disps
        thresh = self.cfg['tracking']['multiview_filter']['thresh'] * depths.mean(dim=[1,2]) 
        count = droid_backends.depth_filter(
            self.poses, self.disps_up if up else self.disps, intrinsic, dirty_index, thresh)
        filter_visible_num = self.cfg['tracking']['multiview_filter']['visible_num']
        multiview_masks = (count >= filter_visible_num) 
        depths[~multiview_masks]=torch.nan
        depths_reshape = depths.view(depths.shape[0],-1)
        depths_median = depths_reshape.nanmedian(dim=1).values
        masks = depths < 3*depths_median[:,None,None]
        if up:
            self.valid_depth_mask[dirty_index] = masks 
            self.dirty[dirty_index] = False
        else:
            self.valid_depth_mask_small[dirty_index] = masks 

    @torch.no_grad()
    def update_all_uncertainty_mask(self):
        if not self.uncertainty_aware:
            # we only estimate uncertainty when we activate the mode
            raise Exception('This function should not be called if uncertainty aware is not activated')
        
        i = 0
        while i*20 < self.counter.value:
            idxs = list(range(i * 20, min((i + 1) * 20, self.counter.value)))
            ready_idxs = []
            feature_batch = []
            for idx in idxs:
                feature = self.get_dino_feature(idx, resized=False)
                if feature is None:
                    continue
                ready_idxs.append(idx)
                feature_batch.append(feature)
            if not ready_idxs:
                i += 1
                continue
            idx_tensor = torch.as_tensor(ready_idxs, dtype=torch.long)
            network_device = next(self.uncer_network.parameters()).device
            train_frac = self.cfg['mapping']['uncertainty_params']['train_frac_fix']
            with Lock():
                uncer = self.uncer_network(
                    torch.stack(feature_batch, dim=0).to(network_device)
                ).to(self.device)
            uncer = torch.clip(uncer, min=0.1) + 1e-3
            uncer = uncer.unsqueeze(1)
            uncer = F.interpolate(
                uncer, size=(self.images.shape[2], self.images.shape[3]), mode="bilinear"
            ).squeeze(1).detach()
            data_rate = 1 + 1 * map_utils.compute_bias_factor(train_frac, 0.8)
            uncer = uncer[:, self.slice_h, self.slice_w]
            uncer = (uncer - 0.1) * data_rate + 0.1
            with self.get_lock():
                self.uncertainties_inv[idx_tensor, :, :] = torch.clamp(0.5/uncer**2, 0.0, 1.0)

            i += 1

    @torch.no_grad()
    def update_uncertainty_mask_given_index(self,idxs):
        if not self.uncertainty_aware:
            # we only estimate uncertainty when we activate the mode
            raise Exception('This function should not be called if uncertainty aware is not activated')
        
        if isinstance(idxs, slice):
            start = 0 if idxs.start is None else int(idxs.start)
            stop = self.counter.value if idxs.stop is None else int(idxs.stop)
            step = 1 if idxs.step is None else int(idxs.step)
            idx_list = list(range(start, stop, step))
        elif torch.is_tensor(idxs):
            idx_list = [int(x) for x in idxs.detach().cpu().flatten().tolist()]
        elif isinstance(idxs, int):
            idx_list = [int(idxs)]
        else:
            idx_list = [int(x) for x in idxs]

        ready_idxs = []
        feature_batch = []
        for idx in idx_list:
            feature = self.get_dino_feature(idx, resized=True)
            if feature is None:
                continue
            ready_idxs.append(idx)
            feature_batch.append(feature)
        if not ready_idxs:
            return

        idx_tensor = torch.as_tensor(ready_idxs, dtype=torch.long)
        network_device = next(self.uncer_network.parameters()).device
        train_frac = self.cfg['mapping']['uncertainty_params']['train_frac_fix']
        with Lock():
            uncer = self.uncer_network(
                torch.stack(feature_batch, dim=0).to(network_device)
            ).to(self.device)
        uncer = torch.clip(uncer, min=0.1) + 1e-3
        uncer = uncer.unsqueeze(1)
        uncer = F.interpolate(
            uncer, size=(self.images.shape[2], self.images.shape[3]), mode="bilinear"
        ).squeeze(1).detach()
        data_rate = 1 + 1 * map_utils.compute_bias_factor(train_frac, 0.8)
        uncer = uncer[:, self.slice_h, self.slice_w]
        uncer = (uncer - 0.1) * data_rate + 0.1
        with self.get_lock():
            self.uncertainties_inv[idx_tensor,:,:] = torch.clamp(0.5/uncer**2, 0.0, 1.0)

    def set_dirty(self,index_start, index_end):
        self.dirty[index_start:index_end] = True
        self.npc_dirty[index_start:index_end] = True

    def save_video(self,path:str):
        poses = []
        depths = []
        timestamps = []
        valid_depth_masks = []
        for i in range(self.counter.value):
            depth, depth_mask, pose = self.get_depth_and_pose(i,'cpu')
            timestamp = self.timestamp[i].cpu()
            poses.append(pose)
            depths.append(depth)
            timestamps.append(timestamp)
            valid_depth_masks.append(depth_mask)
        poses = torch.stack(poses,dim=0).numpy()
        depths = torch.stack(depths,dim=0).numpy()
        timestamps = torch.stack(timestamps,dim=0).numpy() 
        valid_depth_masks = torch.stack(valid_depth_masks,dim=0).numpy()       
        np.savez(path,poses=poses,depths=depths,timestamps=timestamps,valid_depth_masks=valid_depth_masks)
        self.printer.print(f"Saved final depth video: {path}",FontColor.INFO)


    def eval_depth_l1(self, npz_path, stream, global_scale=None):
        """This is from splat-slam, not used in WildGS-SLAM
        """
        # Compute Depth L1 error
        depth_l1_list = []
        depth_l1_list_max_4m = []
        mask_list = []

        # load from disk
        offline_video = dict(np.load(npz_path))
        video_timestamps = offline_video['timestamps']

        for i in range(video_timestamps.shape[0]):
            timestamp = int(video_timestamps[i])
            mask = self.valid_depth_mask[i]
            if mask.sum() == 0:
                print("WARNING: mask is empty!")
            mask_list.append((mask.sum()/(mask.shape[0]*mask.shape[1])).cpu().numpy())
            disparity = self.disps_up[i]
            depth = 1/(disparity)
            depth[mask == 0] = 0
            # compute scale and shift for depth
            # load gt depth from stream
            depth_gt = stream[timestamp][2].to(self.device)
            mask = torch.logical_and(depth_gt > 0, mask)
            if global_scale is None:
                scale, shift, _ = align_scale_and_shift(depth.unsqueeze(0), depth_gt.unsqueeze(0), mask.unsqueeze(0))
                depth = scale*depth + shift
            else:
                depth = global_scale * depth
            diff_depth_l1 = torch.abs((depth[mask] - depth_gt[mask]))
            depth_l1 = diff_depth_l1.sum() / (mask).sum()
            depth_l1_list.append(depth_l1.cpu().numpy())

            # update process but masking depth_gt > 4
            # compute scale and shift for depth
            mask = torch.logical_and(depth_gt < 4, mask)
            disparity = self.disps_up[i]
            depth = 1/(disparity)
            depth[mask == 0] = 0
            if global_scale is None:
                scale, shift, _ = align_scale_and_shift(depth.unsqueeze(0), depth_gt.unsqueeze(0), mask.unsqueeze(0))
                depth = scale*depth + shift
            else:
                depth = global_scale * depth
            diff_depth_l1 = torch.abs((depth[mask] - depth_gt[mask]))
            depth_l1 = diff_depth_l1.sum() / (mask).sum()
            depth_l1_list_max_4m.append(depth_l1.cpu().numpy())

        return np.asarray(depth_l1_list).mean(), np.asarray(depth_l1_list_max_4m).mean(), np.asarray(mask_list).mean()
