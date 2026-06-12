import json
import os
import shutil
import torch
import numpy as np
import time
from collections import OrderedDict
import torch.multiprocessing as mp
from munch import munchify

from src.modules.droid_net import DroidNet
from src.depth_video import DepthVideo
from src.trajectory_filler import PoseTrajectoryFiller
from src.utils.common import setup_seed, update_cam
from src.utils.Printer import Printer, FontColor
from src.utils.eval_traj import kf_traj_eval, full_traj_eval
from src.utils.datasets import BaseDataset
from src.tracker import Tracker
from src.mapper import Mapper
from src.backend import Backend
from src.utils.datasets import RGB_NoPose
from src.utils.dyn_uncertainty.uncertainty_model import generate_uncertainty_mlp
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from src.utils.external_beta import ExternalDinoClient

class SLAM:
    def __init__(self, cfg, stream: BaseDataset):
        super(SLAM, self).__init__()
        self.cfg = cfg
        self.tracking_device = cfg.get("tracking_device", cfg["device"])
        self.mapping_device = cfg.get("mapping_device", self.tracking_device)
        self.device = self.tracking_device
        self.verbose: bool = cfg["verbose"]
        self.logger = None
        self.save_dir = cfg["data"]["output"] + "/" + cfg["scene"]
        self.run_log_dir = os.environ.get("RUN_LOG_DIR")
        self.run_log_scene_dir = (
            os.path.join(self.run_log_dir, cfg["scene"])
            if self.run_log_dir
            else None
        )

        os.makedirs(self.save_dir, exist_ok=True)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = update_cam(cfg)

        self.droid_net: DroidNet = DroidNet()

        self.printer = Printer(
            len(stream)
        )  # use an additional process for printing all the info

        self.load_pretrained(cfg)
        self.droid_net.to(self.device).eval()
        self.droid_net.share_memory()

        self.num_running_thread = torch.zeros((1)).int()
        self.num_running_thread.share_memory_()
        self.all_trigered = torch.zeros((1)).int()
        self.all_trigered.share_memory_()

        self.uncer_network = None
        if self.cfg["mapping"]["uncertainty_params"]["activate"]:
            n_features = self.cfg["mapping"]["uncertainty_params"]["feature_dim"]
            latent_dim = int(self.cfg["mapping"]["uncertainty_params"].get("latent_dim", 3))
            hidden_dim = int(self.cfg["mapping"]["uncertainty_params"].get("hidden_dim", 128))
            net_depth = int(self.cfg["mapping"]["uncertainty_params"].get("net_depth", 2))
            self.uncer_network = generate_uncertainty_mlp(
                n_features,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                net_depth=net_depth,
            )
            self.uncer_network.share_memory()
            self.uncer_network.eval()

        self._load_uncertainty_checkpoint()

        self.beta_cfg = cfg.get("beta_service", {})
        self.beta_client = None
        self.keyframe_queue = None
        self.video = DepthVideo(
            cfg,
            self.printer,
            uncer_network=self.uncer_network,
            beta_client=None,
        )
        self.ba = Backend(self.droid_net, self.video, self.cfg)

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(
            cfg=cfg,
            net=self.droid_net,
            video=self.video,
            printer=self.printer,
            device=self.device,
        )

        self.tracker: Tracker = None
        self.mapper: Mapper = None
        self.stream = stream

    def _start_beta_client(self):
        if self.uncer_network is not None:
            return
        if not self.beta_cfg.get("activate", False):
            return
        if self.beta_client is not None:
            return

        self.beta_client = ExternalDinoClient(
            host=self.beta_cfg.get("host", "127.0.0.1"),
            port=self.beta_cfg.get("port", 5555),
            authkey=self.beta_cfg.get("authkey", "wildgs-beta"),
            jpeg_quality=self.beta_cfg.get("jpeg_quality", 90),
            max_request_queue=self.beta_cfg.get("max_request_queue", 4096),
            max_result_queue=self.beta_cfg.get("max_result_queue", 1024),
        )
        self.beta_client.on_result = self._handle_beta_result
        self.video.set_beta_client(self.beta_client)
        self.beta_client.start()

    def _shutdown_beta_client(self):
        if self.beta_client is None:
            return
        self.beta_client.shutdown()
        self.beta_client = None
        self.video.set_beta_client(None)

    def _uncertainty_ckpt_path(self) -> str:
        return os.path.join(self.save_dir, "uncertainty_student.pth")

    @staticmethod
    def _filter_legacy_uncertainty_state(state: dict, model: torch.nn.Module) -> dict:
        model_state = model.state_dict()
        filtered = {}
        dropped = []
        for key, value in state.items():
            if key in model_state and model_state[key].shape == value.shape:
                filtered[key] = value
            else:
                dropped.append(key)
        return filtered, dropped

    @staticmethod
    def _safe_map_location(device):
        if torch.cuda.is_available():
            return device
        return "cpu"

    def _load_uncertainty_checkpoint(self) -> None:
        if self.uncer_network is None:
            return
        ckpt_path = self._uncertainty_ckpt_path()
        if os.path.exists(ckpt_path):
            state = torch.load(
                ckpt_path,
                map_location=self._safe_map_location(self.device),
                weights_only=True,
            )
            if isinstance(state, dict):
                filtered, dropped = self._filter_legacy_uncertainty_state(
                    state, self.uncer_network
                )
                self.uncer_network.load_state_dict(filtered, strict=False)
                if dropped:
                    self.printer.print(
                        f"Loaded uncertainty student checkpoint from {ckpt_path} with filtered keys: {dropped}",
                        FontColor.INFO,
                    )
                else:
                    self.printer.print(
                        f"Loaded uncertainty student checkpoint from {ckpt_path}",
                        FontColor.INFO,
                    )

    def _load_tracker_stats(self):
        candidates = [os.path.join(self.save_dir, "tracker_metrics.json")]
        if self.run_log_scene_dir is not None:
            candidates.append(os.path.join(self.run_log_scene_dir, "tracker_metrics.json"))

        for path in candidates:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fp:
                    return json.load(fp)
        return None

    def _write_text_artifact(self, rel_path: str, text: str) -> None:
        targets = [os.path.join(self.save_dir, rel_path)]
        if self.run_log_scene_dir is not None:
            targets.append(os.path.join(self.run_log_scene_dir, rel_path))

        for path in targets:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fp:
                fp.write(text)

    def _mirror_run_artifacts(self) -> None:
        if self.run_log_scene_dir is None:
            return

        os.makedirs(self.run_log_scene_dir, exist_ok=True)
        for rel_dir in ("traj", "plots_before_refine", "plots_after_refine"):
            src = os.path.join(self.save_dir, rel_dir)
            if os.path.isdir(src):
                shutil.copytree(
                    src,
                    os.path.join(self.run_log_scene_dir, rel_dir),
                    dirs_exist_ok=True,
                )

        for rel_file in ("tracker_metrics.json", "evaluation_summary.txt", "cfg.yaml"):
            src = os.path.join(self.save_dir, rel_file)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(self.run_log_scene_dir, rel_file))

    @staticmethod
    def _format_eval_section(title: str, scale, rotation, translation, stats) -> str:
        lines = [f"##########{title}##########"]
        if scale is not None:
            lines.append(f"scale: {scale}")
        if rotation is not None:
            lines.append(f"rotation:\n{rotation}")
        if translation is not None:
            lines.append(f"translation:{translation}")
        if stats is not None:
            lines.append(f"statistics:\n{stats}")
        lines.append("#" * 34)
        return "\n".join(lines)

    def _handle_beta_result(self, result):
        status = result.get("status", "ok")
        frame_id = result.get("frame_id", None)
        video_idx = result.get("video_idx", None)
        kf_seq = result.get("kf_seq", None)
        if status != "ok":
            self.printer.print(
                f"DINOv3 beta service error for frame {frame_id}: {result.get('error', 'unknown')}",
                FontColor.ERROR,
            )
            if self.keyframe_queue is not None and kf_seq is not None:
                self.keyframe_queue.put(
                    {
                        "type": "beta_update",
                        "kf_seq": int(kf_seq),
                        "frame_id": int(frame_id) if frame_id is not None else None,
                        "video_idx": int(video_idx) if video_idx is not None else None,
                        "status": status,
                        "error": result.get("error", "unknown"),
                        "beta": None,
                        "features": None,
                    }
                )
            return
        if frame_id is None or video_idx is None:
            self.printer.print(
                "DINOv3 beta result missing frame_id or video_idx",
                FontColor.ERROR,
            )
            return
        beta = result.get("beta", None)
        features = result.get("features", None)
        if features is not None:
            self.video.set_external_dino_feature(
                int(video_idx),
                features,
                frame_id=int(frame_id),
            )
        if beta is not None:
            self.video.set_external_beta(
                int(video_idx),
                beta,
                frame_id=int(frame_id),
            )
        if self.keyframe_queue is not None and kf_seq is not None:
            self.keyframe_queue.put(
                {
                    "type": "beta_update",
                    "kf_seq": int(kf_seq),
                    "frame_id": int(frame_id) if frame_id is not None else None,
                    "video_idx": int(video_idx) if video_idx is not None else None,
                    "beta": beta,
                    "features": features,
                }
            )

    def load_pretrained(self, cfg):
        droid_pretrained = cfg["tracking"]["pretrained"]
        map_location = self._safe_map_location(self.device)
        state_dict = OrderedDict(
            [
                (k.replace("module.", ""), v)
                for (k, v) in torch.load(
                    droid_pretrained,
                    map_location=map_location,
                    weights_only=True,
                ).items()
            ]
        )
        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]
        self.droid_net.load_state_dict(state_dict)
        self.droid_net.eval()
        self.printer.print(
            f"Load droid pretrained checkpoint from {droid_pretrained}!", FontColor.INFO
        )

    def tracking(self, keyframe_queue):
        self.keyframe_queue = keyframe_queue
        self.cfg["device"] = self.tracking_device
        self.device = self.tracking_device
        torch.cuda.set_device(self.device)
        self._start_beta_client()
        self.tracker = Tracker(self, keyframe_queue)
        self.printer.print("Tracking Triggered!", FontColor.TRACKER)
        self.all_trigered += 1

        os.makedirs(f"{self.save_dir}/mono_priors/depths", exist_ok=True)
        os.makedirs(f"{self.save_dir}/mono_priors/features", exist_ok=True)

        while self.all_trigered < self.num_running_thread:
            pass
        self.printer.print("Tracking Starts!", FontColor.TRACKER)
        self.printer.pbar_ready()
        try:
            self.tracker.run(self.stream)
        finally:
            self._shutdown_beta_client()
        self.printer.print("Tracking Done!", FontColor.TRACKER)

    def mapping(self, packet_queue, q_main2vis, q_vis2main):
        self.cfg["device"] = self.mapping_device
        self.device = self.mapping_device
        torch.cuda.set_device(self.device)
        self.droid_net = self.droid_net.to(self.device).eval()
        self.video = DepthVideo(
            self.cfg,
            self.printer,
            uncer_network=self.uncer_network,
            beta_client=None,
        )
        self.ba = Backend(self.droid_net, self.video, self.cfg)
        self.traj_filler = PoseTrajectoryFiller(
            cfg=self.cfg,
            net=self.droid_net,
            video=self.video,
            printer=self.printer,
            device=self.device,
        )
        self.keyframe_queue = packet_queue
        self.mapper = Mapper(self, packet_queue, q_main2vis, q_vis2main, self.uncer_network)
        self.printer.print("Mapping Triggered!", FontColor.MAPPER)

        self.all_trigered += 1
        setup_seed(self.cfg["setup_seed"])

        while self.all_trigered < self.num_running_thread:
            pass
        self.printer.print("Mapping Starts!", FontColor.MAPPER)
        self.mapper.run()
        self.printer.print("Mapping Done!", FontColor.MAPPER)

        self.terminate()

    def backend(self):
        self.printer.print("Final Global BA Triggered!", FontColor.TRACKER)

        metric_depth_reg_activated = self.video.metric_depth_reg
        if metric_depth_reg_activated:
            self.video.metric_depth_reg = False

        self.ba = Backend(self.droid_net, self.video, self.cfg)
        torch.cuda.empty_cache()
        self.ba.dense_ba(7)
        torch.cuda.empty_cache()
        self.ba.dense_ba(12)
        self.printer.print("Final Global BA Done!", FontColor.TRACKER)

        if metric_depth_reg_activated:
            self.video.metric_depth_reg = True

    def terminate(self):
        """fill poses for non-keyframe images and evaluate"""
        tracker_stats = self._load_tracker_stats()
        summary_sections = []

        if (
            self.cfg["tracking"]["backend"]["final_ba"]
            and self.cfg["mapping"]["eval_before_final_ba"]
        ):
            self.video.save_video(f"{self.save_dir}/video.npz")
            if not isinstance(self.stream, RGB_NoPose):
                try:
                    ate_statistics, global_scale, r_a, t_a = kf_traj_eval(
                        f"{self.save_dir}/video.npz",
                        f"{self.save_dir}/traj/before_final_ba",
                        "kf_traj",
                        self.stream,
                        self.logger,
                        self.printer,
                    )
                    summary_sections.append(
                        self._format_eval_section(
                            "Keyframes traj (before final BA)",
                            global_scale,
                            r_a,
                            t_a,
                            ate_statistics,
                        )
                    )
                except Exception as e:
                    self.printer.print(e, FontColor.ERROR)

            self.mapper.save_all_kf_figs(
                self.save_dir,
                iteration="before_refine",
            )

        if self.cfg["tracking"]["backend"]["final_ba"]:
            self.backend()

        self.video.save_video(f"{self.save_dir}/video.npz")
        kf_eval_stats = None
        kf_scale = kf_rot = kf_trans = None
        if not isinstance(self.stream, RGB_NoPose):
            try:
                ate_statistics, global_scale, r_a, t_a = kf_traj_eval(
                    f"{self.save_dir}/video.npz",
                    f"{self.save_dir}/traj",
                    "kf_traj",
                    self.stream,
                    self.logger,
                    self.printer,
                )
                kf_eval_stats = ate_statistics
                kf_scale, kf_rot, kf_trans = global_scale, r_a, t_a
                summary_sections.append(
                    self._format_eval_section(
                        "Keyframes traj",
                        global_scale,
                        r_a,
                        t_a,
                        ate_statistics,
                    )
                )
            except Exception as e:
                self.printer.print(e, FontColor.ERROR)

        if self.cfg["tracking"]["backend"]["final_ba"]:
            self.mapper.final_refine(
                iters=self.cfg["mapping"]["final_refine_iters"]
            )  # this performs a set of optimizations with RGBD loss to correct

        # Evaluate the metrics
        self.mapper.save_all_kf_figs(
            self.save_dir,
            iteration="after_refine",
        )

        ## Not used, see head comments of the function
        # self._eval_depth_all(ate_statistics, global_scale, r_a, t_a)

        # Regenerate feature extractor for non-keyframes
        self.traj_filler.setup_feature_extractor()
        _, _, _, full_traj_output, full_scale, full_rot, full_trans, full_stats = full_traj_eval(
            self.traj_filler,
            self.mapper,
            f"{self.save_dir}/traj",
            "full_traj",
            self.stream,
            self.logger,
            self.printer,
            self.cfg['fast_mode'],
        )
        if full_traj_output is not None:
            self.printer.print(full_traj_output, FontColor.EVAL)
            self.printer.print("#"*29, FontColor.EVAL)
            summary_sections.append(
                self._format_eval_section(
                    "Full traj",
                    full_scale,
                    full_rot,
                    full_trans,
                    full_stats,
                )
            )

        self.mapper.gaussians.save_ply(f"{self.save_dir}/final_gs.ply")
        if self.uncer_network is not None:
            torch.save(self.uncer_network.state_dict(), self._uncertainty_ckpt_path())

        if tracker_stats is not None:
            tracker_section = ["##########Frontend summary##########"]
            for key in [
                "frames_total",
                "keyframes_total",
                "keyframe_ratio",
                "total_elapsed_s",
                "frontend_fps",
                "frontend_avg_ms",
                "frontend_fps_steady",
                "keyframe_avg_ms",
                "non_keyframe_avg_ms",
                "first_keyframe_latency_ms",
            ]:
                if key in tracker_stats:
                    tracker_section.append(f"{key}: {tracker_stats[key]}")
            tracker_section.append("#" * 34)
            summary_sections.insert(0, "\n".join(tracker_section))

        summary_text = "\n\n".join(summary_sections) if summary_sections else "No evaluation summary available."
        self._write_text_artifact("evaluation_summary.txt", summary_text)
        self.printer.print(
            f"Saved evaluation summary to {self.save_dir}/evaluation_summary.txt",
            FontColor.EVAL,
        )
        self._mirror_run_artifacts()

        if self.beta_client is not None:
            self.beta_client.shutdown()

        self.printer.print("Metrics Evaluation Done!", FontColor.EVAL)

    def _eval_depth_all(self, ate_statistics, global_scale, r_a, t_a):
        """From Splat-SLAM. Not used in WildGS-SLAM evaluation, but might be useful in the future."""
        # Evaluate depth error
        self.printer.print(
            "Evaluate sensor depth error with per frame alignment", FontColor.EVAL
        )
        depth_l1, depth_l1_max_4m, coverage = self.video.eval_depth_l1(
            f"{self.save_dir}/video.npz", self.stream
        )
        self.printer.print("Depth L1: " + str(depth_l1), FontColor.EVAL)
        self.printer.print("Depth L1 mask 4m: " + str(depth_l1_max_4m), FontColor.EVAL)
        self.printer.print("Average frame coverage: " + str(coverage), FontColor.EVAL)

        self.printer.print(
            "Evaluate sensor depth error with global alignment", FontColor.EVAL
        )
        depth_l1_g, depth_l1_max_4m_g, _ = self.video.eval_depth_l1(
            f"{self.save_dir}/video.npz", self.stream, global_scale
        )
        self.printer.print("Depth L1: " + str(depth_l1_g), FontColor.EVAL)
        self.printer.print(
            "Depth L1 mask 4m: " + str(depth_l1_max_4m_g), FontColor.EVAL
        )

        # save output data to dict
        # File path where you want to save the .txt file
        file_path = f"{self.save_dir}/depth_stats.txt"
        integers = {
            "depth_l1": depth_l1,
            "depth_l1_global_scale": depth_l1_g,
            "depth_l1_mask_4m": depth_l1_max_4m,
            "depth_l1_mask_4m_global_scale": depth_l1_max_4m_g,
            "Average frame coverage": coverage,  # How much of each frame uses depth from droid (the rest from Omnidata)
            "traj scaling": global_scale,
            "traj rotation": r_a,
            "traj translation": t_a,
            "traj stats": ate_statistics,
        }
        # Write to the file
        with open(file_path, "w") as file:
            for label, number in integers.items():
                file.write(f"{label}: {number}\n")

        self.printer.print(f"File saved as {file_path}", FontColor.EVAL)

    def run(self):
        if self.cfg['gui']:
            from src.gui import gui_utils, slam_gui

        keyframe_queue = mp.Queue()
        self.keyframe_queue = keyframe_queue

        q_main2vis = mp.Queue() if self.cfg['gui'] else None
        q_vis2main = mp.Queue() if self.cfg['gui'] else None

        processes = [
            mp.Process(target=self.tracking, args=(keyframe_queue,)),
            mp.Process(target=self.mapping, args=(keyframe_queue,q_main2vis,q_vis2main)),
        ]
        self.num_running_thread += len(processes)
        if self.cfg['gui']:
            self.num_running_thread += 1
        for p in processes:
            p.start()

        if self.cfg['gui']:
            pipeline_params = munchify(self.cfg["mapping"]["pipeline_params"])
            bg_color = [0, 0, 0]
            background = torch.tensor(
                bg_color, dtype=torch.float32, device=self.device
            )
            gaussians = GaussianModel(self.cfg['mapping']['model_params']['sh_degree'], config=self.cfg)

            params_gui = gui_utils.ParamsGUI(
                pipe=pipeline_params,
                background=background,
                gaussians=gaussians,
                q_main2vis=q_main2vis,
                q_vis2main=q_vis2main,
            )
            gui_process = mp.Process(target=slam_gui.run, args=(params_gui,))
            gui_process.start()
            self.all_trigered += 1


        for p in processes:
            p.join()

        self.printer.terminate()

        for process in mp.active_children():
            process.terminate()
            process.join()


def gen_pose_matrix(R, T):
    pose = np.eye(4)
    pose[0:3, 0:3] = R.cpu().numpy()
    pose[0:3, 3] = T.cpu().numpy()
    return pose
