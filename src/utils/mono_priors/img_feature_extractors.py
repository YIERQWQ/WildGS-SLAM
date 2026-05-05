import os
from typing import Dict, List, Tuple, Union
import inspect
import numpy as np
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

"""
From FiT3D, here we subclass the model instead of overriding the "get_intermediate_layers" method
as it will cause errors in multipprocessing setup of the SLAM system
"""


def _ensure_torch_amp_compat() -> None:
    """Make DINOv3 hub imports work on torch builds without torch.amp decorators."""
    try:
        import torch.amp as torch_amp
    except Exception:
        return

    def _supports_device_type(decorator) -> bool:
        try:
            signature = inspect.signature(decorator)
        except Exception:
            return False
        return "device_type" in signature.parameters

    if (
        hasattr(torch_amp, "custom_fwd")
        and hasattr(torch_amp, "custom_bwd")
        and _supports_device_type(torch_amp.custom_fwd)
        and _supports_device_type(torch_amp.custom_bwd)
    ):
        return

    def _identity_amp_decorator(*decorator_args, **decorator_kwargs):
        if (
            len(decorator_args) == 1
            and callable(decorator_args[0])
            and not decorator_kwargs
        ):
            return decorator_args[0]

        def _decorator(fn):
            return fn

        return _decorator

    torch_amp.custom_fwd = _identity_amp_decorator
    torch_amp.custom_bwd = _identity_amp_decorator


def _ensure_torch_dynamo_compat() -> None:
    """Backfill TorchDynamo config keys expected by DINOv3 on older torch builds."""
    try:
        import torch._dynamo.config as dynamo_config
    except Exception:
        return

    key = "accumulated_cache_size_limit"
    allowed_keys = getattr(dynamo_config, "_allowed_keys", None)
    if not isinstance(allowed_keys, set):
        return
    if key in allowed_keys:
        return

    allowed_keys.add(key)

    config_dict = getattr(dynamo_config, "_config", None)
    default_dict = getattr(dynamo_config, "_default", None)
    if not isinstance(config_dict, dict):
        return

    fallback = config_dict.get("cache_size_limit")
    if fallback is None:
        fallback = config_dict.get("recompile_limit")
    if fallback is None:
        fallback = 64

    config_dict.setdefault(key, fallback)
    if isinstance(default_dict, dict):
        default_dict.setdefault(key, fallback)


def _find_cached_dinov3_repo() -> Path:
    hub_dir = Path(torch.hub.get_dir())
    candidates = sorted(
        p for p in hub_dir.glob("facebookresearch_dinov3*") if p.is_dir()
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not locate a cached DINOv3 repo under {hub_dir}"
        )
    return candidates[0]


def _resolve_dinov3_weights(model_name: str, weights: str | None) -> str:
    if weights:
        candidate = Path(weights).expanduser()
        if candidate.exists():
            return str(candidate)
        if weights.startswith(("http://", "https://", "file://")):
            return weights

    env_weights = os.environ.get("DINOV3_WEIGHTS")
    if env_weights:
        return _resolve_dinov3_weights(model_name, env_weights)

    raise ValueError(
        f"DINOv3 weights are required for {model_name}. "
        "Set mono_prior.feature_weights, pass --weights to dino_beta_service.py, "
        "or export DINOV3_WEIGHTS to an explicit checkpoint path/URL. "
        "Recommended official checkpoint: "
        "https://dl.fbaipublicfiles.com/dinov3/"
        "dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    )


def load_dinov3_backbone(
    device: str,
    model_name: str = "dinov3_vits16",
    weights: str | None = None,
) -> nn.Module:
    _ensure_torch_amp_compat()
    _ensure_torch_dynamo_compat()
    repo_root = _find_cached_dinov3_repo()
    if model_name not in {"dinov3", "dinov3_vits16", "dinov3_vits16_pretrain"}:
        raise NotImplementedError(f"Unsupported DINOv3 backbone: {model_name}")

    weights_ref = _resolve_dinov3_weights(model_name, weights)
    model = torch.hub.load(
        str(repo_root),
        model_name,
        source="local",
        weights=weights_ref,
    )
    return model.to(device).eval()


class Fit3DModels(torch.nn.Module):
    def __init__(self, extractor_model, device):
        super().__init__()
        _ensure_torch_amp_compat()
        self.model = torch.hub.load("ywyue/FiT3D", extractor_model).to(device).eval()

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n=1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        outputs = self.model._intermediate_layers(x, n)
        if norm:
            outputs = [self.model.norm(out) for out in outputs]
        if return_class_token:
            prefix_tokens = [out[:, 0] for out in outputs]
        else:
            prefix_tokens = [
                out[:, 0 : self.model.num_prefix_tokens] for out in outputs
            ]
        outputs = [out[:, self.model.num_prefix_tokens :] for out in outputs]

        if reshape:
            B, C, H, W = x.shape
            grid_size = (
                (H - self.model.patch_embed.patch_size[0])
                // self.model.patch_embed.proj.stride[0]
                + 1,
                (W - self.model.patch_embed.patch_size[1])
                // self.model.patch_embed.proj.stride[1]
                + 1,
            )
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]

        if return_prefix_tokens or return_class_token:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)


"""
Done with overwriting get_intermediate_layers of FiT3D model
"""


def get_feature_extractor(cfg: Dict) -> nn.Module:
    """
    Get the feature extractor model based on the configuration.
    """
    device = cfg["device"]
    extractor_model = cfg["mono_prior"]["feature_extractor"]

    if extractor_model in ["dinov3_vits16", "dinov3_vits16_pretrain", "dinov3"]:
        weights = cfg["mono_prior"].get("feature_weights")
        return load_dinov3_backbone(
            device=device, model_name=extractor_model, weights=weights
        )
    if extractor_model in ["dinov2_reg_small_fine", "dinov2_small_fine"]:
        return Fit3DModels(extractor_model, device)
    elif extractor_model in ["dinov2_vits14", "dinov2_vits14_reg"]:
        return (
            torch.hub.load("facebookresearch/dinov2", extractor_model).to(device).eval()
        )
    else:
        # If use other feature extractor as prior, add code here
        raise NotImplementedError("Unsupported feature extractor")


@torch.no_grad()
def predict_img_features(
    model: nn.Module,
    idx: int,
    input_tensor: torch.Tensor,
    cfg: Dict,
    device: str,
    save_feat: bool = True,
    suffix: str = "",
) -> torch.Tensor:
    """
    Predict image features using the given model.

    Args:
        model (nn.Module): The feature extractor model.
        idx (int): Image index.
        input_tensor (torch.Tensor): Input image tensor of shape (1, 3, H, W).
        cfg (Dict): Configuration dictionary.
        device (str): Device to run the model on.
        save_feat (bool): Whether to save the features.
        suffix (str): Suffix for the output file name.

    Returns:
        torch.Tensor: Extracted features.
    """
    extractor_model = cfg["mono_prior"]["feature_extractor"]
    stride = 16 if "dinov3" in extractor_model else 14
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean, std=std)
    image_resized = process_image(input_tensor, stride, normalize, device)

    if extractor_model in ["dinov3_vits16", "dinov3_vits16_pretrain", "dinov3"]:
        features_dict = model.forward_features(image_resized)
        if "x_norm_patchtokens" in features_dict:
            patch_tokens = features_dict["x_norm_patchtokens"]
        elif "x_prenorm" in features_dict:
            patch_tokens = features_dict["x_prenorm"]
        else:
            raise KeyError("DINOv3 output missing x_norm_patchtokens")
        patch_h = image_resized.shape[2] // stride
        patch_w = image_resized.shape[3] // stride
        features = patch_tokens.view(patch_h, patch_w, -1)
    elif extractor_model in ["dinov2_reg_small_fine", "dinov2_small_fine"]:
        features = model.get_intermediate_layers(
            image_resized,
            n=[8, 9, 10, 11],
            reshape=True,
            return_prefix_tokens=False,
            return_class_token=False,
            norm=True,
        )
        features = features[-1].squeeze().permute((1, 2, 0))
    elif extractor_model in ["dinov2_vits14", "dinov2_vits14_reg"]:
        features_dict = model.forward_features(image_resized)
        features = features_dict["x_norm_patchtokens"].view(
            image_resized.shape[2] // 14, image_resized.shape[3] // 14, -1
        )
    else:
        # If use other feature extractor as prior, add code here
        raise NotImplementedError("Unsupported feature extractor")

    if save_feat:
        _save_features(features, cfg, idx, suffix)

    return features


def process_image(
    image: torch.Tensor, stride: int, transforms: nn.Module, device: str = "cuda"
) -> torch.Tensor:
    """
    Process the input image for feature extraction.

    Args:
        image (torch.Tensor): Input image tensor.
        stride (int): Stride for resizing.
        transforms (nn.Module): Normalization transforms.
        device (str): Device to run the processing on.

    Returns:
        torch.Tensor: Processed image tensor.
    """
    image_tensor = image.to(device)
    image_tensor = transforms(image_tensor).float()
    h, w = image_tensor.shape[2:]
    height_int = (h // stride) * stride
    width_int = (w // stride) * stride
    return F.interpolate(image_tensor, size=(height_int, width_int), mode="bilinear")


def _save_features(features: torch.Tensor, cfg: Dict, idx: int, suffix: str) -> None:
    """
    Save the extracted features to a file.

    Args:
        features (torch.Tensor): Extracted features.
        cfg (Dict): Configuration dictionary.
        idx (int): Image index.
        suffix (str): Suffix for the output file name.
    """
    output_dir = f"{cfg['data']['output']}/{cfg['scene']}"
    output_path = f"{output_dir}/mono_priors/features/{idx:05d}{suffix}.npy"
    final_feat = features.detach().cpu().float().numpy()
    np.save(output_path, final_feat)
