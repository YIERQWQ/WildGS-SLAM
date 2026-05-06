import argparse
import io
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import zmq
except Exception:  # pragma: no cover - optional dependency
    zmq = None

from multiprocessing.connection import Listener

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.mono_priors.img_feature_extractors import (
    load_dinov3_backbone,
    process_image,
)


def decode_rgb_jpeg(payload: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    arr = np.array(image, dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0
    return tensor


def build_model(model_name: str, device: str, weights: str | None):
    return load_dinov3_backbone(
        device=device,
        model_name=model_name,
        weights=weights,
    )


def extract_patch_features(model, image_rgb: torch.Tensor, device: str, stride: int = 16):
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None]
        return (x - mean) / std

    image = process_image(image_rgb.unsqueeze(0), stride, _normalize, device)

    with torch.no_grad():
        features_dict = model.forward_features(image)

    if "x_norm_patchtokens" in features_dict:
        patch_tokens = features_dict["x_norm_patchtokens"]
    elif "x_prenorm" in features_dict:
        patch_tokens = features_dict["x_prenorm"]
    elif "last_hidden_state" in features_dict:
        hidden = features_dict["last_hidden_state"]
        patch_tokens = hidden[:, 1:, :]
    else:
        raise KeyError("DINOv3 output missing patch tokens")

    patch_h = image.shape[2] // stride
    patch_w = image.shape[3] // stride
    patch_tokens = patch_tokens.reshape(1, patch_h, patch_w, -1)
    feat_map = patch_tokens.permute(0, 3, 1, 2).contiguous()
    return feat_map


def compute_beta_from_features(feat_map: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    feat = F.normalize(feat_map.float(), p=2, dim=1)
    local_mean = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
    local_mean = F.normalize(local_mean, p=2, dim=1)
    inconsistency = 1.0 - (feat * local_mean).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)

    flat = inconsistency.flatten(1)
    lo = torch.quantile(flat, 0.05, dim=1, keepdim=True).view(-1, 1, 1, 1)
    hi = torch.quantile(flat, 0.95, dim=1, keepdim=True).view(-1, 1, 1, 1)
    beta = (inconsistency - lo) / (hi - lo + 1e-6)
    beta = beta.clamp(0.0, 1.0)
    beta = 0.1 + 0.9 * beta
    beta = F.interpolate(beta, size=target_size, mode="bilinear", align_corners=False)
    return beta.squeeze(0).squeeze(0).contiguous()


def handle_request(
    request: Dict,
    model,
    device: str,
    out_dir: str,
    feature_stride: int,
    save_feature_npy: bool,
):
    if request.get("cmd") == "shutdown":
        return {"status": "bye"}

    frame_id = int(request["frame_id"])
    video_idx = int(request.get("video_idx", frame_id))
    image = decode_rgb_jpeg(request["rgb_jpeg"])
    target_h, target_w = request.get("image_size", [image.shape[1], image.shape[2]])
    beta_target = (max(1, int(target_h) // 8), max(1, int(target_w) // 8))

    start = time.time()
    feat_map = extract_patch_features(model, image, device=device, stride=feature_stride)
    beta = compute_beta_from_features(feat_map, beta_target)
    latency_ms = (time.time() - start) * 1000.0

    beta_path = Path(out_dir) / f"{frame_id:05d}_beta.npy"
    np.save(beta_path, beta.detach().cpu().numpy())
    feat_path = Path(out_dir) / f"{frame_id:05d}_feat.npy"
    if save_feature_npy:
        np.save(feat_path, feat_map.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())

    return {
        "status": "ok",
        "frame_id": frame_id,
        "video_idx": video_idx,
        "kf_seq": request.get("kf_seq"),
        "latency_ms": latency_ms,
        "beta": beta.detach().cpu().numpy(),
        "features": feat_map.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(),
        "feature_stride": feature_stride,
        "feature_path": str(feat_path) if save_feature_npy else None,
        "beta_path": str(beta_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dinov3_vits16")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--out-dir", type=str, default="./output/dino_beta_cache")
    parser.add_argument("--feature-stride", type=int, default=16)
    parser.add_argument("--feature-npy", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = build_model(args.model, args.device, args.weights)

    use_zmq = zmq is not None
    endpoint = f"tcp://{args.host}:{args.port}"

    if use_zmq:
        context = zmq.Context.instance()
        socket = context.socket(zmq.REP)
        socket.linger = 0
        socket.bind(endpoint)
        print(f"[DINO-BETA] listening on {endpoint}, device={args.device}, model={args.model}")
        while True:
            request = socket.recv_pyobj()
            response = handle_request(
                request,
                model=model,
                device=args.device,
                out_dir=args.out_dir,
                feature_stride=args.feature_stride,
                save_feature_npy=args.feature_npy,
            )
            socket.send_pyobj(response)
            if request.get("cmd") == "shutdown":
                break
    else:
        listener = Listener((args.host, args.port), authkey=b"wildgs-beta")
        print(f"[DINO-BETA] listening on {args.host}:{args.port}, device={args.device}, model={args.model}")
        while True:
            conn = listener.accept()
            print("[DINO-BETA] client connected")
            try:
                while True:
                    request = conn.recv()
                    response = handle_request(
                        request,
                        model=model,
                        device=args.device,
                        out_dir=args.out_dir,
                        feature_stride=args.feature_stride,
                        save_feature_npy=args.feature_npy,
                    )
                    conn.send(response)
                    if request.get("cmd") == "shutdown":
                        return
            except EOFError:
                print("[DINO-BETA] client disconnected")
            except Exception as exc:
                try:
                    conn.send({"status": "error", "error": str(exc)})
                except Exception:
                    pass
            finally:
                conn.close()


if __name__ == "__main__":
    main()
