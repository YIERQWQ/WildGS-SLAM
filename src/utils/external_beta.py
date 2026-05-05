import io
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import zmq
except Exception:  # pragma: no cover - optional dependency
    zmq = None

from multiprocessing.connection import Client


def _to_uint8_rgb(image: torch.Tensor) -> np.ndarray:
    if image.dim() == 4:
        image = image[0]

    if image.dim() != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got {image.shape}")

    if image.shape[0] == 3:
        rgb = image
    elif image.shape[-1] == 3:
        rgb = image.permute(2, 0, 1)
    else:
        raise ValueError(f"Expected RGB image, got shape {image.shape}")

    rgb = rgb.detach().cpu().float().clamp(0.0, 1.0)
    rgb = (rgb * 255.0).byte().permute(1, 2, 0).contiguous().numpy()
    return rgb


def encode_rgb_jpeg(image: torch.Tensor, quality: int = 90) -> Tuple[bytes, Tuple[int, int]]:
    rgb = _to_uint8_rgb(image)
    pil_image = Image.fromarray(rgb, mode="RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=int(quality), optimize=False)
    return buffer.getvalue(), (rgb.shape[0], rgb.shape[1])


def decode_beta_payload(payload: Any) -> np.ndarray:
    if isinstance(payload, np.ndarray):
        beta = payload
    elif torch.is_tensor(payload):
        beta = payload.detach().cpu().float().numpy()
    else:
        beta = np.asarray(payload, dtype=np.float32)

    if beta.ndim == 3 and beta.shape[0] == 1:
        beta = beta[0]
    if beta.ndim == 3 and beta.shape[-1] == 1:
        beta = beta[..., 0]
    return beta.astype(np.float32, copy=False)


def encode_beta_payload(beta: torch.Tensor) -> np.ndarray:
    return decode_beta_payload(beta)


@dataclass
class BetaSample:
    frame_id: int
    video_idx: int
    beta: np.ndarray
    status: str = "ok"
    latency_ms: float = 0.0


class AsyncBetaClient:
    """Non-blocking client used by the main SLAM process."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        authkey: str = "wildgs-beta",
        jpeg_quality: int = 90,
        max_request_queue: int = 32,
        max_result_queue: int = 256,
        connect_retry_s: float = 0.5,
        on_result=None,
    ):
        self.host = host
        self.port = int(port)
        self.authkey = authkey.encode("utf-8")
        self.jpeg_quality = int(jpeg_quality)
        self.max_request_queue = int(max_request_queue)
        self.max_result_queue = int(max_result_queue)
        self.connect_retry_s = float(connect_retry_s)
        self.on_result = on_result

        self._use_zmq = zmq is not None
        self._request_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=self.max_request_queue)
        self._result_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=self.max_result_queue)
        self._cache: Dict[int, np.ndarray] = {}
        self._result_cache_by_video_idx: Dict[int, Dict[str, Any]] = {}
        self._result_cache_by_frame_id: Dict[int, Dict[str, Any]] = {}
        self._frame_cache: Dict[int, int] = {}
        self._last_beta: Optional[np.ndarray] = None
        self._last_result: Optional[Dict[str, Any]] = None
        self._last_video_idx: Optional[int] = None
        self._last_frame_id: Optional[int] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._warned = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _connect(self):
        if self._use_zmq:
            ctx = zmq.Context.instance()
            socket = ctx.socket(zmq.REQ)
            socket.linger = 0
            socket.connect(f"tcp://{self.host}:{self.port}")
            return socket
        return Client((self.host, self.port), authkey=self.authkey)

    def _send(self, conn, request: Dict[str, Any]) -> None:
        if self._use_zmq:
            conn.send_pyobj(request)
        else:
            conn.send(request)

    def _recv(self, conn) -> Dict[str, Any]:
        if self._use_zmq:
            return conn.recv_pyobj()
        return conn.recv()

    def _close(self, conn) -> None:
        try:
            if self._use_zmq:
                conn.close(linger=0)
            else:
                conn.close()
        except Exception:
            pass

    def submit(
        self,
        frame_id: int,
        video_idx: int,
        image: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> bool:
        if self._thread is None:
            self.start()

        try:
            rgb_jpeg, image_size = encode_rgb_jpeg(image, quality=self.jpeg_quality)
        except Exception:
            return False

        if isinstance(intrinsics, torch.Tensor):
            intrinsics_list = [float(x) for x in intrinsics.detach().cpu().flatten().tolist()]
        else:
            intrinsics_list = [float(x) for x in intrinsics]

        request = {
            "cmd": "infer",
            "frame_id": int(frame_id),
            "video_idx": int(video_idx),
            "intrinsics": intrinsics_list,
            "image_size": [int(image_size[0]), int(image_size[1])],
            "rgb_jpeg": rgb_jpeg,
        }

        try:
            self._request_q.put_nowait(request)
            return True
        except queue.Full:
            return False

    def request_sync(
        self,
        frame_id: int,
        video_idx: int,
        image: torch.Tensor,
        intrinsics: torch.Tensor,
        timeout_s: float = 120.0,
    ) -> Dict[str, Any]:
        rgb_jpeg, image_size = encode_rgb_jpeg(image, quality=self.jpeg_quality)
        if isinstance(intrinsics, torch.Tensor):
            intrinsics_list = [float(x) for x in intrinsics.detach().cpu().flatten().tolist()]
        else:
            intrinsics_list = [float(x) for x in intrinsics]

        request = {
            "cmd": "infer",
            "frame_id": int(frame_id),
            "video_idx": int(video_idx),
            "intrinsics": intrinsics_list,
            "image_size": [int(image_size[0]), int(image_size[1])],
            "rgb_jpeg": rgb_jpeg,
        }

        conn = None
        deadline = time.time() + float(timeout_s)
        while conn is None:
            try:
                conn = self._connect()
            except Exception:
                if time.time() > deadline:
                    raise TimeoutError("Could not connect to DINOv3 service")
                time.sleep(self.connect_retry_s)

        try:
            self._send(conn, request)
            result = self._recv(conn)
        finally:
            self._close(conn)

        if isinstance(result, dict) and result.get("status") not in (None, "ok"):
            raise RuntimeError(
                f"DINOv3 beta service returned error: {result.get('error', 'unknown')}"
            )
        return result

    def drain_results(self, max_items: int = 32) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for _ in range(max_items):
            try:
                result = self._result_q.get_nowait()
            except queue.Empty:
                break

            if isinstance(result, dict) and result.get("status") not in (None, "ok"):
                raise RuntimeError(
                    f"DINOv3 beta service returned error: {result.get('error', 'unknown')}"
                )

            frame_id = int(result["frame_id"])
            video_idx = int(result["video_idx"])
            beta = decode_beta_payload(result["beta"])
            result = dict(result)
            result["beta"] = beta

            self._cache[video_idx] = beta
            self._frame_cache[frame_id] = video_idx
            self._result_cache_by_video_idx[video_idx] = result
            self._result_cache_by_frame_id[frame_id] = result
            self._last_beta = beta
            self._last_result = result
            self._last_video_idx = video_idx
            self._last_frame_id = frame_id
            results.append(result)
        return results

    def poll(self, max_items: int = 32) -> int:
        return len(self.drain_results(max_items=max_items))

    def get_latest(self) -> Optional[np.ndarray]:
        return self._last_beta

    def get_result(
        self,
        video_idx: Optional[int] = None,
        frame_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if video_idx is not None and video_idx in self._result_cache_by_video_idx:
            return self._result_cache_by_video_idx[video_idx]
        if frame_id is not None and frame_id in self._result_cache_by_frame_id:
            return self._result_cache_by_frame_id[frame_id]
        return None

    def has_result(self, video_idx: Optional[int] = None, frame_id: Optional[int] = None) -> bool:
        if video_idx is not None and video_idx in self._result_cache_by_video_idx:
            return True
        if frame_id is not None and frame_id in self._result_cache_by_frame_id:
            return True
        return self._last_result is not None

    def get(
        self,
        video_idx: Optional[int] = None,
        frame_id: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        if video_idx is not None and video_idx in self._cache:
            return self._cache[video_idx]
        if frame_id is not None and frame_id in self._frame_cache:
            return self._cache.get(self._frame_cache[frame_id])
        return None

    def wait_for_result(
        self,
        video_idx: Optional[int] = None,
        frame_id: Optional[int] = None,
        timeout_s: float = 30.0,
    ) -> Dict[str, Any]:
        deadline = time.time() + float(timeout_s)
        while True:
            result = self.get_result(video_idx=video_idx, frame_id=frame_id)
            if result is not None:
                return result
            self.drain_results(max_items=32)
            result = self.get_result(video_idx=video_idx, frame_id=frame_id)
            if result is not None:
                return result
            if time.time() > deadline:
                ident = f"video_idx={video_idx}" if video_idx is not None else f"frame_id={frame_id}"
                raise TimeoutError(f"Timed out waiting for DINOv3 beta result ({ident})")
            time.sleep(self.connect_retry_s)

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _worker_loop(self) -> None:
        conn = None
        while not self._stop_event.is_set():
            if conn is None:
                try:
                    conn = self._connect()
                    self._warned = False
                except Exception:
                    if not self._warned:
                        print(f"[DINO-BETA] Waiting for server at {self.host}:{self.port}")
                        self._warned = True
                    time.sleep(self.connect_retry_s)
                    continue

            try:
                request = self._request_q.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._send(conn, request)
                result = self._recv(conn)
                if isinstance(result, dict) and result.get("status") not in (None, "ok"):
                    raise RuntimeError(
                        f"DINOv3 beta service returned error: {result.get('error', 'unknown')}"
                    )
                if self.on_result is not None:
                    try:
                        self.on_result(result)
                    except Exception as exc:
                        print(f"[DINO-BETA] result callback failed: {exc}")
                try:
                    self._result_q.put_nowait(result)
                except queue.Full:
                    try:
                        _ = self._result_q.get_nowait()
                    except queue.Empty:
                        pass
                    self._result_q.put_nowait(result)
            except Exception:
                self._close(conn)
                conn = None
                try:
                    self._request_q.put_nowait(request)
                except queue.Full:
                    pass
                time.sleep(self.connect_retry_s)

        if conn is not None:
            self._close(conn)


class ExternalDinoClient(AsyncBetaClient):
    """Compatibility alias for the DINOv3 service client."""

    pass
