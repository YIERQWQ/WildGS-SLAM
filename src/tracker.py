import json
import os
import time

import torch

from src.backend import Backend
from src.frontend import Frontend
from src.motion_filter import MotionFilter
from src.utils.Printer import FontColor, Printer
from src.utils.datasets import BaseDataset


class Tracker:
    def __init__(self, slam, packet_queue):
        self.cfg = slam.cfg
        self.device = self.cfg['device']
        self.net = slam.droid_net
        self.video = slam.video
        self.verbose = slam.verbose
        self.packet_queue = packet_queue
        self.output = slam.save_dir

        # filter incoming frames so that there is enough motion
        self.frontend_window = self.cfg['tracking']['frontend']['window']
        filter_thresh = self.cfg['tracking']['motion_filter']['thresh']
        self.motion_filter = MotionFilter(self.net, self.video, self.cfg, thresh=filter_thresh, device=self.device)
        self.enable_online_ba = self.cfg['tracking']['frontend']['enable_online_ba']
        # frontend process
        self.frontend = Frontend(self.net, self.video, self.cfg)
        self.online_ba = Backend(self.net,self.video, self.cfg)
        self.ba_freq = self.cfg['tracking']['backend']['ba_freq']

        self.printer:Printer = slam.printer
        self.next_kf_seq = 0
        self.run_log_dir = os.environ.get("RUN_LOG_DIR")
        self.run_log_scene_dir = (
            os.path.join(self.run_log_dir, self.cfg["scene"])
            if self.run_log_dir
            else None
        )

    def _write_tracker_summary(self, stats: dict) -> None:
        paths = [os.path.join(self.output, "tracker_metrics.json")]
        if self.run_log_scene_dir is not None:
            paths.append(os.path.join(self.run_log_scene_dir, "tracker_metrics.json"))

        for path in paths:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(stats, fp, indent=2, sort_keys=True)

    def run(self, stream:BaseDataset):
        '''
        Trigger the tracking process.
        1. check whether there is enough motion between the current frame and last keyframe by motion_filter
        2. use frontend to do local bundle adjustment, to estimate camera pose and depth image, 
            also delete the current keyframe if it is too close to the previous keyframe after local BA.
        3. run online global BA periodically by backend
        4. send immutable keyframe packets to mapper asynchronously.
        '''
        run_start = time.perf_counter()
        frame_elapsed_s = 0.0
        keyframe_elapsed_s = 0.0
        non_keyframe_elapsed_s = 0.0
        keyframe_count = 0
        first_keyframe_latency_s = None
        prev_ba_seq = 0

        intrinsic = stream.get_intrinsic()
        # for (timestamp, image, _, _) in tqdm(stream):
        for i in range(len(stream)):
            timestamp, image, _, _ = stream[i]
            with torch.no_grad():
                frame_start = time.perf_counter()
                starting_count = self.video.counter.value
                ### check there is enough motion
                force_to_add_keyframe = self.motion_filter.track(timestamp, image, intrinsic)

                # local bundle adjustment
                self.frontend(force_to_add_keyframe)

                if (starting_count < self.video.counter.value) and self.cfg['mapping']['full_resolution']:
                    if self.motion_filter.uncertainty_aware:
                        img_full = stream.get_color_full_resol(i)
                        self.motion_filter.get_img_feature(timestamp,img_full,suffix='full')
                new_count = self.video.counter.value
                if new_count > starting_count:
                    curr_kf_idx = new_count - 1
                    kf_seq = self.next_kf_seq
                    just_initialized = new_count == self.frontend.warmup

                    if (
                        self.enable_online_ba
                        and not just_initialized
                        and kf_seq >= prev_ba_seq + self.ba_freq
                    ):
                        self.printer.print(
                            f"Online BA at {curr_kf_idx}th keyframe, frame index: {timestamp}",
                            FontColor.TRACKER,
                        )
                        self.online_ba.dense_ba(2)
                        prev_ba_seq = kf_seq

                    snapshot = self.video.export_keyframe_snapshot(curr_kf_idx)
                    self.packet_queue.put(
                        {
                            "type": "keyframe",
                            "kf_seq": kf_seq,
                            "video_idx": curr_kf_idx,
                            "timestamp": snapshot["timestamp"],
                            "frame_id": snapshot["frame_id"],
                            "just_initialized": just_initialized,
                            "end": False,
                            "image": snapshot["image"],
                            "pose": snapshot["pose"],
                            "disp": snapshot["disp"],
                            "mono_depth": snapshot["mono_depth"],
                            "intrinsic": snapshot["intrinsic"],
                            "fmap": snapshot["fmap"],
                            "net": snapshot["net"],
                            "inp": snapshot["inp"],
                            "dino_feature": snapshot.get("dino_feature"),
                        }
                    )

                    self.next_kf_seq += 1

                    if just_initialized:
                        self.frontend.initialize_second_stage()
            self.printer.update_pbar()
            frame_time_s = time.perf_counter() - frame_start
            frame_elapsed_s += frame_time_s
            if new_count > starting_count:
                keyframe_count += 1
                keyframe_elapsed_s += frame_time_s
                if first_keyframe_latency_s is None:
                    first_keyframe_latency_s = frame_time_s
            else:
                non_keyframe_elapsed_s += frame_time_s

        loop_end_s = time.perf_counter()

        total_elapsed_s = loop_end_s - run_start
        total_frames = len(stream)
        steady_frames = max(total_frames - 1, 1)
        steady_elapsed_s = max(
            total_elapsed_s - (first_keyframe_latency_s or 0.0),
            1e-9,
        )
        stats = {
            "frames_total": int(total_frames),
            "keyframes_total": int(keyframe_count),
            "keyframe_ratio": float(keyframe_count / max(total_frames, 1)),
            "total_elapsed_s": float(total_elapsed_s),
            "frame_elapsed_s": float(frame_elapsed_s),
            "keyframe_elapsed_s": float(keyframe_elapsed_s),
            "non_keyframe_elapsed_s": float(non_keyframe_elapsed_s),
            "frontend_fps": float(total_frames / max(total_elapsed_s, 1e-9)),
            "frontend_avg_ms": float(1000.0 * total_elapsed_s / max(total_frames, 1)),
            "frontend_fps_steady": float(steady_frames / steady_elapsed_s),
            "keyframe_avg_ms": float(
                1000.0 * keyframe_elapsed_s / max(keyframe_count, 1)
            ),
            "non_keyframe_avg_ms": float(
                1000.0 * non_keyframe_elapsed_s / max(total_frames - keyframe_count, 1)
            ),
            "first_keyframe_latency_ms": None
            if first_keyframe_latency_s is None
            else float(first_keyframe_latency_s * 1000.0),
        }
        self._write_tracker_summary(stats)

        self.packet_queue.put({"type": "end"})


                
