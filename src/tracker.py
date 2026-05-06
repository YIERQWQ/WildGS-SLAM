import json
import os
import time

from src.motion_filter import MotionFilter
from src.frontend import Frontend 
from src.backend import Backend
import torch
from colorama import Fore, Style
from multiprocessing.connection import Connection
from src.utils.datasets import BaseDataset
from src.utils.Printer import Printer,FontColor
class Tracker:
    def __init__(self, slam, pipe:Connection):
        self.cfg = slam.cfg
        self.device = self.cfg['device']
        self.net = slam.droid_net
        self.video = slam.video
        self.verbose = slam.verbose
        self.pipe = pipe
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
        self.run_log_dir = os.environ.get("RUN_LOG_DIR")
        self.run_log_scene_dir = (
            os.path.join(self.run_log_dir, self.cfg["scene"])
            if self.run_log_dir
            else None
        )

    def _write_tracker_summary(self, stats):
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
        4. send the estimated pose and depth to mapper, 
            and wait until the mapper finish its current mapping optimization.
        '''
        run_start = time.perf_counter()
        frame_elapsed_s = 0.0
        keyframe_elapsed_s = 0.0
        non_keyframe_elapsed_s = 0.0
        keyframe_count = 0
        first_keyframe_latency_s = None
        prev_kf_idx = 0
        curr_kf_idx = 0
        prev_ba_idx = 0

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
            curr_kf_idx = self.video.counter.value - 1
            is_keyframe = curr_kf_idx != prev_kf_idx and self.frontend.is_initialized
            
            if is_keyframe:
                if self.video.counter.value == self.frontend.warmup:
                    ## We just finish the initialization
                    self.pipe.send({"is_keyframe":True, "video_idx":curr_kf_idx,
                                    "timestamp":timestamp, "just_initialized": True, 
                                    "end":False})
                    self.pipe.recv()
                    self.frontend.initialize_second_stage()
                else:
                    if self.enable_online_ba and curr_kf_idx >= prev_ba_idx + self.ba_freq:
                        # run online global BA every {self.ba_freq} keyframes
                        self.printer.print(f"Online BA at {curr_kf_idx}th keyframe, frame index: {timestamp}",FontColor.TRACKER)
                        self.online_ba.dense_ba(2)
                        prev_ba_idx = curr_kf_idx
                    # inform the mapper that the estimation of current pose and depth is finished
                    self.pipe.send({"is_keyframe":True, "video_idx":curr_kf_idx,
                                    "timestamp":timestamp, "just_initialized": False, 
                                    "end":False})
                    self.pipe.recv()

            prev_kf_idx = curr_kf_idx
            self.printer.update_pbar()

            frame_time_s = time.perf_counter() - frame_start
            frame_elapsed_s += frame_time_s
            if is_keyframe:
                keyframe_count += 1
                keyframe_elapsed_s += frame_time_s
                if first_keyframe_latency_s is None:
                    first_keyframe_latency_s = frame_time_s
            else:
                non_keyframe_elapsed_s += frame_time_s

        total_elapsed_s = time.perf_counter() - run_start
        steady_frames = max(len(stream) - 1, 1)
        steady_elapsed_s = max(
            total_elapsed_s - (first_keyframe_latency_s or 0.0),
            1e-9,
        )
        stats = {
            "frames_total": int(len(stream)),
            "keyframes_total": int(keyframe_count),
            "keyframe_ratio": float(keyframe_count / max(len(stream), 1)),
            "total_elapsed_s": float(total_elapsed_s),
            "frame_elapsed_s": float(frame_elapsed_s),
            "keyframe_elapsed_s": float(keyframe_elapsed_s),
            "non_keyframe_elapsed_s": float(non_keyframe_elapsed_s),
            "frontend_fps": float(len(stream) / max(total_elapsed_s, 1e-9)),
            "frontend_avg_ms": float(1000.0 * total_elapsed_s / max(len(stream), 1)),
            "frontend_fps_steady": float(steady_frames / steady_elapsed_s),
            "keyframe_avg_ms": float(1000.0 * keyframe_elapsed_s / max(keyframe_count, 1)),
            "non_keyframe_avg_ms": float(
                1000.0 * non_keyframe_elapsed_s / max(len(stream) - keyframe_count, 1)
            ),
            "first_keyframe_latency_ms": None
            if first_keyframe_latency_s is None
            else float(first_keyframe_latency_s * 1000.0),
        }
        self._write_tracker_summary(stats)

        self.pipe.send({"is_keyframe":True, "video_idx":None,
                        "timestamp":None, "just_initialized": False, 
                        "end":True})


                
