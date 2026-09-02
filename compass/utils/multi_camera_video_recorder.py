# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Custom video recorder that records from multiple cameras."""
# pylint: disable=line-too-long,redefined-outer-name,broad-exception-caught
# pylint: disable=import-outside-toplevel,reimported,missing-type-doc
# pylint: disable=subprocess-run-check,bare-except,possibly-unused-variable

import os
import shutil
import time
import traceback
from typing import Callable, Optional

import cv2
import gymnasium as gym
import numpy as np
import torch
import warp as wp

KIT_VIEWPORT_HEIGHT = 540


class MultiCameraVideoRecorder(gym.Wrapper):
    """Wrapper that records videos from multiple cameras simultaneously.
    Records from:
      1. An onboard robot camera sensor - by collecting RGB frames every step
         and writing them to a separate ``robot_camera/`` sub-folder using
         OpenCV.
      2. The active Kit viewport, when enabled - by capturing viewport images
         every step, then assembling them into a video at the end of the clip.

    The separate streams are written independently. When both robot-camera and
    Kit viewport clips are available, a side-by-side combined video is also
    written to the main video folder.

    Usage example in ``run.py``::
        from compass.utils.multi_camera_video_recorder import MultiCameraVideoRecorder
        env = MultiCameraVideoRecorder(
            env,
            video_folder=os.path.join(args_cli.output_dir, "videos"),
            step_trigger=lambda step: step % (num_steps_per_iteration * args_cli.video_interval) == 0,
            video_length=num_steps_per_iteration,
            camera_sensor_name="camera",  # name as registered in env.scene.sensors
            record_kit_viewport="kit" in requested_viz,
        )
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        step_trigger: Optional[Callable[[int], bool]] = None,
        video_length: int = 200,
        camera_sensor_name: str = "camera",
        record_kit_viewport: bool = False,
        fps: Optional[float] = None,
    ):
        """Initialise the multi-camera video recorder.
        Args:
            env: The environment to wrap.
            video_folder: Directory where videos will be saved.
                Combined videos land directly in this folder; robot-camera
                videos land in ``<video_folder>/robot_camera/`` and Kit
                viewport videos land in ``<video_folder>/kit_viewport/``.
            step_trigger: Callable ``(global_step: int) -> bool`` that returns
                ``True`` whenever a new recording should start.  Defaults to
                never triggering.
            video_length: Number of environment steps to record per clip.
            camera_sensor_name: Key used to look up the camera inside
                ``env.scene.sensors``.
            record_kit_viewport: Whether to record the active Kit viewport to
                ``<video_folder>/kit_viewport/``.
            fps: Frame-rate used when writing the robot-camera video file.
                When ``None`` (default) the value is read from
                ``env.metadata["render_fps"]``.
        """
        super().__init__(env)

        self.video_folder = video_folder
        self.step_trigger = step_trigger if step_trigger is not None else (lambda step: False)
        self.video_length = video_length
        self.camera_sensor_name = camera_sensor_name
        self.record_kit_viewport = record_kit_viewport

        # Use env metadata so robot-camera, Kit viewport, and combined clips have
        # the same duration.
        if fps is not None:
            self.fps = fps
        else:
            self.fps = self.env.metadata.get("render_fps", 30)

        # Internal state
        self.recording = False
        self.recorded_steps = 0
        self.current_step = 0
        self.robot_camera_frames: list = []
        self.current_kit_viewport_frame_folder: Optional[str] = None
        self.previous_kit_viewport_frame_folder: Optional[str] = None
        self.recording_start_step = 0

        os.makedirs(self.video_folder, exist_ok=True)
        self.robot_camera_folder = os.path.join(video_folder, "robot_camera")
        os.makedirs(self.robot_camera_folder, exist_ok=True)
        self.kit_viewport_folder = os.path.join(video_folder, "kit_viewport")
        self.kit_viewport_frame_folder = os.path.join(self.kit_viewport_folder, "_frames")
        if self.record_kit_viewport:
            os.makedirs(self.kit_viewport_frame_folder, exist_ok=True)

    # ------------------------------------------------------------------
    # gym.Wrapper interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        """Reset environment and clear in-progress recording buffers."""
        obs = self.env.reset(**kwargs)
        self.recording = False
        self.recorded_steps = 0
        self.robot_camera_frames = []
        self._clear_kit_viewport_frame_folder()
        self._clear_kit_viewport_frame_folder(self.previous_kit_viewport_frame_folder)
        self.previous_kit_viewport_frame_folder = None
        # Keep global step numbering continuous across resets.
        return obs

    def close(self):
        """Close the environment and remove unfinished temporary Kit captures."""
        self._clear_kit_viewport_frame_folder()
        self._clear_kit_viewport_frame_folder(self.previous_kit_viewport_frame_folder)
        self.previous_kit_viewport_frame_folder = None
        return self.env.close()

    def step(self, action):
        """Step the environment and, if triggered, record a robot-camera frame."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Start a new robot-camera recording when the trigger fires.
        if not self.recording and self.step_trigger(self.current_step):
            self.recording = True
            self.recorded_steps = 0
            self.robot_camera_frames = []
            self.recording_start_step = self.current_step
            if self.record_kit_viewport:
                self._prepare_kit_viewport_frame_folder()

        # Collect frame while recording.
        if self.recording:
            self._record_robot_camera_frame()
            if self.record_kit_viewport:
                self._record_kit_viewport_frame()
            self.recorded_steps += 1

            # Flush to disk once the clip is complete.
            if self.recorded_steps >= self.video_length:
                robot_video_path = self._save_robot_camera_video()
                kit_viewport_video_path = None
                if self.record_kit_viewport:
                    kit_viewport_video_path = self._save_kit_viewport_video()
                if robot_video_path is not None and kit_viewport_video_path is not None:
                    self._combine_videos_side_by_side(
                        viewport_path=kit_viewport_video_path,
                        robot_path=robot_video_path,
                        output_path=self._combined_video_path(),
                    )
                self.recording = False
                self.robot_camera_frames = []

        self.current_step += 1
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _record_robot_camera_frame(self):
        """Grab a single RGB frame from the robot's onboard camera sensor."""
        try:
            # Walk to the innermost (unwrapped) environment.
            env = self.env
            while hasattr(env, "unwrapped") and env is not env.unwrapped:
                env = env.unwrapped

            if not hasattr(env, "scene"):
                return

            scene = env.scene

            if not hasattr(scene, "sensors"):
                return

            sensors = scene.sensors

            if self.camera_sensor_name not in sensors:
                return

            camera = sensors[self.camera_sensor_name]

            if not hasattr(camera, "data") or not hasattr(camera.data, "output"):
                return

            if "rgb" not in camera.data.output:
                return

            rgb_data = camera.data.output["rgb"]

            # `rgb_data` may be a numpy array, a (possibly GPU) torch.Tensor, or a
            # Warp array on IsaacLab 3.0. Normalise to a CPU numpy array, env index 0.
            if isinstance(rgb_data, np.ndarray):
                frame = rgb_data[0]
            else:
                if not isinstance(rgb_data, torch.Tensor):
                    rgb_data = wp.to_torch(rgb_data)    # Warp array -> torch
                frame = rgb_data[0].detach().cpu().numpy()

            # Normalise to uint8.
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)

            self.robot_camera_frames.append(frame)

        except Exception as exc:    # pylint: disable=broad-except
            if self.recorded_steps == 0:
                print(f"[MultiCameraVideoRecorder][ERROR] Failed to record robot "
                      f"camera frame: {type(exc).__name__}: {exc}")
                traceback.print_exc()

    def _prepare_kit_viewport_frame_folder(self):
        """Create an empty per-clip folder for Kit viewport PNG captures."""
        self._clear_kit_viewport_frame_folder(self.previous_kit_viewport_frame_folder)
        self.previous_kit_viewport_frame_folder = None
        self._clear_kit_viewport_frame_folder()
        self.current_kit_viewport_frame_folder = os.path.join(self.kit_viewport_frame_folder,
                                                              f"step_{self.recording_start_step}")
        os.makedirs(self.current_kit_viewport_frame_folder, exist_ok=True)

    def _clear_kit_viewport_frame_folder(self, frame_folder: Optional[str] = None):
        """Remove temporary Kit viewport PNG captures for one rollout clip."""
        if frame_folder is None:
            frame_folder = self.current_kit_viewport_frame_folder
            self.current_kit_viewport_frame_folder = None

        if frame_folder and os.path.exists(frame_folder):
            try:
                shutil.rmtree(frame_folder)
            except Exception:
                pass

    def _record_kit_viewport_frame(self):
        """Queue one active Kit viewport PNG capture for the current clip."""
        try:
            import asyncio
            import omni.kit.viewport.utility as viewport_utils

            viewport = viewport_utils.get_active_viewport()
            if viewport is None:
                return

            if self.current_kit_viewport_frame_folder is None:
                self._prepare_kit_viewport_frame_folder()

            self._set_kit_viewport_capture_resolution(viewport)

            frame_path = os.path.join(
                self.current_kit_viewport_frame_folder,
                f"frame_{self.recorded_steps:06d}.png",
            )
            if os.path.exists(frame_path):
                os.remove(frame_path)

            capture_helper = viewport_utils.capture_viewport_to_file(viewport, file_path=frame_path)
            if not hasattr(capture_helper, "wait_for_result"):
                return

            wait_task = capture_helper.wait_for_result(completion_frames=1)
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(wait_task)
                finally:
                    asyncio.set_event_loop(None)
                    loop.close()
            else:
                if loop.is_running():
                    asyncio.ensure_future(wait_task)
                else:
                    loop.run_until_complete(wait_task)

        except Exception as exc:    # pylint: disable=broad-except
            if self.recorded_steps == 0:
                print(f"[MultiCameraVideoRecorder][ERROR] Failed to record Kit "
                      f"viewport frame: {type(exc).__name__}: {exc}")
                traceback.print_exc()

    def _set_kit_viewport_capture_resolution(self, viewport):
        """Set active Kit viewport captures to 540 px height."""
        current_width, current_height = viewport.resolution
        target_width = max(2, int(round(current_width * KIT_VIEWPORT_HEIGHT / current_height)))
        viewport.resolution = (target_width, KIT_VIEWPORT_HEIGHT)

    def _read_image_when_ready(self, image_path, timeout=1.0, poll_interval=0.02):
        """Read an image after the writer has produced a stable, non-empty file."""
        deadline = time.monotonic() + timeout
        last_size = None

        while True:
            if not os.path.exists(image_path):
                if time.monotonic() >= deadline:
                    return None
                time.sleep(poll_interval)
                continue

            size = os.path.getsize(image_path)
            if size <= 0:
                if time.monotonic() >= deadline:
                    return None
                time.sleep(poll_interval)
                continue

            if timeout > 0.0 and size != last_size:
                last_size = size
                time.sleep(poll_interval)
                continue

            frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if frame is not None:
                return frame

            if time.monotonic() >= deadline:
                return None
            time.sleep(poll_interval)

    def _save_robot_camera_video(self):
        """Write buffered robot-camera frames to an MP4 file."""
        if not self.robot_camera_frames:
            return None

        first_frame = self.robot_camera_frames[0]
        robot_height, robot_width = first_frame.shape[:2]

        video_filename = f"rl-video-step-{self.recording_start_step}.mp4"
        robot_video_path = os.path.abspath(os.path.join(self.robot_camera_folder, video_filename))

        # Save robot camera video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        robot_writer = cv2.VideoWriter(robot_video_path, fourcc, self.fps,
                                       (robot_width, robot_height))

        if not robot_writer.isOpened():
            print(
                f"[MultiCameraVideoRecorder][ERROR] Could not open VideoWriter for {robot_video_path}"
            )
            return None

        for frame in self.robot_camera_frames:
            # OpenCV expects BGR; the sensor provides RGB.
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            robot_writer.write(frame_bgr)

        robot_writer.release()
        return robot_video_path

    def _save_kit_viewport_video(self):
        """Assemble active Kit viewport PNG captures into an MP4 file."""
        frame_folder = self.current_kit_viewport_frame_folder
        if frame_folder is None or not os.path.isdir(frame_folder):
            return None

        try:
            self._wait_for_kit_viewport_frame_files(frame_folder, self.recorded_steps)
            frame_paths = [
                os.path.join(frame_folder, name)
                for name in sorted(os.listdir(frame_folder))
                if name.endswith(".png")
            ]
            if not frame_paths:
                return None

            video_filename = f"rl-video-step-{self.recording_start_step}.mp4"
            viewport_video_path = os.path.abspath(
                os.path.join(self.kit_viewport_folder, video_filename))

            viewport_writer = None
            frame_count = 0
            try:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                for frame_path in frame_paths:
                    frame = self._read_image_when_ready(frame_path, timeout=0.5)
                    if frame is None:
                        continue

                    frame_height, frame_width = frame.shape[:2]
                    if frame_height != KIT_VIEWPORT_HEIGHT:
                        target_width = max(
                            2, int(round(frame_width * KIT_VIEWPORT_HEIGHT / frame_height)))
                        interpolation = (cv2.INTER_AREA if KIT_VIEWPORT_HEIGHT < frame_height else
                                         cv2.INTER_LINEAR)
                        frame = cv2.resize(frame, (target_width, KIT_VIEWPORT_HEIGHT),
                                           interpolation=interpolation)

                    if viewport_writer is None:
                        viewport_height, viewport_width = frame.shape[:2]
                        viewport_writer = cv2.VideoWriter(viewport_video_path, fourcc, self.fps,
                                                          (viewport_width, viewport_height))

                        if not viewport_writer.isOpened():
                            print(f"[MultiCameraVideoRecorder][ERROR] Could not open "
                                  f"VideoWriter for {viewport_video_path}")
                            return None

                    viewport_writer.write(frame)
                    frame_count += 1
            finally:
                if viewport_writer is not None:
                    viewport_writer.release()

            if frame_count == 0:
                return None
            return viewport_video_path
        finally:
            self.previous_kit_viewport_frame_folder = frame_folder
            self._clear_kit_viewport_frame_folder()

    def _combined_video_path(self):
        """Return the output path for the current side-by-side combined video."""
        return os.path.abspath(
            os.path.join(self.video_folder,
                         f"combined-rl-video-step-{self.recording_start_step}.mp4"))

    def _combine_videos_side_by_side(self, viewport_path: str, robot_path: str, output_path: str):
        """Write a side-by-side video from Kit viewport and robot-camera clips."""
        if not os.path.exists(viewport_path) or not os.path.exists(robot_path):
            return

        viewport_cap = cv2.VideoCapture(viewport_path)
        robot_cap = cv2.VideoCapture(robot_path)
        combined_writer = None

        try:
            if not viewport_cap.isOpened():
                print(f"[MultiCameraVideoRecorder][ERROR] Could not open Kit viewport video: "
                      f"{viewport_path}")
                return
            if not robot_cap.isOpened():
                print(f"[MultiCameraVideoRecorder][ERROR] Could not open robot camera video: "
                      f"{robot_path}")
                return

            viewport_fps = viewport_cap.get(cv2.CAP_PROP_FPS)
            combined_fps = viewport_fps if viewport_fps > 0 else self.fps

            viewport_width = int(viewport_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            viewport_height = int(viewport_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            robot_width = int(robot_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            robot_height = int(robot_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if min(viewport_width, viewport_height, robot_width, robot_height) <= 0:
                return

            scaled_robot_width = max(1, int(round(robot_width * viewport_height / robot_height)))
            combined_size = (viewport_width + scaled_robot_width, viewport_height)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            combined_writer = cv2.VideoWriter(output_path, fourcc, combined_fps, combined_size)
            if not combined_writer.isOpened():
                print(f"[MultiCameraVideoRecorder][ERROR] Could not open VideoWriter for "
                      f"{output_path}")
                return

            last_viewport_frame = None
            last_robot_frame = None
            frame_count = 0

            while True:
                viewport_ok, viewport_frame = viewport_cap.read()
                robot_ok, robot_frame = robot_cap.read()

                if viewport_ok:
                    last_viewport_frame = viewport_frame
                elif last_viewport_frame is not None:
                    viewport_frame = last_viewport_frame

                if robot_ok:
                    last_robot_frame = robot_frame
                elif last_robot_frame is not None:
                    robot_frame = last_robot_frame

                if not viewport_ok and not robot_ok:
                    break
                if viewport_frame is None or robot_frame is None:
                    break

                if viewport_frame.shape[:2] != (viewport_height, viewport_width):
                    viewport_frame = cv2.resize(viewport_frame, (viewport_width, viewport_height))
                robot_frame = cv2.resize(robot_frame, (scaled_robot_width, viewport_height))

                combined_writer.write(np.hstack([viewport_frame, robot_frame]))
                frame_count += 1

            if frame_count == 0 and os.path.exists(output_path):
                os.remove(output_path)
        except Exception as exc:    # pylint: disable=broad-except
            print(f"[MultiCameraVideoRecorder][ERROR] Failed to combine videos: "
                  f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            viewport_cap.release()
            robot_cap.release()
            if combined_writer is not None:
                combined_writer.release()

    def _wait_for_kit_viewport_frame_files(self, frame_folder, expected_count, timeout=2.0):
        """Give asynchronous Kit PNG captures a short window to finish writing."""
        deadline = time.monotonic() + timeout
        expected_count = max(1, expected_count)

        while time.monotonic() < deadline:
            frame_files = [
                name for name in os.listdir(frame_folder)
                if name.endswith(".png") and os.path.getsize(os.path.join(frame_folder, name)) > 0
            ]
            if len(frame_files) >= expected_count:
                return
            time.sleep(0.02)
