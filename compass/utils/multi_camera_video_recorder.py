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

import os
import traceback
from typing import Callable, Optional

import cv2
import gymnasium as gym
import numpy as np
import torch


class MultiCameraVideoRecorder(gym.Wrapper):
    """Wrapper that records videos from multiple cameras simultaneously.

    Records from:
      1. Viewport camera (third-person view) — via gymnasium's ``RecordVideo``
         wrapper which is applied internally.
      2. An onboard robot camera sensor — by collecting RGB frames every step
         and writing them to a separate ``robot_camera/`` sub-folder using
         OpenCV.

    Usage example in ``run.py``::

        from compass.utils.multi_camera_video_recorder import MultiCameraVideoRecorder

        env = MultiCameraVideoRecorder(
            env,
            video_folder=os.path.join(args_cli.output_dir, "videos"),
            step_trigger=lambda step: step % (num_steps_per_iteration * args_cli.video_interval) == 0,
            video_length=num_steps_per_iteration,
            disable_logger=True,
            camera_sensor_name="camera",  # name as registered in env.scene.sensors
        )
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        step_trigger: Optional[Callable[[int], bool]] = None,
        video_length: int = 200,
        disable_logger: bool = True,
        camera_sensor_name: str = "camera",
        fps: Optional[float] = None,
    ):
        """Initialise the multi-camera video recorder.

        Args:
            env: The environment to wrap.
            video_folder: Directory where videos will be saved.
                Viewport videos land directly in this folder; robot-camera
                videos land in ``<video_folder>/robot_camera/``.
            step_trigger: Callable ``(global_step: int) -> bool`` that returns
                ``True`` whenever a new recording should start.  Defaults to
                never triggering (i.e. only viewport recording runs).
            video_length: Number of environment steps to record per clip.
            disable_logger: Passed through to ``gym.wrappers.RecordVideo``.
            camera_sensor_name: Key used to look up the camera inside
                ``env.scene.sensors``.
            fps: Frame-rate used when writing the robot-camera video file.
                When ``None`` (default) the value is read from
                ``env.metadata["render_fps"]`` so the robot-camera clip has
                the same duration as the viewport clip produced by
                ``gym.wrappers.RecordVideo``.
        """
        # Wrap the environment with gymnasium's built-in RecordVideo so the
        # viewport (third-person) stream is handled automatically.
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=step_trigger,
            video_length=video_length,
            disable_logger=disable_logger,
        )

        super().__init__(env)

        self.video_folder = video_folder
        self.step_trigger = step_trigger if step_trigger is not None else (lambda step: False)
        self.video_length = video_length
        self.camera_sensor_name = camera_sensor_name

        # Mirror gym.wrappers.RecordVideo: derive FPS from env metadata so
        # both the viewport and robot-camera clips have the same duration.
        if fps is not None:
            self.fps = fps
        else:
            self.fps = self.env.metadata.get("render_fps", 30)

        # Internal state
        self.recording = False
        self.recorded_steps = 0
        self.current_step = 0
        self.robot_camera_frames: list = []
        self.video_index = 0
        self.recording_start_step = 0

        # Create sub-directory for robot-camera output
        self.robot_camera_folder = os.path.join(video_folder, "robot_camera")
        os.makedirs(self.robot_camera_folder, exist_ok=True)

    # ------------------------------------------------------------------
    # gym.Wrapper interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        """Reset environment and clear in-progress recording buffers."""
        obs = self.env.reset(**kwargs)
        self.recording = False
        self.recorded_steps = 0
        self.robot_camera_frames = []
        return obs

    def step(self, action):
        """Step the environment and, if triggered, record a robot-camera frame."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Start a new robot-camera recording when the trigger fires.
        if not self.recording and self.step_trigger(self.current_step):
            self.recording = True
            self.recorded_steps = 0
            self.robot_camera_frames = []
            self.recording_start_step = self.current_step
            self.video_index += 1

        # Collect frame while recording.
        if self.recording:
            self._record_robot_camera_frame()
            self.recorded_steps += 1

            # Flush to disk once the clip is complete.
            if self.recorded_steps >= self.video_length:
                self._save_robot_camera_video()
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
                if self.recorded_steps == 0:
                    print(
                        f"[MultiCameraVideoRecorder][WARNING] Environment has no 'scene' "
                        f"attribute. Available: {dir(env)}"
                    )
                return

            scene = env.scene

            if not hasattr(scene, "sensors"):
                if self.recorded_steps == 0:
                    print(
                        f"[MultiCameraVideoRecorder][WARNING] Scene has no 'sensors' "
                        f"attribute. Available: "
                        f"{[k for k in dir(scene) if not k.startswith('_')]}"
                    )
                return

            sensors = scene.sensors

            if self.camera_sensor_name not in sensors:
                if self.recorded_steps == 0:
                    available = (
                        list(sensors.keys())
                        if hasattr(sensors, "keys")
                        else [k for k in dir(sensors) if not k.startswith("_")]
                    )
                    print(
                        f"[MultiCameraVideoRecorder][WARNING] Camera sensor "
                        f"'{self.camera_sensor_name}' not found. "
                        f"Available sensors: {available}"
                    )
                return

            camera = sensors[self.camera_sensor_name]

            if not hasattr(camera, "data") or not hasattr(camera.data, "output"):
                if self.recorded_steps == 0:
                    print(
                        f"[MultiCameraVideoRecorder][WARNING] Camera "
                        f"'{self.camera_sensor_name}' has no data.output."
                    )
                return

            if "rgb" not in camera.data.output:
                if self.recorded_steps == 0:
                    available = (
                        list(camera.data.output.keys())
                        if hasattr(camera.data.output, "keys")
                        else []
                    )
                    print(
                        f"[MultiCameraVideoRecorder][WARNING] Camera "
                        f"'{self.camera_sensor_name}' has no 'rgb' output. "
                        f"Available outputs: {available}"
                    )
                return

            rgb_data = camera.data.output["rgb"]

            # Support both torch.Tensor and numpy array; always take env index 0.
            if isinstance(rgb_data, torch.Tensor):
                frame = rgb_data[0].cpu().numpy()
            else:
                frame = rgb_data[0]

            # Normalise to uint8.
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)

            self.robot_camera_frames.append(frame)

        except Exception as exc:  # pylint: disable=broad-except
            if self.recorded_steps == 0:
                print(
                    f"[MultiCameraVideoRecorder][ERROR] Failed to record robot "
                    f"camera frame: {type(exc).__name__}: {exc}"
                )
                traceback.print_exc()

    def _save_robot_camera_video(self):
        """Write buffered frames to an MP4 file and clear the buffer."""
        if not self.robot_camera_frames:
            return

        first_frame = self.robot_camera_frames[0]
        height, width = first_frame.shape[:2]

        # Mirror the naming convention used by gymnasium's RecordVideo:
        # ``rl-video-step-<start_step>.mp4``
        video_filename = f"rl-video-step-{self.recording_start_step}.mp4"
        video_path = os.path.join(self.robot_camera_folder, video_filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))

        if not writer.isOpened():
            print(f"[MultiCameraVideoRecorder][ERROR] Could not open VideoWriter for {video_path}")
            return

        for frame in self.robot_camera_frames:
            # OpenCV expects BGR; the sensor provides RGB.
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            writer.write(frame_bgr)

        writer.release()
        print(
            f"[MultiCameraVideoRecorder][INFO] Saved robot-camera video: "
            f"{video_path} ({len(self.robot_camera_frames)} frames)"
        )
