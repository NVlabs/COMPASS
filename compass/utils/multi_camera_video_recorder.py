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
import time
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

    After recording, automatically combines both videos side-by-side into a
    single video file (``combined-rl-video-step-<N>.mp4``) in the main video
    folder. The separate videos are also kept in their original locations.

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
        self.record_video_wrapper = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=step_trigger,
            video_length=video_length,
            disable_logger=disable_logger,
        )

        super().__init__(self.record_video_wrapper)

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
        self.previous_recording_start_step = None  # Track previous recording to combine when next starts

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
        # Note: We intentionally do NOT reset current_step here, so it persists across resets
        # and continues to increment, matching RecordVideo's step counter.
        return obs

    def close(self):
        """Close the environment and combine the last video if needed."""
        # If we have a recording that finished, try to combine it
        # (RecordVideo might write it on close)
        last_step = None
        if self.recording_start_step > 0 and not self.recording:
            # Current recording finished
            last_step = self.recording_start_step
        elif self.previous_recording_start_step is not None:
            # Previous recording (should have been combined already, but try again)
            last_step = self.previous_recording_start_step

        if last_step is not None:
            # Give RecordVideo a moment to write the video
            time.sleep(0.5)
            # Temporarily set previous_recording_start_step to trigger combination
            original_prev = self.previous_recording_start_step
            self.previous_recording_start_step = last_step
            self._combine_previous_video()
            self.previous_recording_start_step = original_prev

        return self.env.close()

    def step(self, action):
        """Step the environment and, if triggered, record a robot-camera frame."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Start a new robot-camera recording when the trigger fires.
        if not self.recording and self.step_trigger(self.current_step):
            # When a new recording starts, RecordVideo writes the previous video.
            # First, detect what step RecordVideo used for the previous video (just written)
            # and what step it will use for the current recording.
            previous_step, current_step_for_naming = self._detect_recordvideo_step()

            # Update previous_recording_start_step to match what RecordVideo actually used
            # If we detected a previous step, use it; otherwise keep the existing one
            if previous_step is not None:
                self.previous_recording_start_step = previous_step

            # Try to combine the previous recording's videos if we have a previous recording
            # (This is a fallback in case immediate combination failed)
            if self.previous_recording_start_step is not None:
                self._combine_previous_video()

            self.recording = True
            self.recorded_steps = 0
            self.robot_camera_frames = []
            self.recording_start_step = current_step_for_naming
            self.video_index += 1

        # Collect frame while recording.
        if self.recording:
            self._record_robot_camera_frame()
            self.recorded_steps += 1

            # Flush to disk once the clip is complete.
            if self.recorded_steps >= self.video_length:
                # Save robot camera video
                self._save_robot_camera_video()
                # Store the recording start step for combination
                finished_step = self.recording_start_step
                self.previous_recording_start_step = finished_step
                self.recording = False
                self.robot_camera_frames = []

                # Try to combine immediately (RecordVideo writes asynchronously when next recording starts)
                # This is a best-effort attempt - if it fails, we'll retry when next recording starts
                self._try_combine_video(finished_step)

        self.current_step += 1
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_recordvideo_step(self):
        """Detect what step numbers RecordVideo used for previous and current recordings.

        RecordVideo writes the previous video when a new recording starts, so we can
        detect what step it used for the previous video. For the current recording,
        we infer based on whether RecordVideo reset its counter.

        Returns:
            tuple: (previous_step, current_step) where:
                - previous_step: The step number RecordVideo used for the previous video (None if not detected)
                - current_step: The step number RecordVideo will use for the current recording
        """
        try:
            import glob
            import re
            import time

            # Get existing videos before waiting
            existing_videos_before = set(glob.glob(os.path.join(self.video_folder, "rl-video-step-*.mp4")))

            # Wait for RecordVideo to write the previous video (it writes when next recording starts)
            time.sleep(0.5)

            # Get all viewport video files after waiting
            video_files = glob.glob(os.path.join(self.video_folder, "rl-video-step-*.mp4"))
            new_videos = set(video_files) - existing_videos_before

            previous_step = None
            current_step_for_naming = self.current_step

            # Check if a new video was just created (the previous recording's video)
            if new_videos:
                # Extract step number from the newly created video
                for vf in new_videos:
                    match = re.search(r'rl-video-step-(\d+)\.mp4', os.path.basename(vf))
                    if match:
                        previous_step = int(match.group(1))
                        break

                # Determine what step RecordVideo will use for current recording
                if previous_step == 0 and self.current_step > 100:
                    # RecordVideo likely reset - check if there's a large gap
                    all_steps = []
                    for vf2 in video_files:
                        match2 = re.search(r'rl-video-step-(\d+)\.mp4', os.path.basename(vf2))
                        if match2:
                            all_steps.append(int(match2.group(1)))
                    max_step = max(all_steps) if all_steps else 0

                    if max_step < self.current_step - 50:
                        # Large gap detected - RecordVideo reset, will use step 0 again
                        current_step_for_naming = 0
                    else:
                        # No large gap - RecordVideo continuing, will use current_step
                        current_step_for_naming = self.current_step
                else:
                    # Previous video is not step 0, or current_step is low - RecordVideo is continuing
                    current_step_for_naming = self.current_step
            else:
                # No new video detected - might be first recording or RecordVideo hasn't written yet
                # Use current_step as fallback
                current_step_for_naming = self.current_step

        except Exception:
            # Fallback: use current_step
            current_step_for_naming = self.current_step

        return previous_step, current_step_for_naming

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
        """Write buffered frames to an MP4 file. Combination happens later when RecordVideo writes the viewport video."""
        if not self.robot_camera_frames:
            return

        first_frame = self.robot_camera_frames[0]
        robot_height, robot_width = first_frame.shape[:2]

        # Mirror the naming convention used by gymnasium's RecordVideo:
        # ``rl-video-step-<start_step>.mp4``
        video_filename = f"rl-video-step-{self.recording_start_step}.mp4"
        robot_video_path = os.path.abspath(os.path.join(self.robot_camera_folder, video_filename))

        # Save robot camera video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        robot_writer = cv2.VideoWriter(robot_video_path, fourcc, self.fps, (robot_width, robot_height))

        if not robot_writer.isOpened():
            print(f"[MultiCameraVideoRecorder][ERROR] Could not open VideoWriter for {robot_video_path}")
            return

        for frame in self.robot_camera_frames:
            # OpenCV expects BGR; the sensor provides RGB.
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            robot_writer.write(frame_bgr)

        robot_writer.release()
        # Note: Video combination will be attempted immediately and also when the next recording starts.

    def _try_combine_video(self, step):
        """Try to combine videos for a given step immediately after recording finishes.

        This is called right after a recording finishes to try to combine the videos
        as soon as possible, rather than waiting for the next recording to start.
        RecordVideo writes asynchronously, so we retry if the viewport video isn't ready yet.

        Args:
            step: The step number for the video to combine.
        """
        video_filename = f"rl-video-step-{step}.mp4"
        robot_video_path = os.path.abspath(os.path.join(self.robot_camera_folder, video_filename))
        viewport_video_path = os.path.abspath(os.path.join(self.video_folder, video_filename))
        combined_video_path = os.path.abspath(os.path.join(self.video_folder, f"combined-{video_filename}"))

        # Check if combined video already exists
        if os.path.exists(combined_video_path):
            return

        # RecordVideo writes asynchronously, so try multiple times with increasing delays
        max_retries = 20
        retry_delay = 0.2
        viewport_ready = False

        for attempt in range(max_retries):
            if os.path.exists(viewport_video_path):
                try:
                    file_size = os.path.getsize(viewport_video_path)
                    if file_size > 0:
                        test_cap = cv2.VideoCapture(viewport_video_path)
                        if test_cap.isOpened():
                            ret, _ = test_cap.read()
                            test_cap.release()
                            if ret:
                                viewport_ready = True
                                break
                except Exception:
                    pass

            if attempt < max_retries - 1:
                time.sleep(retry_delay)

        if viewport_ready:
            try:
                # Check that robot video exists before combining
                if not os.path.exists(robot_video_path):
                    return

                self._combine_videos_side_by_side(viewport_video_path, robot_video_path, combined_video_path)

                # Verify combined video was created successfully before deleting robot video
                if os.path.exists(combined_video_path) and os.path.getsize(combined_video_path) > 0:
                    # Clean up temporary robot camera video after successful combination
                    try:
                        os.remove(robot_video_path)
                    except Exception:
                        pass
            except Exception:
                # If combination fails, it will be retried when next recording starts
                pass

    def _combine_previous_video(self):
        """Combine the previous recording's viewport and robot camera videos.

        This is called when a new recording starts, because RecordVideo writes
        the previous video when the next recording begins.
        """
        if self.previous_recording_start_step is None:
            return

        video_filename = f"rl-video-step-{self.previous_recording_start_step}.mp4"
        robot_video_path = os.path.abspath(os.path.join(self.robot_camera_folder, video_filename))
        viewport_video_path = os.path.abspath(os.path.join(self.video_folder, video_filename))
        combined_video_path = os.path.abspath(os.path.join(self.video_folder, f"combined-{video_filename}"))

        # Check if combined video already exists (avoid re-combining)
        if os.path.exists(combined_video_path):
            return

        # RecordVideo writes the viewport video when the next recording starts,
        # so it should be available now. Try a few times just in case.
        max_retries = 10
        retry_delay = 0.1
        viewport_ready = False
        actual_viewport_path = viewport_video_path

        for attempt in range(max_retries):
            # Check if viewport video exists and is valid
            if os.path.exists(viewport_video_path):
                try:
                    file_size = os.path.getsize(viewport_video_path)
                    if file_size > 0:
                        # Quick validation - try to open it
                        test_cap = cv2.VideoCapture(viewport_video_path)
                        if test_cap.isOpened():
                            ret, _ = test_cap.read()
                            test_cap.release()
                            if ret:
                                viewport_ready = True
                                actual_viewport_path = viewport_video_path
                                break
                except Exception:
                    pass

            if attempt < max_retries - 1:
                time.sleep(retry_delay)

        if viewport_ready:
            # Check that robot video exists before combining
            if not os.path.exists(robot_video_path):
                return

            try:
                self._combine_videos_side_by_side(actual_viewport_path, robot_video_path, combined_video_path)

                # Verify combined video was created successfully before deleting robot video
                if os.path.exists(combined_video_path) and os.path.getsize(combined_video_path) > 0:
                    # Clean up temporary robot camera video after successful combination
                    try:
                        os.remove(robot_video_path)
                    except Exception:
                        pass
            except Exception as e:
                print(
                    f"[MultiCameraVideoRecorder][ERROR] Failed to combine previous video: {type(e).__name__}: {e}"
                )
                traceback.print_exc()
        # else: viewport video not ready, will retry later

    def _combine_videos_side_by_side(self, viewport_path: str, robot_path: str, output_path: str):
        """Combine viewport and robot camera videos side-by-side into a single video.

        Args:
            viewport_path: Path to the viewport (third-person) video file.
            robot_path: Path to the robot camera video file.
            output_path: Path where the combined video will be saved.
        """
        try:
            # Check if both video files exist
            if not os.path.exists(viewport_path) or not os.path.exists(robot_path):
                return

            # Open both video files
            viewport_cap = cv2.VideoCapture(viewport_path)
            robot_cap = cv2.VideoCapture(robot_path)

            if not viewport_cap.isOpened():
                print(f"[MultiCameraVideoRecorder][ERROR] Could not open viewport video: {viewport_path}")
                return

            if not robot_cap.isOpened():
                print(f"[MultiCameraVideoRecorder][ERROR] Could not open robot camera video: {robot_path}")
                viewport_cap.release()
                return

            # Get video properties
            viewport_fps = viewport_cap.get(cv2.CAP_PROP_FPS)
            robot_fps = robot_cap.get(cv2.CAP_PROP_FPS)
            viewport_frame_count = int(viewport_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            robot_frame_count = int(robot_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Use the viewport FPS (which should match self.fps) for the combined video
            combined_fps = viewport_fps if viewport_fps > 0 else self.fps

            viewport_width = int(viewport_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            viewport_height = int(viewport_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            robot_width = int(robot_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            robot_height = int(robot_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate combined dimensions (side-by-side)
            # Resize robot to match viewport height if needed
            if robot_height != viewport_height:
                scale = viewport_height / robot_height
                new_robot_width = int(robot_width * scale)
                combined_width = viewport_width + new_robot_width
            else:
                combined_width = viewport_width + robot_width
                new_robot_width = robot_width

            combined_height = max(viewport_height, robot_height)

            # Create video writer for combined video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            combined_writer = cv2.VideoWriter(
                output_path, fourcc, combined_fps, (combined_width, combined_height)
            )

            if not combined_writer.isOpened():
                print(f"[MultiCameraVideoRecorder][ERROR] Could not open VideoWriter for {output_path}")
                viewport_cap.release()
                robot_cap.release()
                return

            frame_count = 0
            last_viewport_frame = None
            last_robot_frame = None
            viewport_ended = False
            robot_ended = False

            # Read frames until both videos are exhausted
            while not (viewport_ended and robot_ended):
                # Read viewport frame
                if not viewport_ended:
                    ret_viewport, frame_viewport = viewport_cap.read()
                    if ret_viewport:
                        last_viewport_frame = frame_viewport.copy()
                    else:
                        viewport_ended = True

                # Read robot frame
                if not robot_ended:
                    ret_robot, frame_robot = robot_cap.read()
                    if ret_robot:
                        last_robot_frame = frame_robot.copy()
                    else:
                        robot_ended = True

                # If both videos have ended, stop
                if viewport_ended and robot_ended:
                    break

                # Use last frame if current read failed (repeat last frame for shorter video)
                if viewport_ended and last_viewport_frame is not None:
                    frame_viewport = last_viewport_frame.copy()
                elif viewport_ended:
                    # No frames were ever read from viewport
                    break

                if robot_ended and last_robot_frame is not None:
                    frame_robot = last_robot_frame.copy()
                elif robot_ended:
                    # No frames were ever read from robot
                    break

                # Resize robot frame to match viewport height if needed (maintain aspect ratio)
                if robot_height != viewport_height:
                    frame_robot = cv2.resize(frame_robot, (new_robot_width, viewport_height))

                # Combine frames side-by-side
                combined_frame = np.hstack([frame_viewport, frame_robot])
                combined_writer.write(combined_frame)
                frame_count += 1

            viewport_cap.release()
            robot_cap.release()
            combined_writer.release()

            if frame_count == 0:
                print(
                    f"[MultiCameraVideoRecorder][ERROR] No frames were combined. "
                    f"Check if both videos have valid frames."
                )
                # Clean up empty output file
                if os.path.exists(output_path):
                    os.remove(output_path)
            else:
                # Verify the combined video file was actually created
                if not os.path.exists(output_path):
                    print(
                        f"[MultiCameraVideoRecorder][ERROR] Combined video file was not created: {output_path}"
                    )

        except Exception as e:
            print(
                f"[MultiCameraVideoRecorder][ERROR] Exception in _combine_videos_side_by_side: "
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()
            # Clean up on error
            try:
                if 'viewport_cap' in locals():
                    viewport_cap.release()
                if 'robot_cap' in locals():
                    robot_cap.release()
                if 'combined_writer' in locals():
                    combined_writer.release()
                if os.path.exists(output_path):
                    os.remove(output_path)
            except:
                pass
