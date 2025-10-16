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

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.assets.articulation import Articulation

ACTION_HEIGHT = 0.8
ACTION_GAP_HEIGHT = 0.1


class ActionVisualizer:
    """Visualizer of the actions."""

    def __init__(self, cfg, env):
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._base_markers = None
        self._residual_markers = None

    def visualize(self, base_action, residual_action):
        if not self._base_markers or not self._residual_markers:
            self._initialize_action_markers()
        # Visualize base action.
        robot_pos_w = self.robot.data.root_pos_w.clone()
        robot_pos_w[:, 2] += ACTION_HEIGHT
        base_action_arrow_scale, base_action_arrow_quat = self._resolve_xy_velocity_to_arrow(
            base_action[:, :2])
        self._base_markers.visualize(robot_pos_w, base_action_arrow_quat, base_action_arrow_scale)

        # Visualize residual action.
        robot_pos_w[:, 2] += ACTION_GAP_HEIGHT
        residual_action_arrow_scale, residual_action_arrow_quat = self._resolve_xy_velocity_to_arrow(    # pylint: disable=line-too-long
            residual_action[:, :2])
        self._residual_markers.visualize(robot_pos_w, residual_action_arrow_quat,
                                         residual_action_arrow_scale)

    def _initialize_action_markers(self):
        print("Initialize markers for actions.")
        base_marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
        base_marker_cfg.prim_path = "/Visuals/Action/base"
        self._base_markers = VisualizationMarkers(base_marker_cfg)
        self._base_markers.set_visibility(True)

        residual_marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
        residual_marker_cfg.prim_path = "/Visuals/Action/residual"
        self._residual_markers = VisualizationMarkers(residual_marker_cfg)
        self._residual_markers.set_visibility(True)

    def _resolve_xy_velocity_to_arrow(
            self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self._base_markers.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale,
                                   device=xy_velocity.device).repeat(xy_velocity.shape[0], 1)
        # pylint: disable=not-callable
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 5.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
