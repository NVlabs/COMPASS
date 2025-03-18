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

from omni.isaac.lab.envs import mdp
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg

from mobility_es.config import scene_assets
from mobility_es.config import robots
from mobility_es.config.env_cfg import GoalReachingEnvCfg
from mobility_es.mdp.action.non_holonomic_perfect_control_action import NonHolonomicPerfectControlAction


@configclass
class CarterActionsCfg:
    """Action specifications for the MDP."""

    drive_joints = mdp.ActionTermCfg(class_type=NonHolonomicPerfectControlAction,
                                     asset_name="robot")


@configclass
class CarterGoalReachingEnvCfg(GoalReachingEnvCfg):
    """Configuration for the carter to reach a 2D target pose environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = robots.carter.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.camera = scene_assets.camera.replace(
            prim_path="{ENV_REGEX_NS}/Robot/chassis_link/front_cam")
        self.actions = CarterActionsCfg()
        self.observations.locomotion = None
        self.events.reset_robot_joints = None
        self.observations.eval.fall_down = ObsTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["chassis_link"]),
                "threshold": 1.0
            },
        )
        self.terminations.base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["chassis_link"]),
                "threshold": 1.0
            },
        )
        self.events.base_external_force_torque = EventTerm(
            func=mdp.apply_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="chassis_link"),
                "force_range": (0.0, 0.0),
                "torque_range": (-0.0, 0.0),
            },
        )
