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

import os

from omni.isaac.lab.envs import mdp
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

from mobility_es.config import scene_assets
from mobility_es.config import robots
from mobility_es.config.env_cfg import GoalReachingEnvCfg
from mobility_es.mdp.action.locomotion_policy_action import LocomotionPolicyAction, LocomotionPolicyActionCfg


@configclass
class SpotActionsCfg:
    """Action specifications for the MDP."""

    drive_joints = LocomotionPolicyActionCfg(class_type=LocomotionPolicyAction,
                                             asset_name="robot",
                                             policy_ckpt_path=os.path.join(
                                                 os.path.dirname(__file__),
                                                 "../ckpt/spot_locomotion_policy.pt"),
                                             joint_names=[".*"],
                                             scale=0.2,
                                             use_default_offset=True)


@configclass
class SpotGoalReachingEnvCfg(GoalReachingEnvCfg):
    """Configuration for the Spot to reach a 2D target pose environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = robots.spot.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.camera = scene_assets.camera.replace(
            prim_path="{ENV_REGEX_NS}/Robot/body/front_cam")
        self.actions = SpotActionsCfg()
        self.observations.eval.fall_down = ObsTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*leg"]),
                "threshold": 1.0
            },
        )
        self.events.base_external_force_torque = EventTerm(
            func=mdp.apply_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="body"),
                "force_range": (0.0, 0.0),
                "torque_range": (-0.0, 0.0),
            },
        )
        self.terminations.base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*leg"]),
                "threshold": 1.0
            },
        )
