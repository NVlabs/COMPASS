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

import pathlib

from isaaclab.envs import mdp
from isaaclab.sensors import CameraCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from mobility_es.config import scene_assets
from mobility_es.config.robots import digit, DIGIT_ACTUATED_JOINT_NAMES
from mobility_es.config.env_cfg import GoalReachingEnvCfg
from mobility_es.mdp.action.locomotion_policy_action import LocomotionPolicyAction, LocomotionPolicyActionCfg

FILE_DIR = pathlib.Path(__file__).parent


@configclass
class DigitActionsCfg:
    """Action specifications for the MDP."""

    drive_joints = LocomotionPolicyActionCfg(
        class_type=LocomotionPolicyAction,
        asset_name="robot",
        policy_ckpt_path=FILE_DIR / "../ckpt/digit_locomotion_policy.pt",
        joint_names=DIGIT_ACTUATED_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class DigitGoalReachingEnvCfg(GoalReachingEnvCfg):
    """Configuration for the Digit to reach a 2D target pose environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = digit.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.camera = scene_assets.camera.replace(
            prim_path="{ENV_REGEX_NS}/Robot/torso_base/front_cam")
        self.scene.camera.offset = CameraCfg.OffsetCfg(pos=(0.06, 0.0, -0.065),
                                                       rot=(-0.2705, 0.6532, -0.6532, 0.2705981),
                                                       convention="ros")

        self.actions = DigitActionsCfg()
        self.observations.locomotion.joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=DIGIT_ACTUATED_JOINT_NAMES)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        self.observations.locomotion.joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=DIGIT_ACTUATED_JOINT_NAMES)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        self.observations.eval.fall_down = ObsTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_base"]),
                "threshold": 1.0
            },
        )
        self.events.base_external_force_torque = EventTerm(
            func=mdp.apply_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*torso_base"),
                "force_range": (0.0, 0.0),
                "torque_range": (-0.0, 0.0),
            },
        )
        self.terminations.base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_base"]),
                "threshold": 1.0
            },
        )
