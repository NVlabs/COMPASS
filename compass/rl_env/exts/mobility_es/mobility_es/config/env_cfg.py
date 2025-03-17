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
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.markers.config import CUBOID_MARKER_CFG

from mobility_es.config import environments
from mobility_es.config import scene_assets
from mobility_es.mdp.command import commands_cfg
from mobility_es.mdp.observations import camera_img, root_speed, root_yaw_rate, routing_to_goal_simplified
from mobility_es.mdp.rewards import goal_reaching, masked_action_rate_l2, masked_action_l2
from mobility_es.mdp.termination import goal_reached, nan_pose
from mobility_es.mdp.events import reset_root_state_uniform_collision_free
from mobility_es.mdp.curriculum import increase_command_minimum_distance_prob

EPISODE_LENGTH_S = 100


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Common environment configuration with the nova ."""

    # Ground terrain
    terrain = scene_assets.terrain

    # Environment.
    environment = environments.warehouse_single_rack

    # Light.
    light = scene_assets.light

    # Robot
    robot = None

    # Camera
    camera = None

    # Contact force.
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*",
                                      history_length=3,
                                      track_air_time=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class LocomotionCfg(ObsGroup):
        """Observations for locomotion group."""

        # Locomotion state observation terms.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        # Robot states.
        base_speed = ObsTerm(func=root_speed, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_yaw_rate = ObsTerm(func=root_yaw_rate, noise=Unoise(n_min=-0.1, n_max=0.1))
        # Previous action
        actions = ObsTerm(func=mdp.last_action, params={"action_name": "drive_joints"})

        # Route.
        route = ObsTerm(func=routing_to_goal_simplified)

        # RGB image.
        camera_rgb_img = ObsTerm(func=camera_img, params={"data_type": "rgb"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Extra privileged observations for the critic."""

        # Depth image
        camera_depth_img = ObsTerm(func=camera_img, params={"data_type": "depth"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class EvalCfg(ObsGroup):
        '''Observations for evaluation metrics.'''
        # Fall down
        fall_down = ObsTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"),
                "threshold": 1.0
            },
        )
        # Goal reached
        goal_reached = ObsTerm(func=goal_reached)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # Observation groups
    policy: PolicyCfg = PolicyCfg()

    privileged: PrivilegedCfg = PrivilegedCfg()

    locomotion: LocomotionCfg = LocomotionCfg()

    eval: EvalCfg = EvalCfg()


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    goal_pose = commands_cfg.UniformCollisionFreePose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(EPISODE_LENGTH_S, EPISODE_LENGTH_S),
        debug_vis=False,
        simple_heading=False,
        ranges=commands_cfg.UniformCollisionFreePose2dCommandCfg.Ranges(
            pos_x=(-5, 5),
            pos_y=(-5, 5),
            heading=(0.0, 0.0),
        ),
        minimum_distance=2.0,
        minimum_distance_prob=0.9,
        goal_pose_visualizer_cfg=CUBOID_MARKER_CFG.replace(prim_path="/Visuals/Command/pose_goal"))


@configclass
class EventCfg:
    """Configuration for events."""

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=reset_root_state_uniform_collision_free,
        mode="reset",
        params={
            "pose_range": {
                "x": (-5, 5),
                "y": (-5, 5),
                "yaw": (-3.14, 3.14)
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsObsCfg:
    """Reward terms for the MDP."""

    # Reward of the distance to goal.
    dist_to_goal = RewTerm(func=goal_reaching, params={"offset": (0.0, 0.0)}, weight=1.0)

    # Reward of reached the goal.
    goal_reached_reward = RewTerm(
        func=mdp.is_terminated_term,
        weight=2000.0,
        params={"term_keys": "goal_reached"},
    )

    # Penalty of falling down after collision.
    base_contact_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-1000.0,
        params={"term_keys": "base_contact"},
    )

    # Penalty for large action rate
    action_rate_penalty = RewTerm(func=masked_action_rate_l2,
                                  params={"idx_masks": [0, 1, 5]},
                                  weight=0.0)

    # Penalty for large action
    action_penalty = RewTerm(func=masked_action_l2, params={"idx_masks": [0, 1, 5]}, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate if robots falls.
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"),
            "threshold": 1.0
        },
    )

    # Terminate if robots reached the goal.
    goal_reached = DoneTerm(func=goal_reached)

    # Terminate if robots pose is nan.
    nan_pose = DoneTerm(func=nan_pose)


@configclass
class CurriculumCfg:
    ''' Curriculum configs. '''
    command_min_distance_prob = CurrTerm(func=increase_command_minimum_distance_prob,
                                         params={
                                             "minimum_distance_prob_range": (0.1, 0.9),
                                             "total_iterations": 1000,
                                             "num_steps_per_iteration": 256
                                         })


@configclass
class GoalReachingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the  to reach a 2D target pose environment."""

    # Scene settings
    scene = SceneCfg(num_envs=16, env_spacing=20.0)

    # Basic settings
    actions = None
    observations = ObservationsCfg()
    commands = CommandsCfg()

    # MDP settings
    rewards = RewardsObsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 20
        self.episode_length_s = EPISODE_LENGTH_S
        self.rerender_on_reset = True
        # simulation settings
        self.sim.dt = 0.005
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
        self.sim.render_interval = self.decimation
        # Update sensor update.
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
