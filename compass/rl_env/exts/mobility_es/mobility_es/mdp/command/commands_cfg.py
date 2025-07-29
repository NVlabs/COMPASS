# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from isaaclab.envs.mdp import commands
from isaaclab.utils import configclass
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG

from mobility_es.mdp.command.uniform_collision_free_pose_command import UniformCollisionFreePoseCommand


@configclass
class UniformCollisionFreePose2dCommandCfg(commands.UniformPose2dCommandCfg):
    '''Config for UniformCollisionFreePose2dCommand'''
    class_type: type = UniformCollisionFreePoseCommand

    # Maximum number of trials for the reject sampling.
    max_resample_trial: int = 100

    # Minimum distance to asset for command sampling.
    minimum_distance: float = None

    # Probability of applying minimum distance sampling.
    minimum_distance_prob: float = None

    goal_pose_visualizer_cfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_goal")

    goal_pose_visualizer_cfg.markers["arrow"].scale = (0.1, 0.1, 0.2)
