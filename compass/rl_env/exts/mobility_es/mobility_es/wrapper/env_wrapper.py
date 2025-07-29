# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

from mobility_es.utils.occupancy_map import OccupancyMapCollisionChecker


class RLESEnvWrapper(ManagerBasedRLEnv):
    """Env wrapper of ManagerBasedRLEnv for RL embodiment specialist.
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode=None):
        """
        Initializes the RLESEnvWrapper.

        Args:
            cfg (ManagerBasedRLEnvCfg): The configuration object for the environment.
            render_mode (str, optional): The render mode for the environment. Defaults to None.

        Returns:
            None
        """
        # Initialize collision checker before ManagerBasedRLEnv to avoid loading error.
        self._collision_checker = OccupancyMapCollisionChecker(cfg.scene)
        super().__init__(cfg, render_mode)

    @property
    def collision_checker(self):
        return self._collision_checker
