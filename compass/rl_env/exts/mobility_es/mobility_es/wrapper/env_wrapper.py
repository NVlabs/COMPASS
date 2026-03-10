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

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

from mobility_es.utils.occupancy_map import OccupancyMapCollisionChecker


class RLESEnvWrapper(ManagerBasedRLEnv):
    """Env wrapper of ManagerBasedRLEnv for RL embodiment specialist.
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode=None, precompute_valid_poses=False,
                 precompute_valid_orientations=False):
        """
        Initializes the RLESEnvWrapper.

        Args:
            cfg (ManagerBasedRLEnvCfg): The configuration object for the environment.
            render_mode (str, optional): The render mode for the environment. Defaults to None.
            precompute_valid_poses (bool, optional): Whether to precompute valid pose locations.
                Defaults to False.
            precompute_valid_orientations (bool, optional): Whether to use precomputed valid orientations.
                If False, uses randomly generated orientations. Defaults to False.

        Returns:
            None
        """
        # Initialize collision checker before ManagerBasedRLEnv to avoid loading error.
        self._collision_checker = OccupancyMapCollisionChecker(
            cfg.scene, precompute_valid_poses=precompute_valid_poses)
        self._precompute_valid_orientations = precompute_valid_orientations
        super().__init__(cfg, render_mode)

    @property
    def collision_checker(self):
        return self._collision_checker

    @property
    def precompute_valid_orientations(self):
        return self._precompute_valid_orientations
