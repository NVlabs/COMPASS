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

"""NuRec Real2Sim scene registry (PhysicalAI-Robotics-NuRec dataset).

Each NuRec scene shares the same flat asset layout under ``usd/<folder>/`` and the
same spawn config; only a few per-scene fields differ. Declare one
:class:`NurecScene` per scene below and the module:

* registers its USD path in ``environments.USD_PATHS`` (keyed by prim leaf),
* registers its occupancy-map + origin convention in ``environments.OMAP_PATHS``
  (consumed by :class:`~mobility_es.utils.occupancy_map.OccupancyMapCollisionChecker`),
* builds an :class:`EnvSceneAssetCfg` exposed via ``nurec_envs`` (keyed by the
  ``--environment`` alias; ``run.py`` merges this into ``EnvSceneAssetCfgMap``).

``environments.py`` imports ``nurec_envs`` from here at the bottom of the file, so
importing ``environments`` is enough to register every NuRec scene.
"""

import os
from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

from mobility_es.config.environments import EnvSceneAssetCfg, OMAP_PATHS, USD_PATHS

_USD_DIR = os.path.join(os.path.dirname(__file__), "../usd")

# Defaults shared by all scenes; override per-scene via NurecScene fields.
DEFAULT_USD_FILE = "particle_spg-runtime.usdz"
DEFAULT_OMAP_FILE = "occupancy_map.yaml"
DEFAULT_ORIGIN_CONVENTION = "bottom-left"
DEFAULT_ENV_SPACING = 500.0


@dataclass
class NurecScene:
    """A single NuRec Real2Sim scene.

    Args:
        folder: Asset sub-folder under ``usd/`` (also the ``--environment`` alias).
        prim_leaf: Prim-path leaf — a valid USD identifier (no hyphens).
        usd_file: Scene USD filename under ``usd/<folder>/``.
        omap_file: Occupancy-map YAML filename under ``usd/<folder>/``.
        origin_convention: Occupancy-map origin convention ("bottom-left" or "top-left").
        env_spacing: Per-env spacing [m]; large for the sizeable Real2Sim scenes.
    """

    folder: str
    prim_leaf: str
    usd_file: str = DEFAULT_USD_FILE
    omap_file: str = DEFAULT_OMAP_FILE
    origin_convention: str = DEFAULT_ORIGIN_CONVENTION
    env_spacing: float = DEFAULT_ENV_SPACING


# Add a scene by appending one self-describing line. Place its assets under
# ``usd/<folder>/`` (the USD file + occupancy_map.yaml + .png).
NUREC_SCENES = [
    NurecScene("nova_carter-galileo", "NovaCarterGalileo_NuRec"),
    NurecScene("nova_carter-cafe", "NovaCarterCafe_NuRec"),
    NurecScene("hand_hold-endeavor-andoria", "HandHoldEndeavorAndoria_NuRec"),
    NurecScene("hand_hold-endeavor-livingroom", "HandHoldEndeavorLivingroom_NuRec"),
    NurecScene("hand_hold-endeavor-wormhole", "HandHoldEndeavorWormhole_NuRec"),
    NurecScene("hand_hold-endeavor-wormhole-table", "HandHoldEndeavorWormholeTable_NuRec"),
    NurecScene("hand_hold-voyager-babyboom", "HandHoldVoyagerBabyboom_NuRec"),
    NurecScene("xgrid-wormhole", "XgridWormhole_NuRec",
               usd_file="stage_particle_with_sim_objects.usd",
               omap_file="occupancy_map_with_sim_objects.yaml"),
]


def make_nurec_env(scene: "NurecScene") -> EnvSceneAssetCfg:
    """Build the shared :class:`EnvSceneAssetCfg` for a NuRec Real2Sim scene."""
    usd_dir = os.path.join(_USD_DIR, scene.folder)
    return EnvSceneAssetCfg(
        prim_path="{ENV_REGEX_NS}/" + scene.prim_leaf,
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0, 0, 0.01),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(usd_dir, scene.usd_file),
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=None,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ),
        env_spacing=scene.env_spacing,
    )


# Register USD/OMAP paths into environments' dicts (OMAP_PATHS is read by the
# OccupancyMapCollisionChecker) and build the cfg objects (keyed by --environment alias).
nurec_envs: dict[str, EnvSceneAssetCfg] = {}
for _scene in NUREC_SCENES:
    _usd_dir = os.path.join(_USD_DIR, _scene.folder)
    USD_PATHS[_scene.prim_leaf] = os.path.join(_usd_dir, _scene.usd_file)
    OMAP_PATHS[_scene.prim_leaf] = {
        "path": os.path.join(_usd_dir, _scene.omap_file),
        "origin_convention": _scene.origin_convention,
    }
    nurec_envs[_scene.folder] = make_nurec_env(_scene)
