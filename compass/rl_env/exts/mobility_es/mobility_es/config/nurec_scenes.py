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
"""NuRec Real2Sim scene config builders."""

import os
from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

from mobility_es.config.environments import EnvSceneAssetCfg, OMAP_PATHS, USD_PATHS

_USD_DIR = os.path.join(os.path.dirname(__file__), "../usd")
DEFAULT_OMAP_FILE = "occupancy_map.yaml"
DEFAULT_ORIGIN_CONVENTION = "bottom-left"
DEFAULT_ENV_SPACING = 500.0


@dataclass
class NurecScene:
    """A single NuRec Real2Sim scene."""

    folder: str
    prim_leaf: str
    omap_file: str = DEFAULT_OMAP_FILE
    origin_convention: str = DEFAULT_ORIGIN_CONVENTION
    env_spacing: float = DEFAULT_ENV_SPACING


def _make_prim_leaf(scene_name: str) -> str:
    return "".join(
        part[:1].upper() + part[1:] for part in scene_name.replace("-", "_").split("_")) + "_NuRec"


NUREC_SCENE_OVERRIDES = {
    "xgrid-wormhole": {
        "omap_file": "occupancy_map_with_sim_objects.yaml",
    },
}


def _make_nurec_scene(scene_name: str, omap_file: str | None = None) -> NurecScene:
    """Build NuRec scene metadata from a scene folder name."""
    overrides = dict(NUREC_SCENE_OVERRIDES.get(scene_name, {}))
    if omap_file is not None:
        overrides["omap_file"] = omap_file
    return NurecScene(scene_name, _make_prim_leaf(scene_name), **overrides)


def make_nurec_scene_asset_cfg(scene_name: str,
                               usd_file: str,
                               omap_file: str | None = None) -> EnvSceneAssetCfg:
    """Build the shared :class:`EnvSceneAssetCfg` for a NuRec Real2Sim scene."""
    scene = _make_nurec_scene(scene_name, omap_file)
    usd_dir = os.path.join(_USD_DIR, scene.folder)
    usd_path = os.path.join(usd_dir, usd_file)
    omap_path = os.path.join(usd_dir, scene.omap_file)
    USD_PATHS[scene.prim_leaf] = usd_path
    OMAP_PATHS[scene.prim_leaf] = {
        "path": omap_path,
        "origin_convention": scene.origin_convention,
    }
    return EnvSceneAssetCfg(
        prim_path="{ENV_REGEX_NS}/" + scene.prim_leaf,
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0, 0, 0.01),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=None,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ),
        env_spacing=scene.env_spacing,
    )
