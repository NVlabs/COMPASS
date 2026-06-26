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

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg

# Terrain
terrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    debug_vis=False,
)

# Lights
light = AssetBaseCfg(
    prim_path="/World/light",
    spawn=sim_utils.DistantLightCfg(color=(1.0, 1.0, 1.0), intensity=1000.0),
)

# Camera
camera = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/torso_link/front_cam",
    update_period=0.0,
    height=480,
    width=640,
    data_types=["rgb", "depth"],
    spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(intrinsic_matrix=[
        375.1500244140625, 0.0, 317.8176574707031, 0.0, 374.9283142089844, 248.43441772460938, 0.0,
        0.0, 1.0
    ],
                                                           width=640,
                                                           height=480,
                                                           focal_length=24,
                                                           focus_distance=400.0,
                                                           clipping_range=(0.1, 20.0)),
    offset=CameraCfg.OffsetCfg(pos=(0.10434, 0.0, 0.37439),
                               rot=(-0.5, 0.5, -0.5, 0.5),
                               convention="ros"),
)
