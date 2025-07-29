# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

USD_PATHS = {
    'CombinedSingleRack':
        os.path.join(os.path.dirname(__file__), "../usd/combined_simple_warehouse/combined.usd"),
    'CombinedMultiRack':
        os.path.join(os.path.dirname(__file__), "../usd/combined_multi_rack/combined.usd"),
    'GalileoLab':
        os.path.join(os.path.dirname(__file__),
                     "../usd/galileo_lab_no_robot_no_wall/galileo_lab_no_robot_no_wall.usd"),
    'WarehouseSingleRack':
        os.path.join(
            os.path.dirname(__file__),
            "../usd/sample_small_footprint_one_rack_obst_sdg/sample_small_footprint_one_rack_obst_sdg.usd"    #pylint: disable=line-too-long
        ),
    'WarehouseMultiRack':
        f'{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd',
    'SimpleOffice':
        os.path.join(os.path.dirname(__file__), "../usd/office/office.usd"),
    'SimpleWarehouse':
        os.path.join(os.path.dirname(__file__),
                     "../usd/simple_warehouse_no_roof/simple_warehouse_no_roof.usd"),
    'Hospital':
        f'{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd',
}

OMAP_PATHS = {
    'CombinedSingleRack':
        os.path.join(os.path.dirname(__file__),
                     "../usd/combined_simple_warehouse/omap/occupancy_map.yaml"),
    'CombinedMultiRack':
        os.path.join(os.path.dirname(__file__),
                     "../usd/combined_multi_rack/omap/occupancy_map.yaml"),
    'WarehouseSingleRack':
        os.path.join(os.path.dirname(__file__),
                     "../usd/sample_small_footprint_one_rack_obst_sdg/omap/occupancy_map.yaml"),
    'WarehouseMultiRack':
        os.path.join(os.path.dirname(__file__),
                     "../usd/warehouse_multi_rack/omap/occupancy_map.yaml"),
    'SimpleOffice':
        os.path.join(os.path.dirname(__file__), "../usd/office/omap/occupancy_map.yaml"),
    'Hospital':
        os.path.join(os.path.dirname(__file__), "../usd/hospital/omap/occupancy_map.yaml"),
}


@configclass
class EnvSceneAssetCfg(AssetBaseCfg):
    """EnvSceneAssetCfg to add additional scene parameters to AssetBaseCfg.
    """

    # Range for robot pose sampling.
    pose_sample_range = {"x": (-5, 5), "y": (-5, 5), "yaw": (-3.14, 3.14)}

    # Env spacing
    env_spacing = 50

    # Replicate physics in the scene.
    replicate_physics = True


# Adding a USD scene with combined office, galileo lab and warehouse single rack.
combined_single_rack = EnvSceneAssetCfg(
    prim_path="{ENV_REGEX_NS}/CombinedSingleRack",
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0, 0, 0.01),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATHS['CombinedSingleRack'],
        scale=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    pose_sample_range={
        "x": (-10, 10),
        "y": (-12, 17.5),
        "yaw": (-3.14, 3.14)
    },
)

# Adding a USD scene with combined office, galileo lab and warehouse multi rack.
combined_multi_rack = EnvSceneAssetCfg(
    prim_path="{ENV_REGEX_NS}/CombinedMultiRack",
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0, 0, 0.01),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATHS['CombinedMultiRack'],
        scale=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    pose_sample_range={
        "x": (-31.5, 8),
        "y": (-12, 19),
        "yaw": (-3.14, 3.14)
    },
)

# Adding a USD scene for galileo lab
galileo_lab = EnvSceneAssetCfg(
    prim_path="{ENV_REGEX_NS}/GalileoLab",
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0, 0, 0.01),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATHS['GalileoLab'],
        scale=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    env_spacing=20,
)

# Adding a USD scene for warehouse with single rack
warehouse_single_rack = EnvSceneAssetCfg(
    prim_path="{ENV_REGEX_NS}/WarehouseSingleRack",
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0, 0, 0.01),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATHS['WarehouseSingleRack'],
        scale=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
)

# Adding a USD scene for warehouse with multi rack
warehouse_multi_rack = EnvSceneAssetCfg(
    prim_path="{ENV_REGEX_NS}/WarehouseMultiRack",
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0, 0, 0.01),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATHS['WarehouseMultiRack'],
        scale=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    pose_sample_range={
        "x": (-9, 9),
        "y": (-8, 12),
        "yaw": (-3.14, 3.14)
    },
)

# Adding a USD scene for simple office.
simple_office = EnvSceneAssetCfg(
    prim_path="{ENV_REGEX_NS}/SimpleOffice",
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0, 0, 0.01),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATHS['SimpleOffice'],
        scale=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    env_spacing=15,
)

# Adding a USD scene for hospital.
hospital = EnvSceneAssetCfg(
    prim_path="{ENV_REGEX_NS}/Hospital",
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0, 0, 0.01),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATHS['Hospital'],
        scale=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    pose_sample_range={
        "x": (-49, 27),
        "y": (-4, 17),
        "yaw": (-3.14, 3.14)
    },
    env_spacing=80,
)

# Random sample a USD scene from the given list.
random_envs = EnvSceneAssetCfg(
    prim_path="{ENV_REGEX_NS}/RandomEnvs",
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0, 0, 0.01),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    spawn=sim_utils.MultiUsdFileCfg(
        usd_path=[USD_PATHS['SimpleOffice'], USD_PATHS['GalileoLab'], USD_PATHS['SimpleWarehouse']],
        random_choice=True,
        scale=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    replicate_physics=False,
)
