# Isaac Lab Extension for Mobility Embodiment Specialist using Residual RL

## Overview

This directory contains Isaac Lab extension for mobility embodiment specialist with residual RL.

## Installation

1. First, install Isaac Lab v1.3.0 and Isaac Sim 4.2 by following the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/v1.3.0/source/setup/installation/index.html).

   The repo has been tested in Isaac Lab v1.3.0, so follow the steps below to set the correct version:
   ```bash
   # Clone the Isaac Lab repository
   git clone git@github.com:isaac-sim/IsaacLab.git

   # Switch to the correct version
   cd IsaacLab
   git fetch origin
   git checkout v1.3.0

   # Install Isaac Lab
   ./isaaclab.sh --install
   ```
   > **NOTE**: There might be error messages about RSL_RL not found during installation, we can safely ignore it as a custimized RSL_RL libary is used for residual RL training.


2. Install the mobility_es extension in this directory after Isaac Lab and Sim are installed:

   ```bash
   # Set Isaac Lab path
   export ISAACLAB_PATH=</path/to/IsaacLab>

   # Install the extension
   ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e exts/mobility_es
   ```

## Usage

### Running the Environment

```bash
${ISAACLAB_PATH}/isaaclab.sh -p scripts/play.py --enable_cameras
```

### Adding New Embodiments

To add a new robot embodiment, follow these steps:

1. **Create Robot Configuration**:
   - Follow Isaac Lab [instructions](https://isaac-sim.github.io/IsaacLab/main/source/how-to/write_articulation_cfg.html) to add a new robot articulation configuration in [exts/mobility_es/mobility_es/config/robots.py](exts/mobility_es/mobility_es/config/robots.py)
   - Ensure proper joint configurations, collision properties, and physics parameters are set

2. **Create Environment Configuration**:
   - Create a new RL environment class that inherits from `GoalReachingEnvCfg` in [exts/mobility_es/mobility_es/config/env_cfg.py](exts/mobility_es/mobility_es/config/env_cfg.py)
   - Override the following components as needed:

   | Component | Description |
   |-----------|-------------|
   | `scene.robot` | Set robot articulation configuration |
   | `scene.camera` | Configure camera settings |
   | `actions` | Define action item for the embodiment |
   | `observations` | Configure RL observations |
   | `events` | Configure RL events |
   | `rewards` | Configure RL rewards |
   | `terminations` | Configure RL terminations |

   > **NOTE**: Each embodiment should define its own action term as a low-level controller that maps velocity commands to joint positions. We have pre-defined action terms for different robot embodiments in [exts/mobility_es/mobility_es/mdp/action](exts/mobility_es/mobility_es/mdp/action) that can be re-configured for new embodiments.

3. **Register and Verify Environment**:
   - Import your new environment in the play.py script [scripts/play.py](scripts/play.py)
   - Verify the environment is functional by running the play.py script with the new environment


3. **Examples of Existing Embodiments**:
- [H1 Humanoid Robot](exts/mobility_es/mobility_es/config/h1_env_cfg.py)
- [Carter Mobile Robot](exts/mobility_es/mobility_es/config/carter_env_cfg.py)
- [Spot Quadruped Robot](exts/mobility_es/mobility_es/config/spot_env_cfg.py)
- [G1 Humanoid Robot](exts/mobility_es/mobility_es/config/g1_env_cfg.py)
- [Digit Humanoid Robot](exts/mobility_es/mobility_es/config/digit_env_cfg.py)


### Adding New Scenes

To add a new scene, follow these steps:

1. **Create Scene Configuration**:
   - Follow Isaac Lab [instructions](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/02_scene/create_scene.html) to add a new scene configuration in [exts/mobility_es/mobility_es/config/environments.py](exts/mobility_es/mobility_es/config/environments.py), make sure to inherit from `EnvSceneAssetCfg` with proper parameters like pose_sample_range, env_spacing, replicate_physics, etc.


2. **Add Occupancy Map [Optional]**:
   - We use occupancy map for collision-free pose sampling, while optional, we recommend adding the occupancy map for the new scene to improve sampling efficiency.
   - For occupancy map generation, follow this [insturction](https://docs.omniverse.nvidia.com/isaacsim/latest/features/ext_omni_isaac_occupancy_map.html) to create the occupancy map with the new scene USD file in Isaac Sim. Make sure to rotate the occupancy map properly to match the scene orientation and set the origin as the top-left of the image in the occupancy_map.yaml file.

    ![Occupancy Map Example](../../images/omap_generation.png)

   - Add the occupancy_map.yaml path in `OMAP_PATHS` in [exts/mobility_es/mobility_es/config/environments.py](exts/mobility_es/mobility_es/config/environments.py) with key as the scene name and value as the occupancy map yaml path.

3. **Register and Verify Scene**:
   - Import your new scene in the play.py script [scripts/play.py](scripts/play.py)
   - Verify the scene is functional by running the play.py script with the new scene by setting `env_cfg.scene.environment = MyNewSceneCfg()`
