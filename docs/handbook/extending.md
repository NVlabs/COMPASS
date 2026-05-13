# Adding a new embodiment or scene

The Isaac-Lab extension at
[`compass/rl_env/exts/mobility_es/`](https://github.com/NVlabs/COMPASS/tree/main/compass/rl_env/exts/mobility_es)
defines the embodiments (H1, Carter, Spot, G1, Digit, …) and scenes COMPASS
trains on. Adding your own is a registration exercise — the gin configs and
`run.py` driver pick it up once the entry exists.

## Adding new embodiments

To add a new robot embodiment, follow these steps:

### 1. Create robot configuration

- Follow the Isaac Lab [instructions](https://isaac-sim.github.io/IsaacLab/main/source/how-to/write_articulation_cfg.html)
  to add a new robot articulation configuration in
  [`compass/rl_env/exts/mobility_es/mobility_es/config/robots.py`](https://github.com/NVlabs/COMPASS/blob/main/compass/rl_env/exts/mobility_es/mobility_es/config/robots.py).
- Ensure proper joint configurations, collision properties, and physics
  parameters are set.

### 2. Create environment configuration

- Create a new RL environment class that inherits from `GoalReachingEnvCfg`
  in
  [`compass/rl_env/exts/mobility_es/mobility_es/config/env_cfg.py`](https://github.com/NVlabs/COMPASS/blob/main/compass/rl_env/exts/mobility_es/mobility_es/config/env_cfg.py).
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

:::{note}
Each embodiment should define its own action term as a low-level controller
that maps velocity commands to joint positions. We have pre-defined action
terms for different robot embodiments under
[`compass/rl_env/exts/mobility_es/mobility_es/mdp/action`](https://github.com/NVlabs/COMPASS/tree/main/compass/rl_env/exts/mobility_es/mobility_es/mdp/action)
that can be re-configured for new embodiments.
:::

### 3. Register and verify the environment

- Import your new environment in [`run.py`](https://github.com/NVlabs/COMPASS/blob/main/run.py).
- Verify it is functional by running `run.py` with the new environment.

### Examples of existing embodiments

- [H1 humanoid robot](https://github.com/NVlabs/COMPASS/blob/main/compass/rl_env/exts/mobility_es/mobility_es/config/h1_env_cfg.py)
- [Carter mobile robot](https://github.com/NVlabs/COMPASS/blob/main/compass/rl_env/exts/mobility_es/mobility_es/config/carter_env_cfg.py)
- [Spot quadruped robot](https://github.com/NVlabs/COMPASS/blob/main/compass/rl_env/exts/mobility_es/mobility_es/config/spot_env_cfg.py)
- [G1 humanoid robot](https://github.com/NVlabs/COMPASS/blob/main/compass/rl_env/exts/mobility_es/mobility_es/config/g1_env_cfg.py)
- [Digit humanoid robot](https://github.com/NVlabs/COMPASS/blob/main/compass/rl_env/exts/mobility_es/mobility_es/config/digit_env_cfg.py)

## Adding new scenes

### 1. Create scene configuration

- Follow the Isaac Lab
  [scene tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/02_scene/create_scene.html)
  to add a new scene configuration in
  [`compass/rl_env/exts/mobility_es/mobility_es/config/environments.py`](https://github.com/NVlabs/COMPASS/blob/main/compass/rl_env/exts/mobility_es/mobility_es/config/environments.py).
  Inherit from `EnvSceneAssetCfg` with proper parameters like
  `pose_sample_range`, `env_spacing`, `replicate_physics`, etc.

### 2. Add an occupancy map (optional)

We use an occupancy map for collision-free pose sampling. It's optional but
recommended for improving sampling efficiency, especially in tight scenes.

Generate the map directly from the scene USD with the bundled CLI:

```bash
${ISAACLAB_PATH}/isaaclab.sh -p scripts/generate_omap_from_usd.py path/to/scene.usd
```

By default this writes `<scene>.png` and `occupancy_map.yaml` to
`<usd_dir>/omap/`. Use `--cell-size`, `--z-min` / `--z-max`, `--bounds`, or
`--out-dir` to override the defaults; see `--help` for the full flag list.

You can register the YAML path explicitly in `OMAP_PATHS` in
[`compass/rl_env/exts/mobility_es/mobility_es/config/environments.py`](https://github.com/NVlabs/COMPASS/blob/main/compass/rl_env/exts/mobility_es/mobility_es/config/environments.py),
but it's not required: when the scene has no entry, the collision checker
falls back to looking for an `omap/occupancy_map.yaml` next to the scene's
USD — which is exactly where the generator drops it.

The legacy interactive flow via Isaac Sim's
[occupancy-map UI](https://docs.omniverse.nvidia.com/isaacsim/latest/features/ext_omni_isaac_occupancy_map.html)
still works if you prefer authoring a map by hand. Save the YAML/PNG into
the same `<usd_dir>/omap/` location and the loader will pick it up.

See the [Auto OMap from USDs](omap.md) handbook page for the deeper
generator + verifier reference.

### 3. Register and verify the scene

- Import your new scene in [`run.py`](https://github.com/NVlabs/COMPASS/blob/main/run.py).
- Verify the scene is functional by running `run.py` with the new scene
  by setting `env_cfg.scene.environment = MyNewSceneCfg()`.
