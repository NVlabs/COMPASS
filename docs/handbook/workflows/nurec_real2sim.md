# Training & deploying with NuRec Real2Sim scenes

This guide shows how to train and deploy COMPASS navigation policies on **NuRec
Real2Sim** assets — photoreal Gaussian-splat reconstructions of real spaces —
in Isaac Lab. The Real2Sim scenes bridge simulation and reality, enabling
policies trained in sim to transfer zero-shot to real robots.

```{note}
Validated on **Ubuntu 22.04** / **OVX with RTX**, with **Isaac Lab 3.0.0** and
**Isaac Sim 6.0.1**.
```

## Workflow overview

1. **Create workspace** — a `compass-nurec` directory.
2. **Install Isaac Sim & Isaac Lab** (Terminal 1).
3. **Install COMPASS** (Terminal 2).
4. **Authenticate with Hugging Face** (`hf auth login`).
5. **Download assets** — X-Mobility checkpoint, COMPASS USDs, NuRec Real2Sim dataset.
6. **Place NuRec scenes** under the `mobility_es` extension and register them.
7. **Test the setup** with `play.py`.
8. **Train** a residual RL specialist (`train_config_real2sim.gin`).
9. **Evaluate** the trained policy (`eval_config_real2sim.gin`).
10. **Export** to ONNX / TensorRT.
11. **Deploy** via ROS2 / sim-to-real.

## Setup

### Create a workspace

```bash
mkdir compass-nurec
cd compass-nurec
```

### Terminal 1 — Isaac Lab & Isaac Sim

```bash
# 1. Clone Isaac Lab (latest release branch)
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 2. Create + activate a conda environment (Python 3.12)
./isaaclab.sh --conda env_isaaclab_3.0_compass
conda activate env_isaaclab_3.0_compass

# 3. Install Isaac Sim 6.0 via pip (the extscache extra is required so Kit
#    resolves extensions locally instead of pulling them from the registry)
pip install isaacsim[all,extscache]==6.0.1 --extra-index-url https://pypi.nvidia.com

# 4. Install Isaac Lab
./isaaclab.sh --install

# 5. Verify
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --visualizer kit
```

```{note}
`./isaaclab.sh --conda` must run under Python ≥ 3.11 (it uses `tomllib`). If your
shell isn't already in a 3.11+ env, run it from conda `base` first. Isaac Sim 6.x
requires Python **3.12**, which the `--conda` step provisions.
```

### Terminal 2 — COMPASS

```bash
conda deactivate
conda activate env_isaaclab_3.0_compass

# Clone COMPASS and check out the NuRec-compatible branch
git clone https://github.com/NVlabs/COMPASS.git
cd COMPASS
git fetch
git checkout samc/support_nurec_assets_isaaclab_3.0

# Point at your Isaac Lab install
export ISAACLAB_PATH=</path/to/IsaacLab>

# Dependencies + base policy + the mobility_es extension
${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -r requirements.txt
${ISAACLAB_PATH}/isaaclab.sh -p -m pip install x_mobility/x_mobility-0.1.0-py3-none-any.whl
cd compass/rl_env
${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e exts/mobility_es
cd -
```

## Downloading assets & checkpoints

### Authentication

Generate a [Hugging Face access token](https://huggingface.co/docs/hub/security-tokens), then:

```bash
hf auth login --token <generated access token>
```

### 1. Pre-trained X-Mobility checkpoint

```bash
hf download nvidia/X-Mobility x_mobility-nav2-semantic_action_path.ckpt --local-dir <compass-nurec>/X-Mobility
```

(Or download manually from the [X-Mobility model page](https://huggingface.co/nvidia/X-Mobility/blob/main/x_mobility-nav2-semantic_action_path.ckpt).)

### 2. COMPASS USD assets

```bash
hf download nvidia/COMPASS compass_usds.zip --local-dir <compass-nurec>/COMPASS

cd <compass-nurec>/COMPASS
unzip compass_usds.zip
mv groot_mobility_rl_es_usds/usd compass/rl_env/exts/mobility_es/mobility_es/
```

### 3. NuRec Real2Sim assets

```bash
hf download nvidia/PhysicalAI-Robotics-NuRec --repo-type dataset --local-dir <compass-nurec>/PhysicalAI-Robotics-NuRec
```

````{tip}
The full dataset is large. Use `--include` / `--exclude` glob filters to pull only
what you need (quote the patterns so the shell doesn't expand them):

```bash
# A single environment
hf download nvidia/PhysicalAI-Robotics-NuRec --repo-type dataset \
    --local-dir <compass-nurec>/PhysicalAI-Robotics-NuRec \
    --include "nova_carter-galileo/**"

# Multiple environments
hf download nvidia/PhysicalAI-Robotics-NuRec --repo-type dataset \
    --local-dir <compass-nurec>/PhysicalAI-Robotics-NuRec \
    --include "nova_carter-galileo/**" "nova_carter-cafe/**"
```
````

```{note}
You must accept the dataset terms on Hugging Face before downloading.
```

### Place & register a scene

Each NuRec scene ships a **flat** layout. For example, `nova_carter-galileo/`:

```text
nova_carter-galileo/
├── particle_ppispon_spg.usdz   # Gaussian scene (SPG variant, PPISP baked in) — loaded by the env
├── particle_ppispon_sh.usdz    # spherical-harmonics variant (alternative)
├── volume_ppispon_spg.usdz     # volumetric variant (alternative)
├── occupancy_map.yaml          # ROS / bottom-left origin convention
├── occupancy_map.png
├── training_trajectory.png
├── training_trajectory_poses.tum
└── raw_images.zip
```

Move (or symlink) the scene folder into the extension's `usd/` directory:

```bash
# From the COMPASS root
mv <compass-nurec>/PhysicalAI-Robotics-NuRec/nova_carter-galileo \
   compass/rl_env/exts/mobility_es/mobility_es/usd/
```

Scenes are registered in `mobility_es/config/environments.py` via the `NUREC_SCENES`
table — adding a new scene is **one line** (`"<folder>": "<PrimLeaf>"`). The loader
auto-wires `USD_PATHS`/`OMAP_PATHS`, builds the env cfg, and `run.py` exposes it as
`--environment <folder>`.

```{note}
The env loads `particle_ppispon_spg.usdz` and the scene-root `occupancy_map.yaml`,
which uses a ROS **bottom-left** origin convention. The registry sets this
automatically (`NUREC_ORIGIN_CONVENTION = "bottom-left"`). Using the wrong
convention flips the map's Y axis and the robot spawns offset by ~the map height.
```

Scenes registered out of the box (select with `--environment`):

| `--environment` | Description |
|---|---|
| `nova_carter-galileo` | Galileo lab — aisles, shelves, boxes |
| `nova_carter-cafe` | NVIDIA cafe — open area, natural lighting |
| `hand_hold-endeavor-andoria` | Meeting room, Endeavor building |
| `hand_hold-endeavor-livingroom` | Living room, Endeavor building |
| `hand_hold-endeavor-wormhole` | Conference room, Endeavor building |
| `hand_hold-endeavor-wormhole-table` | Conference room (with table), Endeavor |
| `hand_hold-voyager-babyboom` | Conference room, Voyager building |

### Test the setup

```bash
cd compass/rl_env
${ISAACLAB_PATH}/isaaclab.sh -p scripts/play.py --enable_cameras --visualizer kit
cd -
```

## Training the policy

The Real2Sim training config is `configs/train_config_real2sim.gin`, tuned for these scenes:

- **Collision distances** — 0.5 m for both goal and start poses.
- **Precomputed valid poses** — enabled for fast, reliable pose sampling in constrained spaces.
- **Environment spacing** — 500 m to accommodate the large scenes.

Train a residual RL specialist from the COMPASS root:

```bash
${ISAACLAB_PATH:?}/isaaclab.sh -p run.py \
    -c configs/train_config_real2sim.gin \
    -o <output_dir> \
    -b <path/to/x_mobility_ckpt> \
    --embodiment <embodiment_type> \
    --environment nova_carter-galileo \
    --num_envs 64 \
    --video \
    --video_interval 1 \
    --visualizer kit \
    --enable_cameras \
    --precompute_valid_poses
```

Where:

- `<embodiment_type>` — one of `h1`, `spot`, `carter`, `g1`, `digit`.
- `--environment` — any registered NuRec scene (see the table above).
- Checkpoints land in `<output_dir>/checkpoints/model_<iter>.pt`; videos in `<output_dir>/videos/`.

```{note}
To run headless, omit `--visualizer kit` (or pass `--visualizer None`).
```

```{note}
GPU memory scales ~linearly with `--num_envs`; NuRec scenes cost roughly **2×**
the default COMPASS scene (heavier USDs). Empirical fit on an RTX A6000
(Carter + `nova_carter-galileo`): `VRAM ≈ 9 GB + 1.3 GB × num_envs`.

| GPU | VRAM | Safe `--num_envs` (NuRec) |
|---|---|---|
| RTX 5090 | 32 GB | ~14 |
| RTX A6000 / L40 | 48 GB | ~24 |

Reduce `--num_envs` on OOM, or lower the camera resolution in `scene_assets.camera`.
```

### Advanced training options

- **Collision distances** — `goal_pose_collision_distance` / `start_pose_collision_distance` in the gin config.
- **Precompute valid poses** — `precompute_valid_poses = True` (config) or `--precompute_valid_poses`.
- **Precompute orientations** — `precompute_valid_orientations = True` or `--precompute_valid_orientations` (slower; for very tight spaces).
- **Iterations / envs** — `num_iterations` in the config, `--num_envs` on the CLI.

### Multi-GPU (distributed) training

Add `--distributed` and launch `run.py` under `torch.distributed.run` to fan out
across GPUs. Each rank runs its own Isaac Sim instance; gradients/metrics sync via
all-reduce, and rank 0 owns the logger, checkpoints, and video. No NuRec-specific
flags change — your scene/precompute args stay the same.

```bash
${ISAACLAB_PATH:?}/isaaclab.sh -p -m torch.distributed.run --nproc_per_node=<N> \
    run.py --distributed \
    -c configs/train_config_real2sim.gin \
    -o <output_dir> -b <path/to/x_mobility_ckpt> \
    --embodiment carter --environment nova_carter-galileo \
    --num_envs <per-GPU> \
    --enable_cameras --precompute_valid_poses \
    --video --video_interval 1
```

```{warning}
`--num_envs` is the count **per GPU** (total = `nproc_per_node × num_envs`). NuRec
scenes are VRAM-heavy — roughly `VRAM ≈ 9 GB + 1.3 GB × num_envs` per GPU — so the
real2sim config's default `num_envs=64` will **OOM a single GPU**. Set a per-GPU-safe
value: ~24 on an RTX A6000 / L40 (48 GB), ~14 on an RTX 5090 (32 GB).
```

Notes:

- **Run headless** for multi-GPU — omit `--visualizer kit` (you don't want one
  viewport per rank). `--enable_cameras` still drives the RTX renderer for the NuRec
  Gaussian camera sensors, and video / debug images are recorded on **rank 0 only**.
- `--precompute_valid_poses` works as-is; each rank precomputes its own scene's poses.
- `--nproc_per_node=1` (or plain `run.py --distributed`) is a valid single-rank fallback.
- On the cluster, `osmo/run_osmo.py train --num-gpus {2,8}` routes to the matching
  multi-GPU workflow.

## Evaluating the trained policy

```bash
${ISAACLAB_PATH:?}/isaaclab.sh -p run.py \
    -c configs/eval_config_real2sim.gin \
    -o <output_dir> \
    -b <path/to/x_mobility_ckpt> \
    -p <path/to/residual_policy_ckpt> \
    --embodiment <embodiment_type> \
    --environment nova_carter-galileo \
    --num_envs <num_envs> \
    --video \
    --video_interval 1 \
    --enable_cameras \
    --visualizer kit
```

`<path/to/residual_policy_ckpt>` is e.g. `<output_dir>/checkpoints/model_1000.pt`.

## Model export

Export the trained specialist to ONNX + JIT, then optionally to TensorRT:

```bash
cd <output_dir>/
python3 <path/to/COMPASS>/onnx_conversion.py \
    -b <x_mobility_ckpt> -r <residual_policy_ckpt> \
    -e <embodiment_type> -o <output.onnx> -j <output.jit>

python3 <path/to/COMPASS>/trt_conversion.py -o <model.onnx> -t <output.engine>
```

## Deployment

The trained policy deploys via the ROS2 framework — see the
[COMPASS ROS2 Deployment Guide](https://github.com/NVlabs/COMPASS/tree/main/ros2_deployment).
It supports Isaac Sim simulation testing, zero-shot sim-to-real transfer, and object-navigation integration.

For sim-to-real:

1. Export the policy to ONNX / TensorRT (above).
2. Run inference on the robot via the ROS2 deployment framework.
3. Integrate visual SLAM (e.g. cuVSLAM) for state estimation.
4. The policy emits velocity commands from camera observations + goal poses.

## Troubleshooting

**"Failed to sample collision free poses"** — collision checking is too strict for the scene:
- Lower `goal_pose_collision_distance` / `start_pose_collision_distance`.
- Enable `precompute_valid_poses`.
- Confirm the scene's `occupancy_map.yaml` exists and its `origin_convention` is correct (NuRec → `bottom-left`).

**Robot spawns outside the scene** — almost always a wrong `origin_convention`. NuRec/ROS maps
(negative-Y, lower-left origin, `mode: trinary`) are **bottom-left**; bundled COMPASS maps are `top-left`.

**Isaac Sim fails to launch** (`Failed create an extension after pull: omni.grpc.lib`) — the
`extscache` extra is missing; reinstall with `isaacsim[all,extscache]==<version>`.

**High GPU memory** — reduce `--num_envs` (start with 32–64), or lower camera resolution.
