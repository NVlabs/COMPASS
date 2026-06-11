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
# 1. Clone Isaac Lab and check out the release branch
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout release/3.0.0-beta2

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
git checkout real2sim/isaaclab_3.0

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

Download **directly into the `mobility_es` extension's `usd/` folder** — the dataset's scene folders
sit at the repo root, so they land as `usd/<scene>/...` exactly where the registry expects them, with
no move step afterwards. Run from the COMPASS root:

```bash
hf download nvidia/PhysicalAI-Robotics-NuRec --repo-type dataset \
    --local-dir compass/rl_env/exts/mobility_es/mobility_es/usd \
    --include "nova_carter-galileo/**" \
    --exclude "**/raw_images.zip"
```

````{note}
The dataset is a git repo, so you can pin a specific version with `--revision <tag | branch | commit>`
(defaults to `main`).
For example:

```bash
hf download nvidia/PhysicalAI-Robotics-NuRec --repo-type dataset \
    --revision pr/32 \
    --local-dir compass/rl_env/exts/mobility_es/mobility_es/usd \
    --include "nova_carter-galileo/**" \
    --exclude "**/raw_images.zip"
```

Use tag `pr/32` for a stable, reproducible pull; a branch tracks the latest on that branch.
````

````{tip}
The full dataset is large. Use `--include` / `--exclude` glob filters to pull only
what you need (quote the patterns so the shell doesn't expand them):

```bash
# A single environment
hf download nvidia/PhysicalAI-Robotics-NuRec --repo-type dataset \
    --revision pr/32 \
    --local-dir compass/rl_env/exts/mobility_es/mobility_es/usd \
    --include "nova_carter-galileo/**" \
    --exclude "**/raw_images.zip"

# Multiple environments
hf download nvidia/PhysicalAI-Robotics-NuRec --repo-type dataset \
    --revision pr/32 \
    --local-dir compass/rl_env/exts/mobility_es/mobility_es/usd \
    --include "nova_carter-galileo/**" "nova_carter-cafe/**" \
    --exclude "**/raw_images.zip"
```
````

```{note}
You must accept the dataset terms on Hugging Face before downloading.
```

### Place & register a scene

Each NuRec scene ships a **flat** layout. For example, `nova_carter-galileo/`:

```text
nova_carter-galileo/
├── stage_particle_spg.usdz   # Gaussian scene (SPG variant, PPISP baked in) — loaded by the env
├── stage_particle.usdz       # spherical-harmonics variant (alternative)
├── stage_volume.usdz         # volumetric variant (alternative)
├── occupancy_map.yaml        # ROS / bottom-left origin convention
├── occupancy_map.png
```

Because the download above used `--local-dir .../mobility_es/usd`, each scene already lives at
`compass/rl_env/exts/mobility_es/mobility_es/usd/<scene>/` — **no move/symlink step needed**. (If you
downloaded to a different location, `mv` or symlink the scene folder into that `usd/` directory.)

Scenes are registered in `mobility_es/config/nurec_scenes.py` via the `NUREC_SCENES`
table — adding a new scene is **one line** (`NurecScene("<folder>", "<PrimLeaf>")`, with optional
`usd_file=` / `omap_file=` / `origin_convention=` / `env_spacing=` overrides). The module
auto-wires `USD_PATHS`/`OMAP_PATHS`, builds the env cfg, and `run.py` exposes it as
`--environment <folder>`.

```{note}
The env loads `stage_particle_spg.usdz` and the scene-root `occupancy_map.yaml`,
which uses a ROS **bottom-left** origin convention. The registry sets this
automatically (`DEFAULT_ORIGIN_CONVENTION = "bottom-left"` in `mobility_es/config/nurec_scenes.py`,
overridable per scene via `NurecScene.origin_convention`). Using the wrong
convention lead to robot spawning at wrong location.
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
    --num_envs 12 \
    --video \
    --video_interval 1 \
    --visualizer kit \
    --enable_cameras \
    --precompute_valid_poses
```

Where:

- `<embodiment_type>` — one of `h1`, `spot`, `carter`, `g1`, `digit`.
- `--environment` — any registered NuRec scene (see the table above).
- `--num_envs 12` is a conservative default that fits the smallest supported GPU with the
  default **particle** assets. **Raise it per the VRAM table below** (≈18 on a 32 GB card,
  ≈32 on a 48 GB A6000 / L40). Don't copy a large value blindly — NuRec scenes are memory-heavy,
  and the **volume** asset variant costs far more per env.
- Checkpoints are written directly to `<output_dir>/model_<iter>.pt` — every `ckpt_save_interval`
  iterations (default **50**: `model_0.pt`, `model_50.pt`, …), plus a final one at the end.
  Videos land in `<output_dir>/videos/`. (There is **no** `checkpoints/` subfolder.)

```{note}
To run headless, omit `--visualizer kit` (or pass `--visualizer None`).
```

```{note}
**GPU memory (approximate).** Per-env VRAM scales roughly linearly. The numbers below are a
**single-run** measurement and are rough guides only — your peak will vary with camera resolution,
embodiment, scene, driver version, and other GPU load. Measurement setup:

- **GPU:** RTX A6000 (48 GB), single **dedicated, headless** GPU (no viewport)
- **Run:** `--embodiment carter`, `--environment nova_carter-galileo`, camera 320×512, `--precompute_valid_poses`
- **Value:** peak GPU memory during training, over a ~120 MiB idle baseline

| num_envs | particle (`stage_particle_spg.usdz`) | volume (`stage_volume.usdz`) |
|---|---|---|
| 1  | 9.4 GiB  | 9.0 GiB  |
| 5  | 13.3 GiB | 15.9 GiB |
| 10 | 18.1 GiB | 24.7 GiB |
| 15 | 22.7 GiB | 33.0 GiB |

Linear fits:
- **particle** (default) — `VRAM ≈ 8.5 GB + 1.0 GB × num_envs`
- **volume** — `VRAM ≈ 7 GB + 1.75 GB × num_envs` (lower fixed cost, but ~1.8× the per-env cost)

Safe `--num_envs` (peak ≤ ~85 % of GPU, headless / dedicated GPU):

| GPU | VRAM | particle | volume |
|---|---|---|---|
| RTX 5090 | 32 GB | ~18 | ~10 |
| RTX A6000 / L40 | 48 GB | ~32 | ~18 |

Subtract a couple of envs if the GPU also drives your display. Reduce `--num_envs` on OOM,
or lower the camera resolution in `scene_assets.camera`.
```

````{note}
**Particle vs volume assets.** Every bundled NuRec scene ships two Gaussian variants —
`stage_particle_spg.usdz` (default) and `stage_volume.usdz` — rendering the same scene.
**Particle is recommended for training**: ~1.0 GB/env vs volume's ~1.75 GB/env (they only tie at
a single env, so volume loses badly as you scale).

To switch **all** scenes to the volume variant, change the `DEFAULT_USD_FILE` constant in
`mobility_es/config/nurec_scenes.py`:

```python
DEFAULT_USD_FILE = "stage_particle_spg.usdz"
```

Or switch a **single** scene by setting its `usd_file` on the `NurecScene` entry, e.g.
`NurecScene("nova_carter-galileo", "NovaCarterGalileo_NuRec", usd_file="stage_volume.usdz")`.

The `DEFAULT_USD_FILE` change is a **global** switch — it applies to all registered NuRec scenes, which is fine since
each one ships both files. (The occupancy map is unchanged; only the rendered Gaussian asset differs.)
````

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
scenes are VRAM-heavy (`VRAM ≈ 8.5 GB + 1.0 GB × num_envs` per GPU for the default particle
assets), so the real2sim config's default `num_envs=64` will **OOM a single GPU**. Set a
per-GPU-safe value from the VRAM table above (~32 on a 48 GB A6000 / L40, ~18 on a 32 GB
RTX 5090 with particle; roughly half those with the volume variant).
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

`<path/to/residual_policy_ckpt>` is e.g. `<output_dir>/model_1000.pt`.

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

**High GPU memory / CUDA OOM** — `--num_envs` is the main lever, and NuRec scenes are
memory-heavy (particle: `VRAM ≈ 8.5 GB + 1.0 GB × num_envs`; volume: ~`7 + 1.75 × num_envs`).
Pick a value from the [VRAM table above](#training-the-policy) for your GPU + asset format
(particle: ~18 on a 32 GB card, ~32 on a 48 GB A6000 / L40; ~half that for volume) rather than
a fixed default, and lower it further if you still OOM. With `--distributed`, remember `--num_envs`
is **per GPU**.
