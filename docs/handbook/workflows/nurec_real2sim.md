# Training & deploying with NuRec Real2Sim scenes

This guide shows how to train and deploy COMPASS navigation policies on
**NuRec Real2Sim** assets in Isaac Lab. Use Docker when possible; use
bare-metal only when Docker is unavailable or when you need to debug the Isaac
Lab checkout directly.

```{note}
Validated on **Ubuntu 22.04** / **OVX with RTX**, with **Isaac Lab commit
`20976357cce6498d4f3db91b18540f3969c84247`** and **Isaac Sim 6.0.1**.
```

## Docker workflow (recommended)

### 1. Prepare prerequisites

- Docker with the NVIDIA Container Toolkit.
- An NVIDIA GPU + driver that satisfies the Isaac Sim 6.0.1 requirements.
- A Hugging Face token with access to the COMPASS and NuRec assets.
- The Hugging Face CLI on the host.

Install the Hugging Face CLI before `source ./docker/activate`:

```bash
python3 -m pip install --user -U 'huggingface_hub[cli]'
```

### 2. Check out COMPASS

```bash
git clone https://github.com/NVlabs/COMPASS.git
cd COMPASS
git fetch
git checkout real2sim/isaaclab_3.0
```

### 3. Download assets and checkpoints

The asset helper is host-side. It downloads COMPASS USDs into the `mobility_es`
USD tree, stores the X-Mobility checkpoint at `./assets/x_mobility.ckpt`, and
installs requested NuRec scenes under
`compass/rl_env/exts/mobility_es/mobility_es/usd/`.

```bash
export HF_TOKEN=hf_xxx

./docker/run.sh assets \
    --nurec-scene nova_carter-galileo \
    --nurec-revision refs/pr/34
```

Use one `--nurec-scene <scene>` flag per scene. The helper excludes
`raw_images.zip`, skips existing assets, and is safe to re-run.

```{note}
You must accept the dataset terms on Hugging Face before downloading.
```

### 4. Build and activate the container

The NuRec image uses Isaac Sim 6.0.1 and builds Isaac Lab from the pinned commit
because there is no matching published Isaac Lab 3.0 GA base image for this
workflow yet.

```bash
export COMPASS_IMAGE_TAG=nurec-lab-3.0-ga
export COMPASS_DOCKERFILE=docker/Dockerfile.rl.isaaclab-3.0-ga

./docker/run.sh build
source ./docker/activate
```

After activation, `python`, `python3`, `pip`, `pytest`, and `isaaclab.sh` run
inside the COMPASS container. The repo remains editable on the host through the
bind mount.

### 5. Verify scene placement

Each NuRec scene should live at:

```text
compass/rl_env/exts/mobility_es/mobility_es/usd/<scene>/
```

The asset helper installs scenes there automatically. A scene folder is flat:

```text
nova_carter-galileo/
├── particle_spg-runtime.usdz
├── particle_sh_optimized.usdz
├── volume.usdz
├── occupancy_map.yaml
├── occupancy_map.png
```

Scenes are registered in `mobility_es/config/nurec_scenes.py` via
`NUREC_SCENES`. Adding a scene is one `NurecScene("<folder>", "<PrimLeaf>")`
entry, with optional `usd_file=`, `omap_file=`, `origin_convention=`, and
`env_spacing=` overrides.

Registered scenes:

| `--environment` | Description |
|---|---|
| `nova_carter-galileo` | Galileo lab |
| `nova_carter-cafe` | NVIDIA cafe |
| `hand_hold-endeavor-andoria` | Endeavor meeting room |
| `hand_hold-endeavor-livingroom` | Endeavor living room |
| `hand_hold-endeavor-wormhole` | Endeavor conference room |
| `hand_hold-endeavor-wormhole-table` | Endeavor conference room with table |
| `hand_hold-voyager-babyboom` | Voyager conference room |

```{note}
NuRec maps use a ROS bottom-left origin convention. The registry sets this by
default; using the wrong convention leads to bad robot spawn locations.
```

### 6. Test the setup

```bash
cd compass/rl_env
python scripts/play.py --visualizer kit
cd -
```

### 7. Train

```bash
python run.py \
    -c configs/train_config_real2sim.gin \
    -o <output_dir> \
    -b ./assets/x_mobility.ckpt \
    --embodiment <embodiment_type> \
    --environment nova_carter-galileo \
    --num_envs 12 \
    --video \
    --video_interval 1 \
    --visualizer kit \
    --precompute_valid_poses
```

Key options:

- `<embodiment_type>`: `h1`, `spot`, `carter`, `g1`, or `digit`.
- `--environment`: any registered NuRec scene.
- `--num_envs 12`: conservative default; increase only after checking VRAM.
- `--precompute_valid_poses`: recommended for constrained Real2Sim scenes.
- With `--visualizer kit`, the GUI uses the Kit perspective camera; debug
  images and policy observations still use the robot camera sensor. For NuRec
  scenes, COMPASS routes that perspective camera through a copied PPISP
  RenderProduct, but keeps the generic Kit camera exposure to avoid clipping.
- Debug dumps include `camera_grid_*.png` for robot-camera RGB/depth and
  `kit_viewport_*.png` plus `kit_viewport_*.png.txt` for the GUI viewport.

Output checkpoints are written as `<output_dir>/model_<iter>.pt`; videos land
in `<output_dir>/videos/`.

```{note}
To run headless, omit `--visualizer kit` or pass `--visualizer None`.
```

```{note}
Approximate per-GPU VRAM for `nova_carter-galileo`, Carter, 320x512 camera:
particle assets use `8.5 GB + 1.0 GB × num_envs`; volume assets use
`7 GB + 1.75 GB × num_envs`. Safe particle defaults are about 18 envs on 32 GB
and 32 envs on 48 GB. Reduce `--num_envs` on OOM.
```

```{note}
The default NuRec asset is `particle_spg-runtime.usdz`. To use `volume.usdz`,
change `DEFAULT_USD_FILE` in `mobility_es/config/nurec_scenes.py` or override
`usd_file=` on one `NurecScene`.
```

### 8. Train on multiple GPUs

```bash
python -m torch.distributed.run --nproc_per_node=<N> \
    run.py --distributed \
    -c configs/train_config_real2sim.gin \
    -o <output_dir> \
    -b ./assets/x_mobility.ckpt \
    --embodiment carter \
    --environment nova_carter-galileo \
    --num_envs <per-GPU> \
    --precompute_valid_poses \
    --video \
    --video_interval 1
```

`--num_envs` is per GPU. Run headless for multi-GPU by omitting
`--visualizer kit`. Rank 0 owns logging, checkpoints, and video.

### 9. Evaluate

```bash
python run.py \
    -c configs/eval_config_real2sim.gin \
    -o <output_dir> \
    -b ./assets/x_mobility.ckpt \
    -p <path/to/residual_policy_ckpt> \
    --embodiment <embodiment_type> \
    --environment nova_carter-galileo \
    --num_envs <num_envs> \
    --video \
    --video_interval 1 \
    --visualizer kit
```

`<path/to/residual_policy_ckpt>` is usually `<output_dir>/model_<iter>.pt`.

### 10. Export and deploy

```bash
cd <output_dir>/
python <path/to/COMPASS>/onnx_conversion.py \
    -b <x_mobility_ckpt> \
    -r <residual_policy_ckpt> \
    -e <embodiment_type> \
    -o <output.onnx> \
    -j <output.jit>

python <path/to/COMPASS>/trt_conversion.py \
    -o <model.onnx> \
    -t <output.engine>
```

Deploy through the
[COMPASS ROS2 Deployment Guide](https://github.com/NVlabs/COMPASS/tree/main/ros2_deployment).

### 11. Troubleshoot

- **Pose sampling fails:** lower collision distances, enable
  `--precompute_valid_poses`, and confirm the occupancy map exists.
- **Robot spawns outside the scene:** check the scene's origin convention.
  NuRec scenes should use `bottom-left`.
- **CUDA OOM:** lower `--num_envs`; NuRec scenes are memory-heavy.

## Bare-metal Isaac Lab workflow

### 1. Prepare prerequisites

- An NVIDIA GPU + driver that satisfies the Isaac Sim 6.0.1 requirements.
- Conda.
- A Hugging Face token with access to the COMPASS and NuRec assets.
- The Hugging Face CLI on `PATH`.

```bash
python3 -m pip install --user -U 'huggingface_hub[cli]'
```

### 2. Install Isaac Lab and Isaac Sim

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 20976357cce6498d4f3db91b18540f3969c84247

./isaaclab.sh --conda env_isaaclab_3.0_compass
conda activate env_isaaclab_3.0_compass

pip install isaacsim[all,extscache]==6.0.1 --extra-index-url https://pypi.nvidia.com
./isaaclab.sh --install
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --visualizer kit
```

```{note}
`./isaaclab.sh --conda` must run under Python ≥ 3.11. Isaac Sim 6.x requires
Python 3.12, which the conda step provisions.
```

### 3. Install COMPASS

```bash
conda deactivate
conda activate env_isaaclab_3.0_compass

git clone https://github.com/NVlabs/COMPASS.git
cd COMPASS
git fetch
git checkout real2sim/isaaclab_3.0

export ISAACLAB_PATH=</path/to/IsaacLab>

${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -r requirements.txt
${ISAACLAB_PATH}/isaaclab.sh -p -m pip install x_mobility/x_mobility-0.1.0-py3-none-any.whl
cd compass/rl_env
${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e exts/mobility_es
cd -
```

### 4. Download assets and checkpoints

Use the same host-side asset helper as Docker:

```bash
export HF_TOKEN=hf_xxx

./docker/run.sh assets \
    --nurec-scene nova_carter-galileo \
    --nurec-revision refs/pr/34
```

It installs scene folders under
`compass/rl_env/exts/mobility_es/mobility_es/usd/` and writes the base policy to
`./assets/x_mobility.ckpt`.

### 5. Test the setup

```bash
cd compass/rl_env
${ISAACLAB_PATH}/isaaclab.sh -p scripts/play.py --visualizer kit
cd -
```

### 6. Train

```bash
${ISAACLAB_PATH:?}/isaaclab.sh -p run.py \
    -c configs/train_config_real2sim.gin \
    -o <output_dir> \
    -b ./assets/x_mobility.ckpt \
    --embodiment <embodiment_type> \
    --environment nova_carter-galileo \
    --num_envs 12 \
    --video \
    --video_interval 1 \
    --visualizer kit \
    --precompute_valid_poses
```

Use the same environment names, checkpoint layout, headless mode, and VRAM
sizing guidance from the Docker workflow.

### 7. Train on multiple GPUs

```bash
${ISAACLAB_PATH:?}/isaaclab.sh -p -m torch.distributed.run --nproc_per_node=<N> \
    run.py --distributed \
    -c configs/train_config_real2sim.gin \
    -o <output_dir> \
    -b ./assets/x_mobility.ckpt \
    --embodiment carter \
    --environment nova_carter-galileo \
    --num_envs <per-GPU> \
    --precompute_valid_poses \
    --video \
    --video_interval 1
```

`--num_envs` is per GPU. Run headless for multi-GPU by omitting
`--visualizer kit`.

### 8. Evaluate

```bash
${ISAACLAB_PATH:?}/isaaclab.sh -p run.py \
    -c configs/eval_config_real2sim.gin \
    -o <output_dir> \
    -b ./assets/x_mobility.ckpt \
    -p <path/to/residual_policy_ckpt> \
    --embodiment <embodiment_type> \
    --environment nova_carter-galileo \
    --num_envs <num_envs> \
    --video \
    --video_interval 1 \
    --visualizer kit
```

### 9. Export and deploy

```bash
cd <output_dir>/
python3 <path/to/COMPASS>/onnx_conversion.py \
    -b <x_mobility_ckpt> \
    -r <residual_policy_ckpt> \
    -e <embodiment_type> \
    -o <output.onnx> \
    -j <output.jit>

python3 <path/to/COMPASS>/trt_conversion.py \
    -o <model.onnx> \
    -t <output.engine>
```

Deploy through the
[COMPASS ROS2 Deployment Guide](https://github.com/NVlabs/COMPASS/tree/main/ros2_deployment).

### 10. Troubleshoot

- **Isaac Sim extension errors:** reinstall with
  `isaacsim[all,extscache]==6.0.1`.
- **Pose sampling fails:** lower collision distances, enable
  `--precompute_valid_poses`, and confirm the occupancy map exists.
- **Robot spawns outside the scene:** NuRec scenes should use `bottom-left`.
- **CUDA OOM:** lower `--num_envs`; remember it is per GPU when distributed.
