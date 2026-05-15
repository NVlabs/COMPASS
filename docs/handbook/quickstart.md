# Quick start

The fastest path to a training shell, using the [Docker-as-venv dev environment](installation/docker.md).

## Prerequisites

- Docker (Engine 24+) with the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- An NVIDIA GPU
- A HuggingFace account with [token](https://huggingface.co/settings/tokens)
  and access to [nvidia/COMPASS](https://huggingface.co/nvidia/COMPASS)

## Three commands

```bash
git clone https://github.com/NVlabs/COMPASS.git && cd COMPASS

export HF_TOKEN=hf_xxx
./docker/run.sh assets         # USDs + X-Mobility ckpt → ./assets/   (~5 min, one-time)
./docker/run.sh build          # build the dev image                  (~10 min, first run)
source ./docker/activate       # venv-like activation (prompt: (compass-rl))
```

:::{admonition} What `./docker/run.sh assets` downloads
:class: tip

The build step doesn't pull these — training does. The downloader is gated
by your HF token.

- **`./assets/usd/`** — `compass_usds.zip` from
  [`nvidia/COMPASS`](https://huggingface.co/nvidia/COMPASS). The training
  scenes (warehouse, office, hospital, …) and embodiment USDs.
- **`./assets/x_mobility.ckpt`** — `x_mobility-nav2-semantic_action_path.ckpt`
  from [`nvidia/X-Mobility`](https://huggingface.co/nvidia/X-Mobility). The
  base policy that the residual head is trained on top of.

`./assets/` is part of the repo bind-mount, so inside the container these
paths are visible at the same `./assets/...` (under `/workspace/COMPASS/`).
Re-running `assets` is idempotent — it's a no-op once the files exist.
:::

## First training step

From the activated shell, kick off a 1-env smoke run to confirm everything wires:

```bash
python run.py \
    -c configs/train_config.gin \
    -o /tmp/out \
    -b ./assets/x_mobility.ckpt \
    --num_envs 1 \
    --enable_cameras \
    --visualizer kit
```

You should see the Kit window pop up, Isaac Sim boot, the scene load, and PPO
step into iteration 0. Drop `--visualizer kit` for a headless smoke (faster,
no GUI overhead).

## Next steps

- **Scale up**: drop `--num_envs 1`. The default scales with available VRAM (~1 GB / env).
- **Other embodiments**: pass `--embodiment {h1,carter,spot,g1,digit}`. See
  [Adding a new embodiment / scene](extending.md) to add your own.
- **Submit to OSMO**: see [OSMO cloud submission](osmo.md).
- **Distil a generalist**: collect rollouts ([recording](workflows/recording.md))
  then [distil](workflows/distillation.md).

## Leaving the dev environment

```bash
deactivate                     # remove the shim PATH; keep the container alive
./docker/run.sh down           # stop the container entirely
```

If you can't run Docker, you'll need to install Isaac Lab v3.0.0-beta1 and
the `mobility_es` extension on the host directly — follow the
[Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta1/source/setup/installation/index.html)
and then `${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e compass/rl_env/exts/mobility_es`.
Docker is the supported / tested path; bare-metal works but isn't documented
end-to-end in the handbook.
