# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

COMPASS is a cross-embodiment mobility policy framework: imitation learning + residual RL specialists per embodiment, distilled into one generalist policy, exported to ONNX/TensorRT, deployed via ROS2. Built on Isaac Lab 2.1 / Isaac Sim 4.5 and on top of NVIDIA's [X-Mobility](https://github.com/NVlabs/X-MOBILITY) base policy.

End-to-end pipeline (each stage has a top-level script):
1. `run.py` — train residual RL specialist (or evaluate any policy) in Isaac Lab
2. `record.py` — roll out specialists to collect HDF5 distillation data
3. `distillation_train.py` — distill specialists into one generalist (PyTorch Lightning)
4. `onnx_conversion.py` — export specialist or generalist policy to ONNX + JIT
5. `trt_conversion.py` — convert ONNX → TensorRT engine
6. `ros2_deployment/` — ROS2 nodes that consume the TRT engine in Isaac Sim or on real robots

## Critical: how to invoke Python

Scripts that touch Isaac Lab / Isaac Sim **must** run through the Isaac Lab Python wrapper:

```bash
${ISAACLAB_PATH}/isaaclab.sh -p run.py ...
${ISAACLAB_PATH}/isaaclab.sh -p record.py ...
${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -r requirements.txt
```

Pure-PyTorch scripts can use plain Python:

```bash
python3 distillation_train.py ...
python3 onnx_conversion.py ...
python3 trt_conversion.py ...
python3 scripts/hdf5_to_lerobot_episodic.py ...
```

Environments are created via `python3 -m venv venv && source venv/bin/activate`, but pip installs that need Isaac Lab's site-packages still go through `isaaclab.sh -p -m pip install`.

GPU memory scales with the number of envs in residual RL: 32 envs ≈ 30 GB. Reduce if constrained.

## Repository layout

- `compass/` — main Python package
  - `residual_rl/` — actor-critic that adds a residual on top of the X-Mobility base policy
  - `distillation/` — generalist distillation (PyTorch Lightning trainer, HDF5 data module)
  - `rl_env/exts/mobility_es/` — Isaac Lab extension defining embodiments (H1, Carter, Spot, G1, Digit) and scenes; see `compass/rl_env/README.md` for adding new ones
  - `utils/` — shared helpers
- `configs/` — Gin config files (`train_`, `eval_`, `record_`, `distillation_`, `shared`, `x_mobility_`) plus a YAML template (`distillation_dataset_config_template.yaml`) for selecting specialist checkpoints and split ratios when collecting distillation data
- `x_mobility/x_mobility-0.1.0-py3-none-any.whl` — vendored X-Mobility wheel (installed via pip; imported as `model.x_mobility`)
- `ros2_deployment/` — ROS2 packages, container scripts, and `ISAACSIM_README.md` for deploying COMPASS in Isaac Sim or on real robots
- `docker/` — `Dockerfile.rl` and `Dockerfile.distillation` (PyTorch 24.01 base for distillation; RL training is normally run on the host with Isaac Lab installed)
- `scripts/hdf5_to_lerobot_episodic.py` — converts HDF5 distillation datasets to GR00T LeRobot format

## External assets that must be downloaded manually

- X-Mobility base checkpoint: https://huggingface.co/nvidia/X-Mobility/blob/main/x_mobility-nav2-semantic_action_path.ckpt
- Residual-RL environment USDs: https://huggingface.co/nvidia/COMPASS/blob/main/compass_usds.zip — unzip into `compass/rl_env/exts/mobility_es/mobility_es/usd/`

Most scripts take the X-Mobility checkpoint via `-b/--base-policy-path` and write outputs under `-o/--output-dir`.

## Adding a new embodiment or scene

1. Follow `compass/rl_env/README.md` to extend the Isaac Lab `mobility_es` extension.
2. Register the new entry in `EmbodimentEnvCfgMap` and/or `EnvSceneAssetCfgMap` in `run.py`.
3. Reference it via Gin config or CLI flag.

## Linting and formatting

No test suite. No CI config. Quality is enforced via pre-commit (`.pre-commit-config.yaml`):

- `yapf` — Google style, column limit 100 (`.style.yapf`)
- `pylint` v3.0.3 (`.pylintrc` — many checks disabled; max-line-length 100)
- `clang-format` for C/C++ in ROS2 packages
- `nbstripout`, `trailing-whitespace`, `end-of-file-fixer`, `requirements-txt-fixer`

Run before committing:

```bash
pre-commit run --all-files
```

## Configuration system

Hyperparameters use [`gin-config`](https://github.com/google/gin-config), not argparse. Common pattern: pass one or more `.gin` files via `-c/--config-files`, optionally with command-line overrides. `configs/shared.gin` is included by other configs.

Logging defaults to TensorBoard at `<output_dir>/tensorboard/`. Add `--logger wandb` plus `--wandb-{run,project,entity}-name` flags for W&B.

## Commits

All commits must be DCO-signed (`git commit -s -m "..."`). Unsigned commits are rejected. See `CONTRIBUTING.md`.
