# Changelog

All notable changes to this project will be documented in this file.

## [1.6.0] - TBD

### Added
- OSMO cloud submission: `osmo/run_osmo.py` + four workflow YAMLs (train / eval / record / distillation). Assets (X-Mobility base checkpoint, COMPASS USDs) download from HuggingFace inside the workflow; W&B / HF tokens routed via env vars or interactive `--prompt`. Includes `--embodiment` / `--environment` overrides for multi-embodiment training sweeps without YAML hand-edits.
- Multi-GPU residual-RL training with `--distributed`: torchrun-launched, per-rank Isaac Sim instance, manual gradient / KL / metric all-reduce in `PPO.update`, rank-0-only logger / checkpoint / video / episode-log writes. Pairs with `osmo/run_osmo.py --num-gpus {2,8}` and `osmo/workflows/rl_es_train_8gpu_workflow.yaml`.
- Auto-generated occupancy maps from USD: `scripts/generate_omap_from_usd.py` wraps Isaac Sim's `isaacsim.asset.gen.omap`. `OccupancyMapCollisionChecker` falls back to discovering `<usd_dir>/omap/occupancy_map.yaml` when `OMAP_PATHS` has no entry, so SAGE-driven training and new scenes ramp up without hand-tuned omaps.
- Docker-as-venv dev environment: `docker/run.sh` (build / assets / up / down / exec / shell / status subcommands) plus a sourceable `docker/activate` that transparently routes `python` / `pip` / `tensorboard` through the container. Cuts first-time setup from ~30-60 minutes to ~3 minutes.
- COMPASS docs site: Sphinx handbook (`docs/handbook/`) deployed at `nvlabs.github.io/COMPASS/docs/`; academic project page consolidated under `docs/project_page/`. Auto-deployed via `.github/workflows/docs.yml`.
- Agentic skills for Claude Code under `.claude/skills/`: `/compass` (front-door for training / eval / SAGE / OSMO) plus specialty siblings `/compass-deploy` (checkpoint → ONNX → TRT → ROS2), `/compass-debug` (8-check diagnostic with `scripts/compass_status.sh`), and `/compass-newembodiment` (interactive robot-onboarding flow).
- CI: `.github/workflows/pre-commit.yml` runs yapf / pylint / nbstripout / clang-format / large-files / trailing-whitespace / EOF / requirements-txt-fixer on every PR.
- No-regression benchmark gate: `osmo/run_benchmark.py` fires one OSMO eval workflow per scene for a given embodiment (default sweep across 5 scenes); results land in W&B for offline regression assessment.

### Changed
- **Isaac Lab 2.1 → 3.0.0-beta1.** Breaking for downstream consumers of `mobility_es` APIs: `AdditiveUniformNoiseCfg` → `UniformNoiseCfg`; restructured physics via `isaaclab_physx.physics.PhysxCfg`; `rerender_on_reset` → `num_rerenders_on_reset`; `ActionTerm` import moves from `isaaclab.envs.mdp.actions` to `isaaclab.managers`; asset data fields (`root_pos_w`, `root_quat_w`, etc.) require `wp.to_torch()` wrapping; `write_root_state_to_sim()` split into `write_root_pose_to_sim_index()` + `write_root_velocity_to_sim_index()`; quaternion convention flipped wxyz → xyzw across `environments.py` and `NonHolonomicPerfectControlAction` yaw extraction; `velocity_limit` → `velocity_limit_sim` on carter caster joints; `commands.UniformPose2dCommand` used directly (module reorg).
- `--wandb-project-name` is now **required** in `run.py`, `record.py`, and `distillation_train.py` (previously defaulted to internal project names).
- `requirements.txt`: 17 previously-unpinned dependencies pinned to versions verified inside `compass-rl:latest`.
- `README.md` leads with the Docker quick-start; bare-metal install moved under "Manual install".
- Pre-commit `yapf` bumped to v0.43.0 for Python 3.12 compatibility (the previous v0.31.0 pin imports `lib2to3`, removed from the stdlib in 3.13).

### Fixed
- Occupancy-map generation: wait for `omni.usd.get_context().get_stage_loading_status()` to settle (`to_load == 0`) before invoking `Generator.generate2d()` — previously crashed Kit on USDs with external references (e.g. `combined_simple_warehouse` referencing `galileo_lab.usd`).

## [1.5.0] - 2025-07-29

### Added
- Enabled goal heading alignment for object navigation integration, improving navigation accuracy to targets.
- Added ROS2 deployment examples for COMPASS Navigator, including containerized workflows and Isaac Sim integration.
- Added target_pose_generator node to interface with object localization modules to enable object navigation using COMPASS

### Changed
- Upgraded Isaac Lab to version 2.1 for improved simulation capabilities and compatibility with the latest Isaac Sim and ROS2 releases.

### Fixed
- Disable weights only for generalist policy loading

## [1.0.0] - 2025-03-20

Initial release
