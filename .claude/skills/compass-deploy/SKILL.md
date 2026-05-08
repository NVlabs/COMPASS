---
name: compass-deploy
description: >
  Deploy a TRAINED COMPASS checkpoint to a robot: ONNX export, TensorRT
  engine build, ROS2 launch scaffold. Use whenever the user says deploy,
  export to ONNX, TRT engine, run on a robot, sim2real, ROS2 deployment,
  or has a checkpoint they want to ship. Make sure to use this even when
  the user says things like "how do I get my policy on Carter?" without
  the word "deploy".
allowed-tools:
  - Bash
  - Read
  - Edit
  - Write
  - Grep
  - Glob
---

You take a trained COMPASS policy from a checkpoint (`.pt`) all the way to a runnable robot launch — ONNX export, TensorRT engine build, then a ROS2 launch scaffold pointing at the engine. The user provides the checkpoint; you orchestrate the rest.

This skill assumes the docker-as-venv environment from the `compass` skill — `source ./docker/activate` already done, container up, X-Mobility base policy at `./assets/x_mobility.ckpt`. If the user hasn't set that up, recommend `/compass` first.

## When NOT to use this skill

- The user is still training and hasn't picked a checkpoint yet → `/compass` (training flow).
- The user reports a deploy step is failing without a clear cause → run `/compass-debug` first to surface the root cause.
- The user wants to add a new robot platform (cfg files, embodiment registration) → `/compass-newembodiment`.

---

## Prerequisites Check

```bash
# Activated shell?
command -v deactivate >/dev/null && echo "shell: activated" || echo "NOT activated — run: source ./docker/activate"

# Container running?
./docker/run.sh status

# Base policy present?
test -f ./assets/x_mobility.ckpt && echo "x_mobility ckpt: OK" || echo "MISSING — run: ./docker/run.sh assets"

# Trained checkpoint exists? (user supplies the path)
test -f "<USER_CKPT_PATH>" && echo "ckpt: OK ($(du -h <USER_CKPT_PATH> | cut -f1))" || echo "MISSING ckpt"
```

If anything fails, fix the upstream issue before continuing — pushing forward with a missing ckpt or unactivated shell wastes the user's time.

---

## Workflow

### Step 1: Pick the export branch

The user's checkpoint is one of two kinds. Ask if it isn't obvious from filename or context:

| Checkpoint type | Where it came from | ONNX export flags |
|---|---|---|
| **Specialist (residual)** | A `run.py` training run (per-embodiment) | `--residual-ckpt-path / -r` |
| **Generalist (distilled)** | A `distillation_train.py` run (multi-embodiment) | `--generalist-ckpt-path / -g` plus `--embodiment-type / -e <emb>` |

For a generalist, the `--embodiment-type` flag pins which embodiment's action head to export — pick from `carter`, `h1`, `spot`, `g1`, `digit`. Without it, ONNX export fails because the generalist has multiple action heads and only one can be exported per ONNX file.

### Step 2: ONNX export

Specialist:
```bash
python onnx_conversion.py \
  -b ./assets/x_mobility.ckpt \
  -r <RESIDUAL_CKPT_PATH> \
  -o <OUT_DIR>/policy.onnx
```

Generalist:
```bash
python onnx_conversion.py \
  -b ./assets/x_mobility.ckpt \
  -g <GENERALIST_CKPT_PATH> \
  -e <EMBODIMENT> \
  -o <OUT_DIR>/policy.onnx
```

Optional flags worth surfacing if the user asks:
- `-j <OUT_DIR>/policy.jit` — also export a TorchScript JIT for fallback / debug.
- `-a <wandb-artifact>` — upload the ONNX to W&B as a reusable artifact.
- `-i 224 224` — image size; defaults are usually fine.

### Step 3: TensorRT engine build

```bash
python trt_conversion.py \
  -o <OUT_DIR>/policy.onnx \
  -t <OUT_DIR>/policy.engine
```

The `.engine` file is the runtime artifact — that's what ROS2 loads. Engines are **GPU-architecture-specific**: an engine built on an H100 won't run on a Jetson. Build the engine on the same GPU class as the robot.

### Step 4: ROS2 launch scaffold

The reference launch file is `ros2_deployment/compass_navigator/launch/compass_navigator.launch.py`. It takes the engine via the `runtime_path` launch argument (default: `/tmp/compass_carter.engine`).

For a real-robot bring-up:
```bash
# In a sourced ROS2 workspace
ros2 launch compass_navigator compass_navigator.launch.py \
  runtime_path:=<ABSOLUTE_PATH_TO_ENGINE>
```

For Isaac Sim sim2real testing on the same machine:
```bash
# Terminal 1: bring up the Isaac Sim ROS2 bridge
# (separate process — don't run inside this skill; it competes for the GPU)
./ros2_deployment/launch_compass.sh sim

# Terminal 2: launch the compass navigator
ros2 launch compass_navigator compass_navigator.launch.py \
  runtime_path:=<ABSOLUTE_PATH_TO_ENGINE>
```

Per-platform specifics (Carter vs Spot vs Galileo, sensor remapping, NuRec real2sim setup) are documented in `docs/handbook/deployment/ros2.md`. Point the user there if their target platform isn't generic Carter.

### Step 5: Smoke test (user-driven)

Print the exact launch command for the user's platform and ask them to run it from a separate terminal — bringing up Isaac Sim or a real-robot stack inside this session is risky:
- Isaac Sim launches steal the GPU and block other operations.
- Real-robot bring-up needs hardware ssh / safety-monitor that isn't visible from here.
- Mistakes during deploy can drive a real robot into a wall.

Anti-pattern guard: this skill **does not** invoke `ros2 launch` itself. It produces the engine file and the exact command; the user runs that command in a context where they can monitor and abort.

---

## Common follow-ups

- "It runs but the policy is wrong." Check the `--embodiment-type` flag matches the platform — easy to export with the wrong head. Re-export and rebuild.
- "TRT build fails with an op error." Usually means an Isaac Sim / TensorRT version mismatch. Ask the user which GPU the build is happening on; suggest re-running `./docker/run.sh build` if the image is stale.
- "Engine works in sim but not on the robot." GPU-arch mismatch — rebuild the engine on the robot's GPU class.

For deeper diagnostics, route to `/compass-debug`.

## Key File Locations

| File | Purpose |
|------|---------|
| `onnx_conversion.py` | ONNX export entry point |
| `trt_conversion.py` | TRT engine build |
| `ros2_deployment/compass_navigator/launch/compass_navigator.launch.py` | Reference ROS2 launch file (takes `runtime_path:=`) |
| `ros2_deployment/launch_compass.sh` | Convenience launcher (sim / real / sim2real modes) |
| `docs/handbook/deployment/ros2.md` | Per-platform deployment matrix |
| `./assets/x_mobility.ckpt` | X-Mobility base policy (always required for ONNX export) |
