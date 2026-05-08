---
name: compass
description: >
  End-to-end COMPASS robot navigation pipeline orchestrator. Handles SAGE-10k scene
  search/download, USD conversion, COMPASS training environment setup, scene registration,
  residual RL training, and evaluation. Use this skill whenever the user mentions COMPASS,
  SAGE, robot navigation training, USD scene generation, residual RL, embodiment training
  (carter/h1/spot/g1/digit), IsaacLab training, or wants to generate 3D environments for
  robot training. Also triggers for: "train on a new scene", "set up sage", "generate a
  warehouse", "evaluate a checkpoint", "run compass", "search sage-10k", or any navigation
  policy training task.
allowed-tools:
  - Bash
  - Read
  - Edit
  - Write
  - Grep
  - Glob
  - AskUserQuestion
  - WebFetch
  - Agent
---

You orchestrate the COMPASS robot navigation training pipeline — from scene search to trained policy. You're adaptive: interactive and explanatory during setup, autonomous and efficient for repeated operations like training runs.

COMPASS trains cross-embodiment navigation policies using residual RL on top of X-Mobility (a pretrained VLA). SAGE-10k provides 10,000 pre-generated indoor scenes across 50 room types.

## Skill Scripts

This skill bundles two helper scripts in its own directory:
- `scripts/sage10k_search.py` — search SAGE-10k dataset by text query
- `scripts/sage10k_to_usd.py` — convert SAGE-10k scenes to USD (uses only Isaac Sim native APIs, no extra deps)

The scripts are located relative to this skill's base directory. Use the full path when invoking them:
```
<SKILL_BASE_DIR>/scripts/sage10k_search.py
<SKILL_BASE_DIR>/scripts/sage10k_to_usd.py
```

## Workflow Routing

| Workflow | Trigger |
|----------|---------|
| **Search SAGE-10k** | User wants a scene from the SAGE-10k dataset (default for new scenes) |
| **Setup COMPASS** | User wants to install COMPASS deps, or deps are missing when needed |
| **Setup SAGE** | User wants full local SAGE installation for custom generation (rare) |
| **Register Scene** | User has a USD file to add to COMPASS (auto-triggered after conversion) |
| **Train** | User wants to run residual RL training on a scene |
| **Evaluate** | User wants to evaluate a trained checkpoint |
| **Full Pipeline** | User gives a scene description — chain: search SAGE-10k → download → convert USD → verify in viewer → register → preview train (num_envs=1, GUI) → full train |

If the user gives a scene description like "a cluttered warehouse" or "bedroom", treat it as **Full Pipeline** using SAGE-10k search. Only use local SAGE if explicitly requested.

**IMPORTANT**: Before Train, Evaluate, or Full Pipeline, ALWAYS run the Prerequisites Check first. If anything is missing, auto-route to the relevant Setup step. Never assume the environment is ready.

---

## Prerequisites Check

Run this before ANY operation. Use `dangerouslyDisableSandbox: true` for GPU commands. Run all checks in parallel where possible. Report what's ready and what's missing.

```bash
# Step 1: GPU (MUST use dangerouslyDisableSandbox: true)
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

# Step 2-3: Conda env + Isaac Lab
conda env list 2>/dev/null | grep -i isaac
echo "ISAACLAB_PATH=$ISAACLAB_PATH"
test -f "$ISAACLAB_PATH/isaaclab.sh" && echo "isaaclab.sh: OK" || echo "isaaclab.sh: MISSING"

# Step 4: Base policy
for p in ./model.ckpt ./x_mobility_ckpt/model.ckpt ~/afm_base_policy.ckpt; do
  test -f "$p" && echo "Found base policy: $p"
done

# Step 5: USD assets
ls compass/rl_env/exts/mobility_es/mobility_es/usd/ 2>/dev/null
```

Report summary, then proceed or guide setup for missing pieces.

---

## Execution Environment Rules

These rules apply to ALL commands. Violating them causes failures.

1. **Conda env wrapper**: ALL `isaaclab.sh` and Isaac Sim Python commands MUST use:
   ```bash
   conda run -n <ENV_NAME> ${ISAACLAB_PATH}/isaaclab.sh -p ...
   ```
   Or for plain Python with Isaac Sim imports:
   ```bash
   conda run -n <ENV_NAME> python ...
   ```

2. **GPU access**: ALL GPU commands MUST use `dangerouslyDisableSandbox: true`. The Claude Code sandbox blocks NVIDIA driver access — without this flag, `nvidia-smi` fails and Isaac Sim cannot find CUDA.

3. **Background execution**: Training and evaluation are long-running. Use `run_in_background: true` so the user isn't blocked. Note: `conda run` buffers stdout — check progress via log files instead:
   ```bash
   # Find latest log
   ls -t /tmp/isaaclab/logs/ | head -1
   # Or check the Kit log for errors
   find /home/*/miniconda3/envs/*/lib/python*/site-packages/isaacsim/kit/logs/Kit -name "kit_*.log" -newer /tmp/isaaclab/logs/ 2>/dev/null
   ```

4. **Killing processes**: When killing training, use `kill -9` on the Python process directly — the `conda run` wrapper may leave orphan processes:
   ```bash
   ps aux | grep "run.py.*<scene>" | grep -v grep | awk '{print $2}' | xargs kill -9
   ```

---

## Setup COMPASS

Skip steps already detected in Prerequisites Check.

### Step 1: Install Isaac Lab 3.0.0-beta1
```bash
git clone https://github.com/isaac-sim/IsaacLab.git --branch v3.0.0-beta1
cd IsaacLab && ./isaaclab.sh --install
export ISAACLAB_PATH=/path/to/IsaacLab
```

### Step 2: Install COMPASS dependencies
```bash
conda run -n <ENV_NAME> ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -r requirements.txt
conda run -n <ENV_NAME> ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install x_mobility/x_mobility-0.1.0-py3-none-any.whl
conda run -n <ENV_NAME> ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e compass/rl_env/exts/mobility_es/
```

### Step 3: Download base policy
Ask user for existing checkpoint path, or download:
```bash
conda run -n <ENV_NAME> huggingface-cli login --token <HF_TOKEN>
conda run -n <ENV_NAME> huggingface-cli download nvidia/X-Mobility --local-dir ./x_mobility_ckpt
```

### Step 4: Download USD assets
Download `compass_usds.zip` from HuggingFace `nvidia/COMPASS`, extract to `compass/rl_env/exts/mobility_es/mobility_es/usd/`.

---

## Search SAGE-10k (Recommended for new scenes)

The SAGE-10k dataset (`nvidia/SAGE-10k`) has 10,000 pre-generated indoor scenes across 50 room types. No SAGE installation needed.

### Step 1: Search for matching scenes
```bash
conda run -n <ENV_NAME> python <SKILL_BASE_DIR>/scripts/sage10k_search.py "<USER_PROMPT>" --top 5 --sample-size 50
```
First run builds a scene index from the HF API (cached afterward). Results show room type, style, object count, and description.

### Step 2: Present options to user
Show top results. Let user pick one.

### Step 3: Download the selected scene
```bash
# Download specific scene zip
huggingface-cli download nvidia/SAGE-10k --repo-type dataset \
  --include "scenes/<zip_filename>" --local-dir ./sage_10k_cache

# Extract
mkdir -p ./sage_10k_scenes/<scene_name>
unzip ./sage_10k_cache/scenes/<zip_filename> -d ./sage_10k_scenes/<scene_name>/
```

### Step 4: Convert to USD
Use the bundled converter (only needs Isaac Sim native APIs — no extra deps):
```bash
# MUST use dangerouslyDisableSandbox: true
conda run -n <ENV_NAME> python <SKILL_BASE_DIR>/scripts/sage10k_to_usd.py \
  ./sage_10k_scenes/<scene_name>/<layout_json> \
  ./compass/rl_env/exts/mobility_es/mobility_es/usd/<scene_name>/<scene_name>.usd
```
This script:
- Initializes Isaac Sim headless for `pxr` access
- Parses PLY meshes with proper binary format (separate vertex/texcoord elements)
- Scales objects to match layout dimensions
- Creates walls and floors from room geometry
- Applies textures with UV mapping
- Sets all objects as static collision (appropriate for navigation training)

### Step 5: Verify USD in Isaac Sim viewer
**ALWAYS** launch the Isaac Sim viewer after conversion so the user can visually verify the scene looks correct (geometry, collisions, textures) before proceeding. Use `dangerouslyDisableSandbox: true` and `run_in_background: true`:
```bash
conda run -n <ENV_NAME> python -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': False, 'width': 1280, 'height': 720})
import omni
omni.usd.get_context().open_stage('./compass/rl_env/exts/mobility_es/mobility_es/usd/<scene_name>/<scene_name>.usd')
while app.is_running():
    app.update()
app.close()
" &
```
Tell the user the viewer is open and to close it when they're done inspecting. Ask if the scene looks good before proceeding. Enable collision visualization via **Show > Physics > Colliders**.

### Step 6: Register and train
Proceed to **Register Scene** → **Train**.

---

## Register Scene

Edit two files. Read each file first to find insertion points.

### File 1: `compass/rl_env/exts/mobility_es/mobility_es/config/environments.py`

**Add to `USD_PATHS` dict:**
```python
'<SceneName>':
    os.path.join(os.path.dirname(__file__), "../usd/<scene_dir>/<scene_file>.usd"),
```

**Add scene config at end of file:**
```python
<scene_var> = EnvSceneAssetCfg(
    prim_path="{ENV_REGEX_NS}/<SceneName>",
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0, 0, 0.01),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATHS['<SceneName>'],
        scale=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    pose_sample_range={"x": (-3, 3), "y": (-3, 3), "yaw": (-3.14, 3.14)},
    env_spacing=20,
)
```
For SAGE-10k rooms (typically 4-6m), use `pose_sample_range` ±3m and `env_spacing` 20. For larger scenes, increase both.

### File 2: `run.py`

**Add to `EnvSceneAssetCfgMap` (around line 118-128):**
```python
'<scene_var>': environments.<scene_var>,
```

---

## Train

**Remember**: `conda run`, `dangerouslyDisableSandbox: true`.

### Step 1: Preview training (ALWAYS do this first)
**ALWAYS** start with a preview run: `--num_envs 1` WITHOUT `--headless`, so the user can verify the robot spawns correctly in the scene and the environment looks right. Use `run_in_background: true`:
```bash
conda run -n <ENV_NAME> ${ISAACLAB_PATH}/isaaclab.sh -p run.py \
  -c configs/train_config.gin \
  --enable_cameras \
  -o <OUTPUT_DIR>_preview \
  -b <BASE_POLICY_PATH> \
  --logger tensorboard \
  --video \
  --embodiment <EMBODIMENT> \
  --environment <SCENE_KEY> \
  --num_envs 1
```
Tell the user the preview is running with the GUI open. Ask them to verify:
1. The robot spawns in a valid location (not clipping through objects)
2. The scene geometry and collisions look correct
3. The robot can navigate without issues

Once the user confirms the preview looks good, kill the preview process and proceed to full-scale training.

### Step 2: Full-scale training
After user confirms the preview, launch the full training run with `run_in_background: true`:
```bash
conda run -n <ENV_NAME> ${ISAACLAB_PATH}/isaaclab.sh -p run.py \
  -c configs/train_config.gin \
  --enable_cameras \
  -o <OUTPUT_DIR> \
  -b <BASE_POLICY_PATH> \
  -n <WANDB_PROJECT> \
  -r <RUN_NAME> \
  --logger tensorboard \
  --headless \
  --video \
  --embodiment <EMBODIMENT> \
  --environment <SCENE_KEY> \
  --num_envs <NUM_ENVS>
```

Remove `--headless` if the user wants GUI mode. For GUI mode with SAGE-10k scenes (many meshes), use `--num_envs 1` to avoid OOM.

### OSMO cluster submission

The Python launcher in [osmo/run_osmo.py](../../../osmo/run_osmo.py) handles build+push+submit and is the recommended entry point:
```bash
export WANDB_API_KEY=<key>
export HF_TOKEN=<token>
export COMPASS_OSMO_REGISTRY=nvcr.io/<org>/<team>

python osmo/run_osmo.py train \
    --experiment-name <name> \
    --wandb-project <project> \
    --base-policy-ckpt <wandb-artifact>
```

For direct `osmo workflow submit` invocation (advanced), see the [OSMO cloud submission](https://nvlabs.github.io/COMPASS/docs/osmo.html) handbook page. Workflow YAMLs live in [osmo/workflows/](../../../osmo/workflows/).

### Defaults
| Parameter | Default | Source |
|-----------|---------|--------|
| num_iterations | 1000 | train_config.gin |
| num_envs | 64 | train_config.gin |
| num_steps_per_iteration | 256 | shared.gin |
| seed | 20 | train_config.gin |
| embodiment | g1 | shared.gin |

### Valid embodiments
`carter`, `h1`, `spot`, `g1`, `digit`

### Built-in environments
`warehouse_single_rack`, `galileo_lab`, `simple_office`, `combined_single_rack`, `combined_multi_rack`, `random_envs`, `hospital`, `warehouse_multi_rack`

### Outputs
- Checkpoints: `<OUTPUT_DIR>/model_*.pt`
- Videos: `<OUTPUT_DIR>/videos/`
- Logs: TensorBoard in `<OUTPUT_DIR>/` or W&B

---

## Evaluate

```bash
CKPT=$(ls <TRAIN_OUTPUT_DIR>/model_*.pt | sort -V | tail -n 1)

conda run -n <ENV_NAME> ${ISAACLAB_PATH}/isaaclab.sh -p run.py \
  -c configs/eval_config.gin \
  --enable_cameras \
  -o <OUTPUT_DIR> \
  -b <BASE_POLICY_PATH> \
  -p $CKPT \
  -n <WANDB_PROJECT> \
  -r eval_<RUN_NAME> \
  --logger tensorboard \
  --headless \
  --video \
  --video_interval 1 \
  --embodiment <EMBODIMENT> \
  --environment <SCENE_KEY>
```

---

## Visualize Scene in Isaac Sim

To inspect a USD scene before training (check collision meshes, layout, etc.):
```bash
conda run -n <ENV_NAME> python -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': False, 'width': 1280, 'height': 720})
import omni
omni.usd.get_context().open_stage('<USD_PATH>')
while app.is_running():
    app.update()
app.close()
" &
```
Use `dangerouslyDisableSandbox: true`. Enable collision visualization: **Show > Physics > Colliders**.

---

## Setup SAGE (Local — Advanced)

Only needed for fully custom scene generation. SAGE has heavy deps (pytorch3d, TRELLIS, VLM servers) and requires a multi-GPU machine. SAGE-10k search is strongly preferred.

See the SAGE repo README for full setup: https://github.com/NVlabs/sage

---

## Key File Locations

| File | Purpose |
|------|---------|
| `run.py` | Main entry point + `EnvSceneAssetCfgMap` (lines 118-128) |
| `compass/rl_env/exts/mobility_es/mobility_es/config/environments.py` | Scene definitions, `USD_PATHS`, `EnvSceneAssetCfg` class |
| `compass/rl_env/exts/mobility_es/mobility_es/usd/` | USD asset directories |
| `configs/train_config.gin` | Training parameters |
| `configs/eval_config.gin` | Evaluation parameters |
| `configs/shared.gin` | Shared config (embodiment, environment, steps) |
| `osmo/workflows/rl_es_train_workflow.yaml` | OSMO training workflow template |
| `osmo/run_osmo.py` | Python launcher for OSMO submission (train / eval / record / distill) |

## W&B Artifact Format
`<entity>/<project>/<artifact>:<version>` — e.g. `<your-entity>/<your-project>/x_mobility:v1`
