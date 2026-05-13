---
name: compass
description: >
  Front-door for COMPASS — training, evaluation, SAGE scene workflows
  (search / USD conversion / scene registration), and OSMO cloud
  submission. Use whenever the user mentions training a policy, adding
  a SAGE scene, evaluating a checkpoint, or running COMPASS in general.
  For debug / onboarding-a-new-robot, see the specialty
  siblings: compass-doctor, compass-newembodiment.
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

You orchestrate the COMPASS robot navigation training pipeline — from scene search to trained policy. Adapt to context: interactive and explanatory during setup, autonomous and efficient for repeated operations like training runs.

COMPASS trains cross-embodiment navigation policies using residual RL on top of X-Mobility (a pretrained VLA). SAGE-10k provides 10,000 pre-generated indoor scenes across 50 room types.

## Skill Scripts

This skill bundles two helper scripts in its own directory:
- `scripts/sage10k_search.py` — search SAGE-10k dataset by text query
- `scripts/sage10k_to_usd.py` — convert SAGE-10k scenes to USD (uses only Isaac Sim native APIs, no extra deps)

Use the full path when invoking them:
```
<SKILL_BASE_DIR>/scripts/sage10k_search.py
<SKILL_BASE_DIR>/scripts/sage10k_to_usd.py
```

## Workflow Routing

| Workflow | Trigger |
|----------|---------|
| **Search SAGE-10k** | User wants a scene from the SAGE-10k dataset (default for new scenes) |
| **Setup COMPASS** | User wants to install COMPASS deps, or deps are missing when needed |
| **Setup SAGE (local)** | User wants full local SAGE installation for custom generation (rare; see `references/setup-sage-local.md`) |
| **Register Scene** | User has a USD file to add to COMPASS (auto-triggered after conversion) |
| **Train** | User wants to run residual RL training on a scene |
| **Evaluate** | User wants to evaluate a trained checkpoint |
| **Full Pipeline** | User gives a scene description — chain: search SAGE-10k → download → convert USD → verify in viewer → register → preview train (num_envs=1, GUI) → full train |

If the user gives a scene description like "a cluttered warehouse" or "bedroom", treat it as **Full Pipeline** using SAGE-10k search. Use local SAGE only when the user explicitly asks for it.

Before any of Train, Evaluate, or Full Pipeline, run the **Prerequisites Check** first. The cost of a stale check is a confusing CUDA error 30 minutes into a training run.

### Mandatory visual-verification gates

The Full Pipeline has **two visual checkpoints** that you MUST NOT skip, even when the user has told you to work without clarifying questions:

1. **USD viewer check** after conversion (see *Step 5: Verify USD in Isaac Sim viewer*) — open the headed Isaac Sim viewer on the converted USD and **wait for the user to confirm in writing** ("looks ok" / "looks good" / a description of any issue) before doing scene registration.
2. **`--viz kit` preview training** (see *Train → Step 1: Preview training*) — launch with `num_envs=1 --viz kit` so the Kit window shows the robot in the scene, and **wait for the user's confirmation** before launching the full-scale headless run.

"Work without clarifying questions" applies to *ambiguities about what to do* (scene choice, embodiment, num_envs). It does **not** override these two human-eye verification handoffs. SAGE conversions and Isaac Lab spawn settings can produce silently-broken scenes (rooms missing 3 of 4 walls, robot spawned outside the room, scene upside-down from wrong quaternion) that no programmatic check catches — only a human watching the viewer does. Skipping either step usually costs hours of wasted GPU on a broken scene.

## Specialty skills

Some user intents are better handled by sibling skills. When the user's task fits one of these, **say so to the user and recommend the matching specialty** — don't try to handle it inside `compass`:

| User intent | Specialty skill |
|---|---|
| Diagnose why training won't start, "what's wrong", quick health check | `/compass-doctor` |
| Add a new robot platform (cfg files, EmbodimentEnvCfgMap registration) | `/compass-newembodiment` |

Tell the user the matching specialty and suggest they invoke it (or rephrase their ask so the auto-router picks it up). Don't programmatically invoke the sibling via the Skill tool — that adds latency and removes the user's ability to redirect.

---

## Prerequisites Check

Run this before any operation. The skill assumes the user has run `source ./docker/activate` (the docker-as-venv shim from the `docker/` subdir). Run all checks in parallel where possible. Report what's ready and what's missing.

```bash
# Container running?
./docker/run.sh status

# Activated shell? `deactivate` is a shell function defined by ./docker/activate
command -v deactivate >/dev/null && echo "shell: activated" || echo "shell: NOT activated — run: source ./docker/activate"

# GPU (use dangerouslyDisableSandbox: true)
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

# Assets present
test -f ./assets/x_mobility.ckpt && echo "x_mobility ckpt: OK ($(du -h ./assets/x_mobility.ckpt | cut -f1))"
ls ./assets/usd/ 2>/dev/null | head -5
```

If anything fails, route to **Setup COMPASS**. If the user reports a vague "training won't start" issue, that's `compass-doctor`'s job — recommend it.

---

## Execution Environment Rules

These rules apply to every command. Each one exists for a specific reason; the explanation matters more than the rule itself, because edge cases need judgment.

1. **Activated-shell rule.** Every Python invocation runs inside a shell where the user has already done `source ./docker/activate`. Inside that shell, `python`, `pip`, `tensorboard`, etc. are shims that route into the COMPASS container automatically. The right command is just `python run.py …` — no Isaac Lab launcher prefix, no conda env wrapper. If `command -v deactivate` returns nothing, the shell isn't activated — pause and ask the user to source the activate script before continuing.

2. **GPU access needs `dangerouslyDisableSandbox: true`.** The Claude Code sandbox blocks NVIDIA driver access — without this flag, `nvidia-smi` fails and Isaac Sim can't find CUDA. This is a Claude Code concern, not a container concern, so it applies even though the GPU work happens inside the container.

3. **Background execution for long jobs.** Training and evaluation can run for hours. Use `run_in_background: true` so the user isn't blocked. To check progress, look at log files instead of stdout (the shim forwards stdout but it's easy to miss when the run is backgrounded):
   ```bash
   # Find latest training log
   ls -t /tmp/isaaclab/logs/ | head -1
   # Or check the Kit log for errors
   find ~/.local/share/ov/pkg/isaac-sim-* -name "kit_*.log" 2>/dev/null | head -1 | xargs tail -100
   ```

4. **Killing processes.** `kill -9 <PID>` of the python process. The shim forwards SIGTERM cleanly into the container, no orphan-conda-wrapper issues to worry about:
   ```bash
   ps aux | grep "run.py.*<scene>" | grep -v grep | awk '{print $2}' | xargs kill -9
   ```

---

## Setup COMPASS

The new dev environment uses Docker as a venv: build the image once, download assets once, then activate. Three commands:

```bash
export HF_TOKEN=hf_xxx                  # https://huggingface.co/settings/tokens
./docker/run.sh build                   # ~10 min cold build of compass-rl image
./docker/run.sh assets                  # ~5 min: downloads compass_usds + x_mobility ckpt
source ./docker/activate                # prompt becomes (compass-rl)
```

After this, `python run.py …` (and `pip`, `tensorboard`, etc.) Just Work — they route to the container via `docker exec`. No conda env, no manual pip-install of requirements, no Isaac Lab clone. Assets land at `./assets/usd/` (built-in scenes) and `./assets/x_mobility.ckpt` (base policy), bind-mounted into the container at the same paths under `/workspace/COMPASS/`.

Full reference: `docs/handbook/installation/docker.md`. If the user wants the bare-metal install (no Docker), point them at `docs/handbook/installation/bare-metal.md` — supported but slower to set up and not the recommended default.

---

## Search SAGE-10k (recommended for new scenes)

The SAGE-10k dataset (`nvidia/SAGE-10k`) has 10,000 pre-generated indoor scenes across 50 room types. No SAGE installation needed.

### Step 1: Search for matching scenes

```bash
python <SKILL_BASE_DIR>/scripts/sage10k_search.py "<USER_PROMPT>" --top 5 --sample-size 50
```

First run builds a scene index from the HF API (cached afterward). Results show room type, style, object count, and description.

### Step 2: Present options to user

Show top results. Let user pick one.

### Step 3: Download the selected scene

```bash
huggingface-cli download nvidia/SAGE-10k --repo-type dataset \
  --include "scenes/<zip_filename>" --local-dir ./sage_10k_cache

mkdir -p ./sage_10k_scenes/<scene_name>
unzip ./sage_10k_cache/scenes/<zip_filename> -d ./sage_10k_scenes/<scene_name>/
```

### Step 4: Convert to USD

The bundled converter only needs Isaac Sim native APIs, no extra deps:

```bash
# dangerouslyDisableSandbox: true (Isaac Sim needs GPU)
python <SKILL_BASE_DIR>/scripts/sage10k_to_usd.py \
  ./sage_10k_scenes/<scene_name>/<layout_json> \
  ./compass/rl_env/exts/mobility_es/mobility_es/usd/<scene_name>/<scene_name>.usd
```

The output USD lives under the mobility_es extension directory because that's where `environments.py` expects to find it (the registered `USD_PATHS` dict resolves `../usd/<scene>/...` relative to the config/ subdir). Built-in COMPASS scenes from `./assets/usd/` are referenced separately; user-added SAGE scenes go here.

The script:
- Initializes Isaac Sim headless for `pxr` access
- Parses PLY meshes with proper binary format (separate vertex/texcoord elements)
- Scales objects to match layout dimensions
- Creates walls and floors from room geometry
- Applies textures with UV mapping
- Sets all objects as static collision (appropriate for navigation training)

### Step 5: Verify USD in Isaac Sim viewer  **(MANDATORY GATE)**

This step is **not optional and cannot be skipped**, including under "no clarifying questions" mode — see [Mandatory visual-verification gates](#mandatory-visual-verification-gates).

Launch the Isaac Sim viewer on the converted USD so the user can verify the scene looks correct (geometry, collisions, textures). Use `dangerouslyDisableSandbox: true` and `run_in_background: true`:

```bash
python -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': False, 'width': 1280, 'height': 720})
import omni
omni.usd.get_context().open_stage('./compass/rl_env/exts/mobility_es/mobility_es/usd/<scene_name>/<scene_name>.usd')
while app.is_running():
    app.update()
app.close()
" &
```

Then:
1. Tell the user the viewer is open on display `:1` and to inspect it.
2. Suggest enabling **Show → Physics → Colliders** to verify collision meshes.
3. **Stop and wait for the user's explicit confirmation.** Do not proceed to Step 6 until they reply with "looks ok"/"looks good" (or describe an issue to fix).

If they describe an issue, diagnose it (re-convert, adjust converter, etc.) and re-open the viewer for another round of verification. Repeat until they confirm.

### Step 6: Register and train

Only after the user has explicitly confirmed the USD in Step 5, proceed to **Register Scene** → **Train**.

---

## Register Scene

Edit two files. Read each first to find the right insertion point — line numbers drift over time.

### File 1: `compass/rl_env/exts/mobility_es/mobility_es/config/environments.py`

Add to `USD_PATHS` dict:
```python
'<SceneName>':
    os.path.join(os.path.dirname(__file__), "../usd/<scene_dir>/<scene_file>.usd"),
```

Add scene config at end of file:
```python
<scene_var> = EnvSceneAssetCfg(
    prim_path="{ENV_REGEX_NS}/<SceneName>",
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(0, 0, 0.01),
        # Isaac Lab 3.0 quaternion is (x, y, z, w) — w LAST. Identity is
        # (0,0,0,1). Do NOT use (1,0,0,0): that's a 180° flip about X and
        # turns the scene upside-down. Existing scenes in this file use
        # (0,0,0,1) for the same reason.
        rot=(0.0, 0.0, 0.0, 1.0),
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
    # IMPORTANT: SAGE-10k rooms are NOT centered on origin — their walls span
    # absolute layout coordinates, e.g. x:(0..3.7), y:(0..3). pose_sample_range
    # is in env-local coords, so set it to match the actual wall bounds with
    # a ~0.5m safety margin inside the walls. Inspect the converted USD's
    # bbox before picking values; "symmetric ±N" only works if the room
    # happens to be centered at origin.
    pose_sample_range={"x": (0.5, 3.2), "y": (0.5, 2.5), "yaw": (-3.14, 3.14)},
    env_spacing=20,
)
```

For SAGE-10k rooms (typically 4–6m wall-to-wall), `env_spacing=20` provides plenty of clearance between parallel envs. Always compute `pose_sample_range` from the actual wall extent rather than assuming the room is centered.

Quick way to dump the room's world bbox (run inside the activated shell):
```python
from pxr import Usd, UsdGeom
stage = Usd.Stage.Open("<path/to/scene>.usd")
bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                         includedPurposes=[UsdGeom.Tokens.default_]
                        ).ComputeWorldBound(stage.GetPseudoRoot()
                        ).ComputeAlignedRange()
print(bbox.GetMin(), bbox.GetMax())
```

### File 2: `run.py`

Add to `EnvSceneAssetCfgMap` (around line 118–128):
```python
'<scene_var>': environments.<scene_var>,
```

---

## Train

### Step 1: Preview training  **(MANDATORY GATE — always do this first)**

This step is **not optional and cannot be skipped**, including under "no clarifying questions" mode — see [Mandatory visual-verification gates](#mandatory-visual-verification-gates).

Launch with `--num_envs 1` and **`--viz kit`** so the Isaac Sim Kit window opens and the user can verify the robot spawns correctly in the scene. Use `run_in_background: true`:

```bash
python run.py \
  -c configs/train_config.gin \
  --enable_cameras \
  --viz kit \
  -o <OUTPUT_DIR>_preview \
  -b ./assets/x_mobility.ckpt \
  -n <WANDB_PROJECT> \
  -r <RUN_NAME>_preview \
  --logger tensorboard \
  --video \
  --embodiment <EMBODIMENT> \
  --environment <SCENE_KEY> \
  --num_envs 1
```

Notes on flags:
- `--viz kit` opens the Kit viewer. Isaac Lab 3.0 deprecated `--headless`; headless is the new default, and you opt **in** to a visualizer with `--viz {kit,newton,rerun,viser}` (or `--viz none` to be explicit). Do **not** pass `--headless`.
- `-n/--wandb-project-name` is required by `run.py` even when `--logger tensorboard` — any string works.
- With `--viz kit`, expect ~30% slower throughput (renderer overhead) and ~9 GB extra GPU than headless.

Tell the user the preview is running with the GUI open on display `:1`. Ask them to verify:
1. The robot spawns in a valid location (not clipping through objects, inside the room).
2. The scene geometry and collisions look correct.
3. The robot can navigate without issues.

**Stop and wait for the user's explicit confirmation** before proceeding to Step 2. Do not auto-progress to full-scale training. If the user describes an issue, kill the preview, fix it (USD re-convert, `pose_sample_range` adjustment, embodiment swap, etc.), and re-launch the preview for another round.

### Step 2: Full-scale training

After preview confirmation, launch the full run with `run_in_background: true`:

```bash
python run.py \
  -c configs/train_config.gin \
  --enable_cameras \
  -o <OUTPUT_DIR> \
  -b ./assets/x_mobility.ckpt \
  -n <WANDB_PROJECT> \
  -r <RUN_NAME> \
  --logger tensorboard \
  --video \
  --embodiment <EMBODIMENT> \
  --environment <SCENE_KEY> \
  --num_envs <NUM_ENVS>
```

Headless is the default in Isaac Lab 3.0 — omit `--viz` entirely (or use `--viz none` to be explicit). Do **not** pass `--headless`; it's deprecated. If you want a viewer for a brief look, add `--viz kit` and drop `--num_envs` to 1 (renderer + many envs will OOM, especially on SAGE-10k scenes).

### OSMO cluster submission

The Python launcher in [osmo/run_osmo.py](../../../osmo/run_osmo.py) handles build+push+submit and is the recommended entry point:

```bash
export WANDB_API_KEY=<key>
export HF_TOKEN=<token>
export COMPASS_OSMO_REGISTRY=nvcr.io/<org>/<team>

python osmo/run_osmo.py train \
    --experiment-name <name> \
    --wandb-project <project>
```

The X-Mobility base checkpoint is now downloaded inside the workflow from `huggingface.co/nvidia/X-Mobility`, so no `--base-policy-ckpt` flag is needed. For direct `osmo workflow submit` invocation, see the [OSMO cloud submission](https://nvlabs.github.io/COMPASS/docs/osmo.html) handbook page. Workflow YAMLs live in [osmo/workflows/](../../../osmo/workflows/).

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

python run.py \
  -c configs/eval_config.gin \
  --enable_cameras \
  -o <OUTPUT_DIR> \
  -b ./assets/x_mobility.ckpt \
  -p $CKPT \
  -n <WANDB_PROJECT> \
  -r eval_<RUN_NAME> \
  --logger tensorboard \
  --video \
  --video_interval 1 \
  --embodiment <EMBODIMENT> \
  --environment <SCENE_KEY>
```

Headless by default (Isaac Lab 3.0). Add `--viz kit` to open the viewer for visual inspection of episodes. Do **not** pass `--headless` — deprecated.

---

## Visualize Scene in Isaac Sim

To inspect a USD scene before training (check collision meshes, layout, etc.):
```bash
python -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': False, 'width': 1280, 'height': 720})
import omni
omni.usd.get_context().open_stage('<USD_PATH>')
while app.is_running():
    app.update()
app.close()
" &
```
Use `dangerouslyDisableSandbox: true`. Suggest the user enable **Show > Physics > Colliders** to verify collision meshes.

**Path convention:** `<USD_PATH>` must be a path the **container** can see — pass either a path **relative to the repo root** (e.g. `./compass/rl_env/.../scene.usd`) or the container-side absolute path (`/workspace/COMPASS/...`). Do **not** pass the host absolute path (`/home/<user>/Projects/COMPASS/...`) — the container has the repo bind-mounted at `/workspace/COMPASS` and will fail to open a host path with "Failed to get crate info from file".

---

## Setup SAGE (local — advanced)

Most users do **not** need this. SAGE-10k search (above) covers the typical case without any SAGE install. Only use a local SAGE install when the user explicitly wants to generate fully custom scenes (novel layouts not in SAGE-10k).

For installation steps, dependencies, and trade-offs, see `references/setup-sage-local.md` — that file is loaded only when a local SAGE install is actually needed, to keep the main flow lean.

---

## Key File Locations

| File | Purpose |
|------|---------|
| `docker/run.sh` | Build / assets / up / down / exec / shell / status |
| `docker/activate` | Sourceable activate script (sets up python/pip shims) |
| `./assets/x_mobility.ckpt` | X-Mobility base policy (downloaded by `./docker/run.sh assets`) |
| `./assets/usd/` | Built-in scene USDs (downloaded by `./docker/run.sh assets`) |
| `run.py` | Main entry point + `EnvSceneAssetCfgMap` (around lines 118–128) |
| `compass/rl_env/exts/mobility_es/mobility_es/config/environments.py` | Scene definitions, `USD_PATHS`, `EnvSceneAssetCfg` class |
| `compass/rl_env/exts/mobility_es/mobility_es/usd/` | User-added scene USDs (e.g., SAGE-10k conversions) |
| `configs/train_config.gin` | Training parameters |
| `configs/eval_config.gin` | Evaluation parameters |
| `configs/shared.gin` | Shared config (embodiment, environment, steps) |
| `osmo/workflows/rl_es_train_workflow.yaml` | OSMO training workflow template |
| `osmo/run_osmo.py` | Python launcher for OSMO submission (train / eval / record / distill) |

## W&B Artifact Format
`<entity>/<project>/<artifact>:<version>` — e.g. `<your-entity>/<your-project>/x_mobility:v1`
