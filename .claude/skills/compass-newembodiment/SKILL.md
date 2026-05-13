---
name: compass-newembodiment
description: >
  Onboard a new robot platform to COMPASS: generate robot ArticulationCfg
  + per-embodiment env_cfg, register in EmbodimentEnvCfgMap, smoke-test
  the spawn with --num_envs 1. Use whenever the user wants to add a new
  robot, register an embodiment, or extend COMPASS with their platform.
  NOT for switching between built-in embodiments (use --embodiment) or
  for adding a new scene (the `compass` skill covers SAGE → register).
allowed-tools:
  - Bash
  - Read
  - Edit
  - Write
  - Grep
  - Glob
  - AskUserQuestion
---

You walk a user through onboarding a new robot platform to COMPASS — generating the right config files, registering the embodiment, and verifying the robot spawns correctly. The handbook's [`extending.md`](../../../docs/handbook/extending.md) is the canonical written guide; this skill is the operational sibling that does the work interactively.

This skill assumes the docker-as-venv environment from the `compass` skill — `source ./docker/activate` already done, container up. If the user hasn't set that up, recommend `/compass` first.

## When NOT to use this skill

- The user wants to switch between built-in embodiments (carter / h1 / spot / g1 / digit) → just use `--embodiment <name>` on `run.py`. No code change needed.
- The user is adding a new **scene** (USD), not a new robot → that's `/compass` (Register Scene).
- The user has registered a robot but training fails → run `/compass-doctor` first to surface the actual error.

---

## Workflow

### Step 1: Parse what the user already supplied

A user prompt like "add a new robot called Galileo at `/workspace/galileo.usd`, similar to Spot" already contains three of the four fields the skill needs. Extract them first; only ask about what's missing or genuinely ambiguous. Don't re-ask for things the user already said — that's the dialogue-bot trap.

The four fields:
- **Robot name** (lowercase identifier like `galileo`; used in CLI flags and config keys).
- **USD path** (absolute or repo-relative; the asset that will be spawned).
- **Base-class to inherit from** (which existing pattern to mirror). See "Base-class choice" below.
- **(Optional) display name** for docs (e.g., "Galileo").

### Step 2: Ask only for what's missing (`AskUserQuestion`)

If any of the three required fields are missing or ambiguous, ask in one batch. Sample question structure:

| Field | Question | Options |
|---|---|---|
| Robot name | "What's the robot's identifier?" | free text (default: lowercase the user's name) |
| USD path | "Where's the robot's USD file?" | free text (suggest checking `./assets/usd/` first) |
| Base-class | "Which existing robot's pattern is the best fit?" | wheeled (carter), quadruped (spot), humanoid (h1, g1, digit) |

For the base-class question, surface the trade-off in the option descriptions:
- **wheeled**: single base, velocity over base; simplest to onboard. Mirror `carter`.
- **quadruped**: articulated legs, ABS-style controller. Mirror `spot`.
- **humanoid**: full body; more complex action space. Mirror `h1` (or `g1`/`digit` for smaller form factors).

### Step 3: Read the existing-pattern files

Before generating anything, read the two files that match the user's base-class choice:

| Base class | Robot block | Env cfg file |
|---|---|---|
| wheeled | `carter` block in `compass/rl_env/exts/mobility_es/mobility_es/config/robots.py` | `compass/rl_env/exts/mobility_es/mobility_es/config/carter_env_cfg.py` |
| quadruped | `spot` block in same `robots.py` | `compass/rl_env/exts/mobility_es/mobility_es/config/spot_env_cfg.py` |
| humanoid | `h1` block in same `robots.py` | `compass/rl_env/exts/mobility_es/mobility_es/config/h1_env_cfg.py` |

Read both top-to-bottom. Don't skim — joint names, action term wiring, and reward shaping all come from this file. Confirming the reference exists before writing keeps you from generating content based on outdated assumptions.

### Step 4: Show the diff before writing

Generate a preview of what will be created or modified:

1. **New ArticulationCfg block** to be appended to `compass/rl_env/exts/mobility_es/mobility_es/config/robots.py`. Mirror the matching reference, swap robot name + USD path + joint config.
2. **New file**: `compass/rl_env/exts/mobility_es/mobility_es/config/<robot_name>_env_cfg.py`. Mirror the matching reference; swap class name (e.g., `<RobotName>GoalReachingEnvCfg`), `scene.robot` to point at the new ArticulationCfg, action term to whatever joint controller fits.
3. **Edit `run.py`**:
   - Add import: `from mobility_es.config.<robot_name>_env_cfg import <RobotName>GoalReachingEnvCfg`.
   - Add row to `EmbodimentEnvCfgMap`: `'<robot_name>': <RobotName>GoalReachingEnvCfg,` (around line 116–122).
4. **Optional gin update**: if the user wants `<robot_name>` to be the default embodiment, edit `configs/shared.gin`'s `embodiment` field. By default, leave existing default in place and let the user pass `--embodiment <robot_name>` on the CLI.

Show the user each preview block (or unified diff). Ask for confirmation before writing.

Anti-pattern guard: never silently write multi-file scaffolding. Robot onboarding is high-blast-radius — wrong joint counts or an inverted action sign can cause the robot to flail into the floor in simulation. The diff-then-confirm gate is non-negotiable.

### Step 5: Apply the changes

After confirmation, write the files and edit `run.py`. Use the `Edit` tool for `run.py` (one Edit per insertion: import + map row). Use `Write` for the new env_cfg.py. For `robots.py`, use `Edit` to append the new ArticulationCfg block at the end (read the file first to find the right insertion point — the last existing config).

### Step 6: Smoke test

Verify the robot spawns by running training with `--num_envs 1` and **without** `--headless` so the user can see the robot appear:

```bash
python run.py \
  -c configs/train_config.gin \
  --enable_cameras \
  -o /tmp/<robot_name>_smoke \
  -b ./assets/x_mobility.ckpt \
  --logger tensorboard \
  --embodiment <robot_name> \
  --environment warehouse_single_rack \
  --num_envs 1
```

Use `run_in_background: true`. Tell the user to watch the GUI:
1. Robot spawns at the configured initial pose (typically (0, 0, ~1.0)).
2. Robot doesn't immediately fall through the floor (z-init too low) or shoot up (z-init too high).
3. First PPO iteration completes without joint-controller errors in the log.

If the robot misbehaves, route to `/compass-doctor` for a deeper diagnostic. Common first-onboarding issues:
- Z-init too low → robot clips through floor → bump `init_state.pos[2]`.
- Joint names mismatch → ArticulationCfg fails to load → recheck against the USD's joint hierarchy.
- Action term sign flipped → robot drives backwards → invert sign in the action term.

### Step 7: Document the new embodiment (optional)

Once smoke test passes, suggest the user add their robot to `docs/handbook/extending.md` (under "Built-in embodiments" or as a new subsection). This is optional — many users keep their custom embodiment private. Don't push the doc change if the user is in a hurry; just flag it for later.

---

## Reference: file map

| File | What changes |
|------|---|
| `compass/rl_env/exts/mobility_es/mobility_es/config/robots.py` | Append new `ArticulationCfg` block |
| `compass/rl_env/exts/mobility_es/mobility_es/config/<robot_name>_env_cfg.py` | New file (mirror matching base-class reference) |
| `run.py` | Add import; add `EmbodimentEnvCfgMap` row (~L116–122) |
| `configs/shared.gin` | Optional default override; usually leave alone |
| `docs/handbook/extending.md` | Optional doc update (suggest, don't enforce) |

## Reference: base-class examples

- **Wheeled (carter)**: see `carter` block at `robots.py:23` and `carter_env_cfg.py`.
- **Quadruped (spot)**: see `spot` block at `robots.py:275` and `spot_env_cfg.py`.
- **Humanoid (h1)**: see `h1` block at `robots.py:72` and `h1_env_cfg.py`.

For canonical written guidance (joint config, controller selection, reward shaping), the handbook's [`extending.md`](../../../docs/handbook/extending.md) covers depth this skill won't replicate.
