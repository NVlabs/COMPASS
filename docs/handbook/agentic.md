# Agentic skills

COMPASS ships three [Claude Code](https://claude.com/claude-code) skills that
take you from a natural-language prompt to a runnable, deployed policy.
Each skill is a focused recipe the agent follows on your behalf.

## Pick the right skill

| If you want to… | Use this skill | Lives at |
|---|---|---|
| Train a policy, search SAGE-10k scenes, register a new scene, evaluate a checkpoint, submit to OSMO, deploy a checkpoint | [`/compass`](https://github.com/NVlabs/COMPASS/blob/main/.claude/skills/compass/SKILL.md) | umbrella |
| Diagnose vague COMPASS errors — "training won't start", "what's wrong" | [`/compass-doctor`](https://github.com/NVlabs/COMPASS/blob/main/.claude/skills/compass-doctor/SKILL.md) | specialty |
| Onboard a new robot platform (cfg files, EmbodimentEnvCfgMap registration) | [`/compass-newembodiment`](https://github.com/NVlabs/COMPASS/blob/main/.claude/skills/compass-newembodiment/SKILL.md) | specialty |

`/compass` is the default front door — if you're not sure, start there
and it'll point you at the matching specialty when your intent is
debugging or onboarding a new robot.

## How a skill gets invoked

Inside a Claude Code session at the COMPASS repo root:

```
/skill compass <natural-language task>
/skill compass-doctor
/skill compass-newembodiment add Galileo robot, USD at ./assets/usd/galileo.usd, like Spot
```

Or just describe the task in plain English; Claude Code auto-routes when
the intent matches the skill's description. The descriptions are written
to be non-overlapping, so a diagnostic-flavored question routes to
`/compass-doctor` even without the slash command.

## What each skill does

### `/compass` — umbrella front door

End-to-end training pipeline. Covers:

- **Scene search** against the SAGE-10k catalogue (`scripts/sage10k_search.py`).
- **SAGE → USD conversion** via `scripts/sage10k_to_usd.py`.
- **Scene registration** in
  [`compass/rl_env/exts/mobility_es/mobility_es/config/environments.py`](https://github.com/NVlabs/COMPASS/blob/main/compass/rl_env/exts/mobility_es/mobility_es/config/environments.py)
  and `EnvSceneAssetCfgMap` in `run.py`.
- **OMap generation** for the new scene (see [Auto OMap from USDs](omap.md)).
- **Training launch** via `run.py` — preview run with one env first,
  then full-scale.
- **Evaluation** via `run.py` with `eval_config.gin`.
- **OSMO cloud submission** via [`osmo/run_osmo.py`](osmo.md).

Auto-OMap is the SAGE-driven training smoothness win: SAGE scenes don't
ship with hand-authored occupancy maps, and the [auto OMap from
USDs](omap.md) generator closes that gap so every new scene the agent
registers comes with a ready-to-use OMap, no manual UI step.

### `/compass-doctor` — one-shot diagnostics

Read-only health check. Bundles a `scripts/compass_status.sh` helper
that runs eight diagnostic checks in parallel and prints a markdown
table:

| Check | What it verifies |
|---|---|
| Container | `compass-rl` is up |
| Activated shell | `./docker/activate` shim is on PATH |
| GPU | `nvidia-smi` responds |
| Base ckpt | `./assets/x_mobility.ckpt` exists |
| USDs | `./assets/usd/` has scene content |
| Recent log | a training run has produced log output |
| (--deep) Isaac Sim init | headless `SimulationApp` starts and closes |
| (--ckpt PATH) Ckpt load | `torch.load(...)` succeeds on the file |

The skill reports the table, identifies the root cause, and points you
at the specialty skill that owns the fix — but does **not** auto-fix the
issue. Understanding the root cause is more useful than a silent
recovery.

### `/compass-newembodiment` — onboard a new robot

Interactive walkthrough of adding a new robot platform. Parses what you
already supplied (robot name, USD path, base-class hint) and only asks
about what's missing.

1. Ask the four onboarding questions (skipping any you already answered).
2. Read the matching reference (`carter` for wheeled, `spot` for
   quadrupeds, `h1` for humanoids).
3. Show a diff of the proposed changes — new `ArticulationCfg` block in
   `robots.py`, new `<robot>_env_cfg.py`, edits to `run.py`.
4. Apply after your confirmation.
5. Smoke-test with `--num_envs 1` and the GUI on so you can watch the
   robot spawn.

The handbook page [Adding a new embodiment or scene](extending.md) is
the canonical written guide; the skill is the operational sibling.

## Reading the skill source

Skills are plain markdown with YAML frontmatter under `.claude/skills/`.
Each `SKILL.md` documents every sub-flow, the prompts the agent uses,
and the verification steps it runs. Treat it as the spec for what the
agent will do on your behalf — useful both for understanding what
happens and for editing the skill if you want to fork the workflow for
your own purposes.
