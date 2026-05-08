# Agentic skills

COMPASS ships a Claude Code skill that orchestrates the end-to-end
training pipeline from natural-language prompts: scene search / download
(SAGE-10k), USD conversion, scene registration, training launch, and
optional OSMO submission.

The skill itself lives at
[`.claude/skills/compass/SKILL.md`](https://github.com/NVlabs/COMPASS/blob/main/.claude/skills/compass/SKILL.md).
Helper scripts live at
[`.claude/skills/compass/scripts/`](https://github.com/NVlabs/COMPASS/tree/main/.claude/skills/compass/scripts).

## How it gets invoked

Inside a [Claude Code](https://claude.com/claude-code) session at the COMPASS
repo root:

```
/skill compass <natural-language task>
```

…or just describe the task in plain English; Claude Code auto-routes when the
intent matches the skill's description.

## Capabilities

The skill encapsulates these concrete sub-flows; each is a pre-recorded
recipe that the agent follows:

- **Scene search** against the SAGE-10k catalogue.
- **SAGE → USD conversion** via `scripts/sage10k_to_usd.py`.
- **Scene registration** in
  [`compass/rl_env/exts/mobility_es/mobility_es/config/environments.py`](https://github.com/NVlabs/COMPASS/blob/main/compass/rl_env/exts/mobility_es/mobility_es/config/environments.py)
  and `EnvSceneAssetCfgMap` in `run.py`.
- **OMap generation** for the new scene (see [Auto OMap from USDs](omap.md)).
- **Training launch** via `run.py` (or [`osmo/run_osmo.py`](osmo.md) for cloud).

## Auto OMap is the SAGE smoothness win

SAGE-driven scenes don't ship with hand-authored occupancy maps. The
[auto OMap from USDs](omap.md) generator closes that gap so every new scene
the agent registers comes with a ready-to-use OMap for collision-free pose
sampling, with no manual UI step.

## Reading the skill source

Skills are plain markdown with YAML frontmatter. The
[full `SKILL.md`](https://github.com/NVlabs/COMPASS/blob/main/.claude/skills/compass/SKILL.md)
documents every sub-flow, the prompts the agent uses, and the verification
steps it runs. Treat it as the spec for what the agent will do on your
behalf.
