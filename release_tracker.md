# COMPASS 2.0 Release Tracker

> Working tracker for the next release. Distilled into `CHANGELOG.md` at ship time; do not duplicate that history here.

**Target version:** `2.0.0` (TBD — confirm vs `1.6.0`; major bump assumed because of Isaac Lab 3.0 break)
**Target date:** TBD
**Release manager:** @liuw
**Status legend:** ⬜ not started · 🟡 in progress · 🟢 done · 🔴 blocked · ⚪ deferred

## Summary

| # | Workstream | Priority | Status | Owner |
|---|-----------|----------|--------|-------|
| 1 | OSMO code migration (training runnable on OSMO) | P0 | ⬜ | TBD |
| 2 & 3 | Isaac Lab 2.1 → 3.0+ upgrade **+** NuRec official support (single branch) | P0 | ⬜ | @samc + @liuw |
| 4 | Agentic skills for automatic model training (also enables SAGE) | P1 | ⬜ | TBD |
| 5 | Auto OMap generation from USDs | P1 | ⬜ | TBD |
| 6 | GitHub Pages docs site (X-Mobility → COMPASS) | P1 | ⬜ | TBD |
| — | No-regression benchmark (gate) | P0 | ⬜ | TBD |
| — | CHANGELOG + version bump + tag | P0 | ⬜ | @liuw |

## Dependencies

```
#2&3 cherry-pick from samc/support_nurec_assets_isaaclab_3.0:
   PR-1 (Isaac Lab 3.0 migration) ──► PR-2 (NuRec assets + occupancy_map) ──┐
                                                                            │
OSMO (#1) ──────────────────────────────────────────────────────────────────┤
Agentic skills (#4) ────────────────────────────────────────────────────────┼──► Benchmark ──► Release
Auto-OMap (#5) ─────────────────────────────────────────────────────────────┘
Docs (#6) — parallel; finalized before tag
PR-FOLLOWUP (multi-cam recorder + video upload + debug images) — post-release, outside critical path
```

**SAGE note:** SAGE training is enabled by the agentic skills (#4). The only SAGE-specific gap is auto OMap generation (#5), which makes SAGE-driven training smoother. No standalone "SAGE integration" workstream.

---

## 1. OSMO code migration — P0

Bring OSMO training-launch code from internal `gitlab-master.nvidia.com/ml_nav/compass` to public `NVlabs/COMPASS`. Existing public repo already references OSMO dataset names (`record.py:147` `--dataset-name`).

- [ ] Inventory OSMO-specific files in internal repo (launch scripts, configs, manifests)
- [ ] Sanitize for public release (strip internal paths, secrets, unsupported clusters)
- [ ] Decide landing directory (`scripts/osmo/` or `osmo/`)
- [ ] Port docs / README section explaining OSMO submission flow
- [ ] Smoke test: launch one specialist training run end-to-end on OSMO
- [ ] Smoke test: launch one distillation run end-to-end on OSMO
- [ ] PR: <link>

**Internal source:** `gitlab-master.nvidia.com/ml_nav/compass` — OSMO directory TBD

## 2 & 3. Isaac Lab 3.0+ upgrade **+** NuRec official support — P0

The source branch `origin/samc/support_nurec_assets_isaaclab_3.0` (5 commits, ~1818 LOC) bundles unrelated work and **will not be merged whole**. We cherry-pick into two PRs for this release; the rest defers to a follow-up PR after the release tag.

### Branch decomposition (cherry-pick plan)

| Bucket | What it is | LOC | Lands in |
|--------|-----------|-----|----------|
| **A** | Isaac Lab 3.0 API migration: env_cfg signatures, ActionTerm import path, Warp interop (`wp.to_torch()`), PhysxCfg restructure, separate `write_root_pose_to_sim_index` / `write_root_velocity_to_sim_index`, `num_rerenders_on_reset`, **global quaternion convention flip in `environments.py` (wxyz → xyzw)**. Touches env_cfg.py, MDP terms, robots, scene_assets, *_env_cfg.py, pyproject.toml, Dockerfile.rl. | ~150 | **PR-1** |
| **B** | NuRec real2sim asset support: 41-line `environments.py` block (NuRec USD path + occupancy entry + `nova_carter_galileo_nurec` cfg), `configs/{train,eval}_config_real2sim.gin`, `run.py` registration. | ~95 | **PR-2** |
| **C** | New file `compass/utils/multi_camera_video_recorder.py` (689 LOC). General-purpose `gym.Wrapper` recording viewport + camera side-by-side. | 689 | Deferred |
| **D** | `residual_ppo_trainer.py` video upload dedup (`_find_video_files`, rewritten `_upload_video`, dedup sets). Pairs with C. | ~140 | Deferred |
| **E** | `occupancy_map.py` 429-line refactor: origin-convention support (top-left vs ROS bottom-left) **+** `precompute_valid_poses()` that buffers obstacles via scipy.ndimage and caches valid start/goal locations. **NOT USD-derived** — independent of #5. | 429 | **PR-2** (origin convention is required to parse NuRec's occupancy entry; precompute rides along, gated by opt-in flag) |
| **F** | Browser-compatible video: ffmpeg H.264 re-encode + `+faststart`, libx264 preset/crf tuning. Commits `ede049d` and `c34645f`. | ~80 | Deferred |
| **G** | Debug images / logger: `compass/utils/logger.py:74-91` adds `log_image()`; trainer `_save_debug_images()` + `_create_image_grid()` write PNG grids of obs every N iters. Commit `bafac90`. | ~190 | Deferred |
| **H** | Cleanups (run.py duplicated lines 244-246/247-250; `EnvSceneAssetCfgMap['nova_carter-galileo']` overrides existing `warehouse_multi_rack` mapping). | small | **PR-2** |

### PR-1 — Isaac Lab 3.0 API migration (Bucket A) — P0

Foundation. No new features; pure 2.1 → 3.0 compat. Off `main`.

- [ ] Extract A-only hunks from commit `3e6dcd9`
- [ ] Survey Isaac Lab 2.1 → 3.0 release notes / migration guide; confirm branch covers them
- [ ] Update version pins: README badges, `compass/rl_env/README.md`
- [ ] Update `docker/Dockerfile.distillation` base image if needed (`Dockerfile.rl` already on branch)
- [ ] **Reviewer spot-check**: quaternion convention flip across all rows of `environments.py` (wxyz → xyzw) — confirm no Y-up vs Z-up assumption breaks
- [ ] Re-validate USD assets load under 3.0 (`compass/rl_env/exts/mobility_es/mobility_es/usd/`)
- [ ] Smoke test: one short training per supported embodiment (H1, Carter, Spot, G1, Digit)
- [ ] PR: <link>

### PR-2 — NuRec real2sim assets + occupancy_map plumbing (Buckets B + E + H fixes) — P0

Off PR-1. NuRec asset support, the `occupancy_map.py` refactor it depends on, and cleanups in the same files.

- [ ] Cherry-pick `environments.py` NuRec block + `configs/train_config_real2sim.gin` + `configs/eval_config_real2sim.gin`
- [ ] Cherry-pick `compass/rl_env/exts/mobility_es/mobility_es/utils/occupancy_map.py` refactor (origin convention + `precompute_valid_poses`)
- [ ] **Cleanup**: collapse duplicated `run.py` lines 244-246 / 247-250
- [ ] **Cleanup**: register `'nova_carter-galileo'` as a *new* `EnvSceneAssetCfgMap` key alongside the existing `warehouse_multi_rack` (do not overwrite — fixes commit `86f9664`)
- [ ] Add NuRec asset section to `compass/rl_env/README.md`
- [ ] Add NuRec entry under "External assets that must be downloaded manually" in `README.md`
- [ ] Smoke test: training run with NuRec asset
- [ ] Smoke test: confirm a non-NuRec run on a `warehouse_multi_rack` scene still works (regression check)
- [ ] PR: <link>

### Deferred to follow-up PR (Buckets C + F + D + G)

Not on the release critical path. Ship after the release tag as one coherent observability PR.

- Multi-camera video recorder (`compass/utils/multi_camera_video_recorder.py`) — **C**
- Browser-compat video re-encoding (ffmpeg H.264 + faststart) — **F**
- Trainer video upload dedup (viewport + combined uploads) — **D**
- Debug image grid logging (`logger.log_image()` + `_save_debug_images()`) — **G**

Rationale: C + F record videos to disk; without D's wandb upload plumbing they're half-baked. G is the same observability stack. All four ship together post-release. Also revisit `bafac90`'s `debug_image_interval` default at that time so output dirs don't fill up.

**Branch:** `origin/samc/support_nurec_assets_isaaclab_3.0` (5 commits, ~1818 LOC)
**Predecessor branch:** `origin/samc/support_nurec_assets_isaaclab_2.3.1` (kept for reference; not merged)
**Owners:** @samc (NuRec) + @liuw (Isaac Lab integration review)

## 4. Agentic skills for automatic model training — P1

Migrate the agentic-skills tooling that automates training-loop execution from the internal repo. **This already enables SAGE training** — no separate SAGE workstream is needed; the skills cover it. (Scope to be sharpened — do these run as Claude Code skills, as a CLI orchestrator, or as OSMO-side jobs?)

- [ ] Identify scope: skill manifests vs. orchestrator code vs. both
- [ ] Decide landing directory (`.claude/skills/`? `agentic/`?)
- [ ] Sanitize for public release
- [ ] Hook into training entry points (`run.py`, `distillation_train.py`)
- [ ] Document trigger commands and expected behavior
- [ ] Demo: end-to-end automated specialist training using one skill
- [ ] Demo: SAGE-driven training using the migrated skills
- [ ] PR: <link>

**Internal source:** `gitlab-master.nvidia.com/ml_nav/compass` — agentic-skills directory TBD

## 5. Auto OMap generation from USDs — P1

Replace the manual occupancy-map authoring step with a USD-derived generator so SAGE-driven training (and any new scene) can ramp up without hand-tuned omaps. The current manual flow is documented in `compass/rl_env/README.md:94-96` and consumed by `compass/rl_env/exts/mobility_es/mobility_es/utils/occupancy_map.py`.

> **Relationship with #2&3:** Complementary, not overlapping. The NuRec branch's `occupancy_map.py` change is precomputation + origin convention (loads pre-baked YAML faster), not USD generation. #5 is genuinely separate work that produces the YAML automatically from a USD.

- [ ] Add CLI/script (e.g., `scripts/generate_omap_from_usd.py`) that produces the omap YAML directly from a USD scene
- [ ] Update `compass/rl_env/exts/mobility_es/mobility_es/utils/occupancy_map.py` to consume the auto-generated YAML (no breaking change to `OMAP_PATHS` if avoidable)
- [ ] Regenerate OMaps for all bundled scenes; commit YAML or document the one-line regeneration command
- [ ] Update `compass/rl_env/README.md` to point at the auto-generation flow as the default
- [ ] Measure training-throughput delta vs. manual OMaps; record in benchmark report
- [ ] PR: <link>

## 6. GitHub Pages docs site — P1

End-to-end workflow docs from X-Mobility install through COMPASS distillation and ROS2 deployment.

- [ ] Pick stack (MkDocs Material? Sphinx?) — confirm with team
- [ ] Decide branch strategy (existing `gh_page` is the academic project page; create `gh-pages` for docs OR host docs at `/docs/`)
- [ ] Outline: install → X-Mobility primer → specialists → distillation → export → ROS2 → OSMO + agentic skills (incl. SAGE) → API reference
- [ ] Migrate existing README + `compass/rl_env/README.md` + `ros2_deployment/README.md` content into the site (don't duplicate, link)
- [ ] Wire GitHub Actions to build & deploy on push to `main`
- [ ] PR: <link>

## Pre-release gates

- [ ] **No-regression benchmark** — run all supported embodiments × scenes on `main` post-merge, compare success rate / SPL / collision rate to 1.5.0 baselines, post results in this tracker
- [ ] All P0 items 🟢
- [ ] CHANGELOG.md `[2.0.0]` entry drafted (Added / Changed / Fixed / Removed)
- [ ] Version bump committed
- [ ] X-Mobility wheel updated if needed
- [ ] Pre-commit clean: `pre-commit run --all-files`
- [ ] Docs site builds without warnings
- [ ] Release notes drafted

## Release

- [ ] Tag `v2.0.0` and push
- [ ] GitHub Release published with notes
- [ ] HuggingFace assets (USDs, checkpoints) updated/uploaded if changed
- [ ] Internal announcement posted

## Out of scope / deferred

_Move items here with a one-line reason once cut. Empty at planning start._

## Notes

- DCO sign-off (`git commit -s`) is required for every commit — see `CONTRIBUTING.md`
- Internal repo: `gitlab-master.nvidia.com/ml_nav/compass`
- Public repo: `github.com/NVlabs/COMPASS`
- Existing `gh_page` branch hosts the academic project page (Bulma static site) — do not confuse with the new docs site
