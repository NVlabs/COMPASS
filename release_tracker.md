# COMPASS 2.0 Release Tracker

> Working tracker for the next release. Distilled into `CHANGELOG.md` at ship time; do not duplicate that history here.

**Target version:** `2.0.0` (TBD — confirm vs `1.6.0`; major bump assumed because of Isaac Lab 3.0 break)
**Target date:** TBD
**Release manager:** @liuw
**Status legend:** ⬜ not started · 🟡 in progress · 🟢 done · 🔴 blocked · ⚪ deferred

**Integration branch:** `liuw/training_time_improve` — 12 commits ahead of `main` (`30612e4`, 2025-10-16). Nothing merged yet; per-workstream PRs not split out. Tip commit: `9f478af`.

## Summary

| # | Workstream | Priority | Status | Owner |
|---|-----------|----------|--------|-------|
| 1 | OSMO code migration (training runnable on OSMO) | P0 | 🟡 | @liuw |
| 2 & 3 | Isaac Lab 2.1 → 3.0+ upgrade **+** NuRec official support (single branch) | P0 | 🟡 | @samc + @liuw |
| 4 | Agentic skills for automatic model training (also enables SAGE) | P1 | 🟡 | @liuw |
| 5 | Auto OMap generation from USDs | P1 | 🟡 | @liuw |
| 6 | GitHub Pages docs site (X-Mobility → COMPASS) | P1 | 🟡 | @liuw |
| 7 | Docker-as-venv dev environment (`docker/run.sh` + `docker/activate`) | P1 | 🟡 | @liuw |
| 8 | Pre-release leak audit + sanitization | P0 | 🟡 | @liuw |
| 9 | CI/CD setup + dependency pinning | P1 | 🟡 | @liuw |
| 10 | Agentic skills refresh + new onboarding skills | P1 | 🟡 | @liuw |
| 11 | Multi-GPU PPO training + perf instrumentation | TBD | 🟡 | @liuw |
| — | No-regression benchmark (gate) | P0 | ⬜ | TBD |
| — | CHANGELOG + version bump + tag | P0 | ⬜ | @liuw |

## Branch state (current snapshot)

All 12 commits below sit on `liuw/training_time_improve` (oldest → newest); nothing has merged to `main`. The per-workstream branches in `git branch -vv` (`liuw/osmo_migration`, `liuw/agentic_skills_migration`, …, `liuw/skills_enhancement`) are stacking checkpoints, not separate trees.

| Commit | Workstream | Note |
|--------|-----------|------|
| `22b25ef` | #2&3 PR-1 | Isaac Lab 3.0 API migration (Bucket A) |
| `1253f2a` | #1 | OSMO workflows + `run_osmo.py` (initial port) |
| `000c3be` | #4 | Migrate `compass` Claude Code skill from internal repo |
| `b20fb67` | #5 | USD-derived OMap generator + loader fallback |
| `725b79c` | #7 | Docker-as-venv (`docker/run.sh` + `docker/activate`) |
| `f4d89b8` | #6 | Docs site (academic landing + Sphinx handbook) |
| `d0275cc` | #8 | OSMO sanitization (HF asset sources) |
| `3cc5eaf` | #9 | Pre-commit CI workflow + requirements pinning |
| `e0b6f20` | #10 | `/compass` refresh + deploy/debug/newembodiment skills |
| `d605901` | #1 | Thread `--embodiment` / `--environment` through OSMO train |
| `424a96a` | gate | Add benchmark.py sanitization subtask under gate |
| `9f478af` | #11 | Multi-GPU PPO training + perf instrumentation |

**Still missing in code:** NuRec PR-2 (Buckets B+E+H from `samc/support_nurec_assets_isaaclab_3.0`), sanitized `benchmark.py` (only the tracker subtask exists), CHANGELOG `[2.0.0]` entry, version bump.

## Dependencies

```
#2&3 cherry-pick from samc/support_nurec_assets_isaaclab_3.0:
   PR-1 (Isaac Lab 3.0 migration) ──► PR-2 (NuRec assets + occupancy_map) ──┐
                                                                            │
OSMO (#1) ──────────────────────────────────────────────────────────────────┤
Agentic skills (#4) ────────────────────────────────────────────────────────┼──► Benchmark ──► Release
Auto-OMap (#5) ─────────────────────────────────────────────────────────────┘
Docs (#6), Dev environment (#7) — parallel; finalized before tag
Agentic skills refresh (#10) ── depends on Dev environment (#7) and Sanitization (#8)
PR-FOLLOWUP (multi-cam recorder + video upload + debug images) — post-release, outside critical path
```

**SAGE note:** SAGE training is enabled by the agentic skills (#4). The only SAGE-specific gap is auto OMap generation (#5), which makes SAGE-driven training smoother. No standalone "SAGE integration" workstream.

---

## 1. OSMO code migration — P0

Bring OSMO training-launch code from internal `gitlab-master.nvidia.com/ml_nav/compass` to public `NVlabs/COMPASS`. Existing public repo already references OSMO dataset names (`record.py:147` `--dataset-name`).

- [x] Inventory OSMO-specific files in internal repo (launch scripts, configs, manifests) — landed in `1253f2a`
- [x] Sanitize for public release (strip internal paths, secrets, unsupported clusters) — `1253f2a` (initial) + `d0275cc` (HF asset sources, no internal defaults)
- [x] Decide landing directory: `osmo/`
- [x] Port docs / README section explaining OSMO submission flow — `osmo/README.md` + root README section
- [x] Thread `--embodiment` / `--environment` through train workflow (so multi-embodiment sweeps don't require YAML hand-edits) — `d605901`
- [ ] Smoke test: launch one specialist training run end-to-end on OSMO
- [ ] Smoke test: launch one distillation run end-to-end on OSMO
- [ ] PR: <link>

**Internal source:** `gitlab-master.nvidia.com/ml_nav/compass`
**Commits on integration branch:** `1253f2a`, `d0275cc`, `d605901`

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

- [x] Extract A-only hunks from commit `3e6dcd9` — landed in `22b25ef` ("Migrate mobility_es extension to Isaac Lab 3.0 API")
- [x] Update version pins: README badge + `compass/rl_env/README.md` install instructions bumped to `v3.0.0-beta1` — part of `22b25ef`
- [ ] Survey Isaac Lab 2.1 → 3.0 release notes / migration guide; confirm branch covers them
- [ ] Update `docs/handbook/extending.md` bare-metal install pin
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
- [ ] Add NuRec asset section to `docs/handbook/extending.md` (or a new handbook page if the section grows)
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

Land the Claude Code skill that automates training-loop execution. Migration phase done — `.claude/skills/compass/SKILL.md` plus the two helper scripts (`scripts/sage10k_search.py`, `scripts/sage10k_to_usd.py`) are in the public repo and document the SAGE-10k → USD → register → train pipeline.

- [x] Identify scope: Claude Code skill (markdown + YAML frontmatter + helper scripts).
- [x] Decide landing directory: `.claude/skills/compass/`.
- [x] Sanitize for public release.
- [x] Hook into training entry points (`run.py`, evaluation, OSMO submission).
- [x] Document trigger commands and expected behavior (`docs/handbook/agentic.md`).
- [ ] Demo: end-to-end automated specialist training using one skill — covered by **#10** verification.
- [ ] Demo: SAGE-driven training using the skill — covered by **#10** verification.
- [ ] PR: <link>

**Continued in:** #10 (refresh for docker-as-venv + new specialty skills). The two demo boxes flip 🟢 once #10's verification step runs them.

## 5. Auto OMap generation from USDs — P1

Replace the manual occupancy-map authoring step with a USD-derived generator so SAGE-driven training (and any new scene) can ramp up without hand-tuned omaps. The auto-gen flow is documented at https://nvlabs.github.io/COMPASS/docs/omap.html and consumed by `compass/rl_env/exts/mobility_es/mobility_es/utils/occupancy_map.py`.

> **Relationship with #2&3:** Complementary, not overlapping. The NuRec branch's `occupancy_map.py` change is precomputation + origin convention (loads pre-baked YAML faster), not USD generation. #5 is genuinely separate work that produces the YAML automatically from a USD.

- [x] Add CLI/script `scripts/generate_omap_from_usd.py` that produces the omap PNG+YAML directly from a USD scene (wraps `isaacsim.asset.gen.omap.bindings._omap.Generator`)
- [x] Update `compass/rl_env/exts/mobility_es/mobility_es/utils/occupancy_map.py` so a scene without an `OMAP_PATHS` entry auto-discovers `<usd_dir>/omap/occupancy_map.yaml` (no breaking change to existing entries)
- [x] Update `compass/rl_env/README.md` and `.claude/skills/compass/SKILL.md` to point at the auto-generation flow
- [x] Verify generation + collision-free sampling on representative USDs:
  - `office.usd` — ✅ 200/200 free samples in unoccupied regions
  - `combined_simple_warehouse/combined.usd` (default training scene) — ✅ 253/300 free samples; visually free dots avoid obstacles
  - `sample_small_footprint_one_rack_obst_sdg.usd` — ✅ 145/300 free samples; same
- [ ] Regenerate OMaps for all bundled scenes (optional follow-up; current loader auto-discovers when no `OMAP_PATHS` entry)
- [ ] Measure training-throughput delta vs. manual OMaps; record in benchmark report
- [ ] PR: <link>

**Branch:** `liuw/auto_omap_from_usd` (off `liuw/agentic_skills_migration`)
**Key fix:** wait on `omni.usd.get_context().get_stage_loading_status()` until `to_load == 0` before invoking `generate2d()` — without this, kit crashes on USDs with external references (e.g. `combined_simple_warehouse` referencing `galileo_lab.usd`) when the omap generator queries fabric for prims whose references are still resolving. Mirrors the pattern Isaac Sim's own `isaacsim.asset.gen.omap` tests use.

## 6. GitHub Pages docs site — P1

End-to-end docs site auto-deployed from `main/docs/` via GitHub Actions. Academic landing stays at `nvlabs.github.io/COMPASS/`; new **Sphinx handbook with the NVIDIA theme** (matching `agentic_model_training/docs/` and the rest of NVIDIA OSS) serves at `nvlabs.github.io/COMPASS/docs/`. Replaces the hand-served `gh_page` branch.

- [x] Stack decided: **Sphinx 7.x + nvidia-sphinx-theme + myst-parser** (markdown survives; matches NVIDIA house style)
- [x] Source location: `main/docs/` (so doc edits go through normal PRs); `gh_page` left as a frozen archive
- [x] URL layout: academic at `/`, handbook at `/docs/`
- [x] Migrate `gh_page` → `docs/project_page/` (264 files; mp4/png LFS-tracked, rest as Git blobs) — landed in commit `305c3a1`
- [x] Add `docs/handbook/{conf.py, Makefile, requirements.txt, _static/, docs/}` with `{toctree}` nav (Installation / Workflows / Deployment / Reference)
- [x] Transclude existing READMEs (Docker, OSMO, ROS2, mobility_es, CONTRIBUTING) via MyST `` ```{include} `` directives; no copy-paste of content
- [x] Add Documentation CTA on academic landing (`docs/project_page/index.html` → `./docs/`)
- [x] Wire `.github/workflows/docs.yml` (`make html` / `sphinx-build -W`, copy academic landing to root, deploy via `actions/deploy-pages@v4`)
- [ ] Local build verification: `make html` runs clean; all 16 handbook pages render; both `/` and `/docs/` serve correctly from `_site/`
- [ ] Repo settings: **Settings → Pages → Source = "GitHub Actions"** (one-time owner action; documented in PR description)
- [ ] Push to main + watch first deploy succeed
- [ ] PR: <link>

**Branch:** `liuw/docs_site` (off `liuw/dev_environment`, latest in the stack).
**Status:** files written; local Sphinx build pending; Pages source switch + push pending.

## 7. Docker-as-venv dev environment — P1

Quality-of-life: cut first-time UX from "30–60 min, 6 manual steps" to **"3 commands, ~3 min"**, and make the steady-state dev loop feel like a Python venv (host-side editor, host-side shell, but every `python`/`pip`/`tensorboard` invocation transparently routed through the container via `source ./docker/activate`). Reuses `docker/Dockerfile.rl` (single image, +5 lines). Detailed plan at [`dev_env_plan.md`](dev_env_plan.md).

- [x] Add `docker/run.sh` (subcommands: build / assets / up / down / exec / shell / status)
- [x] Add `docker/activate` (sourceable — venv-like; shim PATH for python/pip/tensorboard/etc., CWD translation, deactivate)
- [x] Add `docker/prepare_assets.sh` (USDs + X-Mobility ckpt → `./assets/`, cache-aware)
- [x] Add `docker/README.md` (subcommand reference, multi-checkout / multi-GPU notes, troubleshooting)
- [x] Modify `docker/Dockerfile.rl`: install COMPASS at `/workspace/COMPASS` (so `/workspace/isaaclab` survives the bind-mount); add a `python`/`python3` wrapper that exec's Isaac Sim's bundled `python.sh`
- [x] Update `.dockerignore` (`./assets/`, `./.cache/`, `./.git/`) and `.gitignore` (`/assets/`, `/.cache/`)
- [x] Update root `README.md` to lead with the Docker quick-start; bare-metal moved under "Manual install"
- [ ] Verify: `./docker/run.sh build && ./docker/run.sh assets && source ./docker/activate && python run.py … --num_envs 1 --headless` reaches first PPO iteration
- [ ] Verify: `git commit -s` from an activated shell triggers pre-commit through the shim and signs cleanly
- [ ] Verify: `osmo/run_osmo.py train` still works against the same image (regression)
- [ ] PR: <link>

**Branch:** `liuw/dev_environment` (off `liuw/auto_omap_from_usd`, latest in the stack).
**Status:** files written; verification + commit pending.

## 8. Pre-release leak audit + sanitization — P0

Scrub all internal-only references before tagging 2.0. Inventory in the
planning round; the meaningful work is in the OSMO entry script.

- [x] OSMO workflows: replace `groot_mobility_rl_es_usds` dataset input with HF download `nvidia/COMPASS / compass_usds.zip` (3 YAMLs)
- [x] OSMO workflows: replace `wandb artifact get …base_policy_ckpt…` with HF download `nvidia/X-Mobility / x_mobility-nav2-semantic_action_path.ckpt` (3 YAMLs)
- [x] `osmo/run_osmo.py`: drop `nvidia-isaac` wandb-project defaults (`compass_rl_enhance`, `afm_train`); make `--wandb-project` required; drop `--base-policy-ckpt` flag (workflow now hardcodes the HF source)
- [x] Update `docs/handbook/osmo.md` to reflect HF-sourced assets + required `--wandb-project`
- [ ] `ros2_deployment/compass_navigator/setup.py`: review maintainer attribution (flag for @liuw; team alias preferred)
- [ ] OSMO smoke test: resubmit `compass_rl_es_g1_official` with the rebuilt image; confirm HF download steps succeed and training reaches first PPO iter
- [ ] Repo-wide grep gate: `grep -rnE "groot_mobility_rl_es_usds|nvidia-isaac/|afm_train|groot_mobility_rl_enhance|afm_rl_enhance" --include='*.py' --include='*.sh' --include='*.yaml' --include='*.gin' --include='*.html' .` returns no live-source hits
- [ ] Distill `release_tracker.md` + `dev_env_plan.md` into `CHANGELOG.md` and remove from the repo at tag time (existing gate row)
- [ ] PR: <link>

**Branch:** `liuw/sanitize_for_public` (off `liuw/docs_site`, latest in the stack).

## 9. CI/CD setup + dependency pinning — P1

Bring the public repo up to a "first-line" CI posture before tagging:

- [x] Add `.github/workflows/pre-commit.yml` (yapf / pylint / nbstripout / clang-format / large-files / trailing-whitespace / EOF / requirements-txt-fixer)
- [x] Pin `requirements.txt` to versions verified inside `compass-rl:latest` (the image used for the just-passed OSMO smoke). 17 unpinned → 17 pinned; `diffusers==0.29.2` was already pinned.
- [ ] First run of the workflow on PR-9: confirm pre-commit passes `--all-files` against the current branch. If legacy violations surface, fix in a follow-up commit on the same branch.
- [ ] Decide later whether to add a Sphinx-handbook `linkcheck` job (out of scope for now; tracked under #6 docs).
- [ ] PR: <link>

**Branch:** `liuw/ci_setup` (off `liuw/sanitize_for_public`, latest in the stack).

## 10. Agentic skills refresh + new onboarding skills — P1

Refresh the existing `compass` skill for the new docker-as-venv flow (#7) and add three onboarding-focused specialty skills following the **hybrid front-door** pattern (`compass` is the umbrella; `compass-deploy` / `compass-debug` / `compass-newembodiment` are narrow siblings the auto-router picks unambiguously).

- [x] Update `.claude/skills/compass/SKILL.md` for docker-as-venv (drop conda wrappers; new prereq check; new setup recipe; OSMO example aligned with #8). Add "Specialty skills" front-door section. Soften MUST-style rules with explanatory why.
- [x] Pre-emptive progressive-disclosure split: extract "Setup SAGE Local (Advanced)" to `.claude/skills/compass/references/setup-sage-local.md`.
- [x] Add `.claude/skills/compass-deploy/SKILL.md` (ckpt → ONNX → TRT → ROS2 launch scaffold).
- [x] Add `.claude/skills/compass-debug/SKILL.md` + bundled `scripts/compass_status.sh` (8 diagnostic checks; markdown table; --deep + --ckpt flags).
- [x] Add `.claude/skills/compass-newembodiment/SKILL.md` (interactive robot onboarding; parses pre-supplied user input; AskUserQuestion only for missing fields; diff-then-confirm before writing).
- [x] Update `docs/handbook/agentic.md` with four-skill index ("Pick the right skill" matrix at the top).
- [ ] Live invocation test in a fresh Claude Code session for each skill; confirm prereq check passes inside an activated shell.
- [ ] Run the two #4 demo flows (specialist training + SAGE-driven training) using `/compass`; flip #4 demo boxes 🟢 on success.
- [ ] (Post-PR follow-up) Run `/skill-creator`'s description-optimization loop against all four skills; apply the highest-scoring descriptions on a held-out trigger eval split.
- [ ] PR: <link>

**Branch:** `liuw/skills_enhancement` (off `liuw/ci_setup`, latest in the stack).

## 11. Multi-GPU PPO training + perf instrumentation — TBD priority

8-GPU distributed residual-RL training (torchrun + manual all-reduce for gradients / KL / metrics), per-stage timing instrumentation, supporting OSMO 8-GPU workflow, and a perf-analysis report.

Scope (commit `9f478af`):
- `run.py`, `compass/residual_rl/ppo.py`, `compass/residual_rl/residual_ppo_trainer.py`: `--distributed`, `dist.init_process_group(nccl)`, per-rank device pinning, rank-0 gating on logger / ckpt / video / episode-log, weighted per-rank log aggregation, manual gradient all-reduce (AVG) BEFORE clip, KL all-reduce before LR adaptation, metric all-reduce, diagnostics dict for `ppo/learning_rate` / `kl_mean` / `entropy` / `action_std_mean`.
- `osmo/workflows/rl_es_train_8gpu_workflow.yaml`: 8-GPU / 80-CPU / 800-GiB on ovx-l40; train via `torch.distributed.run --nproc_per_node=8 run.py --distributed --num_envs 32` (256 total envs); single-process eval.
- `osmo/run_osmo.py`: `--num-gpus` flag (choices 2, 8) routes to matching workflow YAML.

- [x] Multi-GPU code path landed (`9f478af`)
- [x] OSMO 8-GPU workflow YAML
- [ ] **Release-scope decision:** ship in 2.0 vs defer? (impacts CHANGELOG, benchmark matrix, validation surface)
- [ ] Numerical-equivalence check vs single-GPU baseline (loss curve / success-rate parity within seed noise)
- [ ] 8-GPU OSMO smoke run reaches first PPO iter and produces ckpts
- [ ] 2-GPU path smoke (the other `--num-gpus` choice)
- [ ] Document `--distributed` in handbook / OSMO docs
- [ ] PR: <link>

**Branch:** currently sits at the tip of `liuw/training_time_improve` (this commit is the branch name's eponym). Needs to be carved into its own PR or rolled into the release stack.

## Pre-release gates

- [ ] **No-regression benchmark** — run all supported embodiments × scenes on `main` post-merge, compare success rate / SPL / collision rate to 1.5.0 baselines, post results in this tracker
  - [x] Sanitize and land internal `benchmark.py` (113 lines; hardcodes `nvcr.io/nvstaging/isaac-amr/groot_mobility_rl_enhance` registry + `afm_rl_enhance` defaults). Landed as `osmo/run_benchmark.py` on `liuw/benchmark_port`: Apache-2.0 header, `--registry-prefix` (with `$COMPASS_OSMO_REGISTRY` fallback), `--wandb-project-name` now `required=True`, workflow path resolved relative to `osmo/`. Reuses existing `osmo/workflows/rl_es_eval_workflow.yaml`.
  - [ ] Define the regression matrix: which embodiments × scenes × seeds, with what success/SPL/collision thresholds vs the v1.5.0 baseline.
  - [ ] Capture baseline numbers from v1.5.0 (or last known-good run) into this file before kicking off the new run.
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
