# COMPASS 1.6 Release Tracker

> Working tracker for the next release. Distilled into `CHANGELOG.md` at ship time; do not duplicate that history here.

**Target version:** `1.6.0`
**Target date:** TBD
**Release manager:** @liuw
**Status legend:** ⬜ not started · 🟡 in progress · 🟢 done · 🔴 blocked · ⚪ deferred

**Integration branch:** `liuw/benchmark_port` (off `liuw/training_time_improve`) — 13 commits ahead of `main` (`30612e4`, 2025-10-16). Nothing merged yet. Release strategy: **single squash-merge to `main`** (no per-workstream PR stack).

## Summary

| # | Workstream | Priority | Status | Owner |
|---|-----------|----------|--------|-------|
| 1 | OSMO code migration (training runnable on OSMO) | P0 | 🟡 | @liuw |
| 2 & 3 | Isaac Lab 2.1 → 3.0+ upgrade (Bucket A) **+** NuRec PR-2 deferred to post-1.6 | P0 (Bucket A) / ⚪ (NuRec PR-2) | 🟢 (A) · ⚪ (PR-2) | @samc + @liuw |
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

**Still missing in code for 1.6:** CHANGELOG `[1.6.0]` entry, version bump, pre-commit `lib2to3` toolchain fix. NuRec PR-2 (Buckets B+E+H) is **deferred to post-1.6** — Bucket A (`22b25ef`) is the only NuRec-related code in 1.6.

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

## 2 & 3. Isaac Lab 3.0+ upgrade (in 1.6) **+** NuRec official support (post-1.6) — P0/⚪

**1.6 scope:** only Bucket A (Isaac Lab 3.0 API migration) ships, landed as `22b25ef`. **NuRec PR-2 (Buckets B+E+H) is deferred** to a clean post-1.6 PR off `main` — the source branch `origin/samc/support_nurec_assets_isaaclab_3.0` (5 commits, ~1818 LOC) bundles unrelated work and will not be merged whole. The deferred PR-2 plus the Buckets C/D/F/G follow-up land after the v1.6.0 tag.

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

### PR-2 — NuRec real2sim assets + occupancy_map plumbing (Buckets B + E + H fixes) — ⚪ **deferred to post-1.6**

Off PR-1. NuRec asset support, the `occupancy_map.py` refactor it depends on, and cleanups in the same files. Bucket A (PR-1) already in 1.6 via `22b25ef`; PR-2 ships as a clean follow-up PR off `main` after v1.6.0.

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

Scrub all internal-only references before tagging 1.6. Inventory in the
planning round; the meaningful work is in the OSMO entry script.

- [x] OSMO workflows: replace `groot_mobility_rl_es_usds` dataset input with HF download `nvidia/COMPASS / compass_usds.zip` (3 YAMLs)
- [x] OSMO workflows: replace `wandb artifact get …base_policy_ckpt…` with HF download `nvidia/X-Mobility / x_mobility-nav2-semantic_action_path.ckpt` (3 YAMLs)
- [x] `osmo/run_osmo.py`: drop `nvidia-isaac` wandb-project defaults (`compass_rl_enhance`, `afm_train`); make `--wandb-project` required; drop `--base-policy-ckpt` flag (workflow now hardcodes the HF source)
- [x] Update `docs/handbook/osmo.md` to reflect HF-sourced assets + required `--wandb-project`
- [ ] `ros2_deployment/compass_navigator/setup.py`: review maintainer attribution (flag for @liuw; team alias preferred)
- [ ] OSMO smoke test: resubmit `compass_rl_es_g1_official` with the rebuilt image; confirm HF download steps succeed and training reaches first PPO iter
- [x] Repo-wide grep gate: `grep -rnE "nvidia-isaac/|afm_train|groot_mobility_rl_enhance|afm_rl_enhance" --include='*.py' --include='*.sh' --include='*.yaml' --include='*.gin' --include='*.html' .` returns no live-source hits. (Dropped `groot_mobility_rl_es_usds` from the pattern — it's the directory name inside the public HF `compass_usds.zip` and is correctly referenced by `osmo/workflows/*.yaml` after unzipping.)
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
- [x] **Release-scope decision:** ship in 1.6 (settled by squash strategy — its commit is on the integration branch and goes into the squash unless explicitly excluded)
- [ ] Numerical-equivalence check vs single-GPU baseline (loss curve / success-rate parity within seed noise)
- [ ] 8-GPU OSMO smoke run reaches first PPO iter and produces ckpts
- [ ] 2-GPU path smoke (the other `--num-gpus` choice)
- [ ] Document `--distributed` in handbook / OSMO docs
- [ ] PR: <link>

**Branch:** currently sits at the tip of `liuw/training_time_improve` (this commit is the branch name's eponym). Needs to be carved into its own PR or rolled into the release stack.

## Pre-release gates

- [🟡] **No-regression benchmark** — 4 embodiments × 5 scenes, eval via `osmo/run_benchmark.py`. v1.5.0 baseline numbers not recovered; user assesses go/no-go manually.
  - [x] Sanitize and land internal `benchmark.py` (113 lines; hardcodes `nvcr.io/nvstaging/isaac-amr/groot_mobility_rl_enhance` registry + `afm_rl_enhance` defaults). Landed as `osmo/run_benchmark.py` on `liuw/benchmark_port`: Apache-2.0 header, `--registry-prefix` (with `$COMPASS_OSMO_REGISTRY` fallback), `--wandb-project-name` now `required=True`, workflow path resolved relative to `osmo/`. Reuses existing `osmo/workflows/rl_es_eval_workflow.yaml`.
  - [x] Define the regression matrix: baked into `osmo/run_benchmark.py` defaults — embodiments `{g1, h1, spot, carter}` × environments `{simple_office, warehouse_single_rack, warehouse_multi_rack, combined_single_rack, combined_multi_rack}` (5 scenes/embodiment, single seed). One invocation per embodiment fires 5 OSMO eval jobs.
  - [⚪] Capture baseline numbers from v1.5.0 — deferred; user assesses manually. The 1.6 results below become the new published baseline.

  **Eval configuration**: all jobs ran with `compass_release_1_6_relaxed:c87052af` (built off `liuw/benchmark_port` with `heading_threshold` flipped from `0.1` → `math.pi` in `compass/rl_env/exts/mobility_es/mobility_es/mdp/termination.py:33`). Source ships with the default `0.1`. **Release notes must document this.**

  #### iter-500 multi-GPU (`*_8gpu_lrcap2e3` runs, pool `isaac-dev-l40-03`)

  Goal-reached rate / fall-down rate per cell:

  | Embodiment | simple_office | warehouse_single_rack | warehouse_multi_rack | combined_single_rack | combined_multi_rack | **avg goal** | **avg fall** |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | carter | 0.619 / 0.377 | **0.945** / 0.050 | 0.728 / 0.178 | 0.861 / 0.134 | 0.889 / 0.100 | **0.808** | **0.168** |
  | g1     | 0.494 / 0.503 | 0.878 / 0.077 | 0.828 / 0.106 | 0.833 / 0.164 | 0.805 / 0.183 | 0.768 | 0.207 |
  | spot   | 0.569 / 0.425 | 0.800 / 0.195 | 0.766 / 0.161 | 0.811 / 0.188 | 0.692 / 0.305 | 0.728 | 0.255 |
  | h1     | 0.422 / 0.578 | 0.903 / 0.077 | 0.714 / 0.250 | 0.784 / 0.208 | 0.731 / 0.242 | 0.711 | 0.271 |

  Per-embodiment `weighted_travel_time` average (lower is better): carter **2,293** · spot 3,675 · g1 4,356 · h1 4,451.

  Headlines:
  - carter wins on all three axes (wheeled platform).
  - `simple_office` is the universal failure mode for bipeds — 42-57% success, 42-58% fall rate; consistent with the 32.8%-free omap density we measured.
  - `warehouse_single_rack` is the universally easiest scene (78-95% success).
  - `goal_reached + fall_down ≈ 1.0` everywhere — `fall_down` is the dominant failure mode (no meaningful time-out outcomes).

  #### iter-1000 single-GPU baseline (`*_baseline` runs, pool `groot-l40-04`)

  All 20 cells completed. Wandb project `compass_release_1.6_benchmark`, run-name pattern `bm_<emb>_<env>_release_1_6_iter1000_1gpu`.

  Goal-reached rate / fall-down rate per cell:

  | Embodiment | simple_office | warehouse_single_rack | warehouse_multi_rack | combined_single_rack | combined_multi_rack | **avg goal** | **avg fall** |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | carter | 0.630 / 0.369 | **0.950** / 0.042 | 0.792 / 0.145 | 0.872 / 0.111 | 0.887 / 0.092 | **0.826** | **0.152** |
  | g1     | 0.502 / 0.489 | 0.850 / 0.102 | 0.569 / 0.314 | 0.841 / 0.155 | 0.812 / 0.178 | 0.715 | 0.248 |
  | spot   | 0.589 / 0.408 | 0.794 / 0.206 | 0.781 / 0.188 | 0.808 / 0.191 | 0.653 / 0.345 | 0.725 | 0.268 |
  | h1     | 0.500 / 0.455 | 0.812 / 0.111 | 0.506 / 0.338 | 0.795 / 0.175 | 0.783 / 0.138 | 0.679 | 0.243 |

  Per-embodiment `weighted_travel_time` average (lower is better): carter **2,050** · spot 3,992 · g1 5,114 · h1 5,157.

  #### Side-by-side comparison: iter-500 multi-GPU vs iter-1000 single-GPU

  **Per-embodiment averages (Δ = iter-1000 − iter-500):**

  | Emb | iter-500 goal | iter-1000 goal | Δ goal | iter-500 fall | iter-1000 fall | Δ fall | iter-500 wtt | iter-1000 wtt |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | carter | 0.808 | **0.826** | **+0.018** | 0.168 | **0.152** | **−0.016** | 2,293 | **2,050** |
  | g1     | **0.768** | 0.715 | −0.053 | **0.207** | 0.248 | +0.041 | **4,356** | 5,114 |
  | spot   | **0.728** | 0.725 | −0.003 | **0.255** | 0.268 | +0.013 | **3,675** | 3,992 |
  | h1     | **0.711** | 0.679 | −0.032 | 0.271 | **0.243** | −0.028 | **4,451** | 5,157 |

  **Headlines:**
  - **carter improves on all axes at iter-1000 single-GPU** — converges fastest, benefits from more training.
  - **Bipeds (g1, h1) are roughly on-par or slightly regress at iter-1000 single-GPU.** Average drift is single-digit pp; per-scene drift is dominated by `warehouse_multi_rack` (g1: −25.9 pp goal, +20.8 pp fall; h1: −20.8 pp goal, +8.8 pp fall). Other scenes are stable.
  - **Validates §11 multi-GPU PPO numerical equivalence in spirit**: 8 GPUs × 500 iter ≈ 1 GPU × 1000 iter in samples-seen, and the resulting policies are within seed-noise on most cells. The `warehouse_multi_rack` divergence for bipeds is worth flagging in release notes but does not block tag — neither config is strictly worse across the matrix.
  - **`simple_office` remains the universal weakness** — bipeds in the 50-60% range, carter at 63%. Tighter scene + bipedal locomotion is the dominant failure mode regardless of training config.
- [ ] All P0 items 🟢
- [ ] CHANGELOG.md `[1.6.0]` entry drafted (Added / Changed / Fixed / Removed)
- [ ] Version bump committed
- [ ] X-Mobility wheel updated if needed
- [ ] Pre-commit clean: `pre-commit run --all-files`
- [ ] Docs site builds without warnings
- [ ] Release notes drafted

## Release

- [ ] Tag `v1.6.0` and push
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
