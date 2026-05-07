# Docker-driven dev environment plan

> Quality-of-life follow-up to the COMPASS 2.0 release. Not a release blocker; revisit after the P0 items in `release_tracker.md` land.

## Context

The current README quick-start is **6–8 manual steps and ~30–60 min on first run**: clone Isaac Lab, checkout `v3.0.0-beta1`, `./isaaclab.sh --install`, set `ISAACLAB_PATH`, create a venv, `pip install -r requirements.txt` through the Isaac Lab wrapper, install the X-Mobility wheel, download the X-Mobility checkpoint from HuggingFace, download `compass_usds.zip` and unzip into `compass/rl_env/exts/mobility_es/mobility_es/usd/`.

PR-1 (`liuw/isaaclab_3.0_migration`) already bakes those installs into `docker/Dockerfile.rl` so the *image* is one-step ready, but the *user-facing quick start* still walks through the host-side install dance.

Reference pattern (well-liked by the team): `/home/liuw/Projects/video_to_data/robotic_grounding/workflow/run.sh` — single-script entry point with `build` / `start` / `shell` / `exec` / `stop` subcommands that runs an interactive container with the repo bind-mounted, GPU + X11 forwarded, host UID/GID preserved. Daily UX: `./workflow/run.sh start` and you're in a shell that behaves like a venv.

**Goal:** cut COMPASS first-time UX from "30–60 min, 6 manual steps" to **"3 commands, ~3 min after the image build"** without removing the bare-metal path for users who can't use Docker.

## Approach

Add a new top-level `workflow/` directory that mirrors the robotic_grounding shape (familiar to the team, pattern proven). The image itself stays `docker/Dockerfile.rl` — already correct after PR-1; no second Dockerfile needed. The dev script just wraps `docker run` with the right flags.

```
COMPASS/
├── workflow/                    ← NEW
│   ├── run.sh                   ← single entry point (subcommand-based)
│   ├── prepare_assets.sh        ← USDs + X-Mobility ckpt downloader (cache-aware)
│   └── README.md                ← quickstart + reference for run.sh subcommands
├── docker/
│   ├── Dockerfile.rl            ← unchanged, reused by workflow/ AND osmo/
│   └── Dockerfile.distillation  ← unchanged
├── osmo/                        ← unchanged (uses same image, different launcher)
└── README.md                    ← updated Installation section to lead with workflow/run.sh
```

## `workflow/run.sh` design

Subcommands (mirror `robotic_grounding/workflow/run.sh:1-18`):

```
./workflow/run.sh build [tag]                     # docker build -f docker/Dockerfile.rl
./workflow/run.sh assets [--hf-token TOK]         # download/cache USDs + X-Mobility ckpt to ./assets
./workflow/run.sh start [tag] [gpu]               # start container as daemon + drop into bash
./workflow/run.sh shell [tag] [gpu]               # re-enter the running container
./workflow/run.sh exec  [tag] [gpu] -- <cmd>      # one-shot command in container
./workflow/run.sh stop  [tag] [gpu]               # docker stop + rm
```

`start` does (lifted from `robotic_grounding/workflow/run.sh:101-218` with COMPASS-specific paths):

| Aspect | Detail |
|--------|--------|
| Image tag | `compass-rl:${tag-latest}` (default `latest`) |
| Container name | `compass-rl-${tag}-gpu${GPU_DEVICE}` so multiple devs / multiple GPUs don't collide |
| GPU select | `--gpus device=${GPU_DEVICE-all}` |
| Repo bind-mount | `$(pwd) → /workspace` (overrides the COPY in Dockerfile.rl so host edits hot-reload, while the editable `pip install -e ./compass/rl_env/exts/mobility_es` still resolves) |
| Assets bind-mount | `./assets/usd → /workspace/compass/rl_env/exts/mobility_es/mobility_es/usd:ro` and `./assets/x_mobility.ckpt → /workspace/x_mobility.ckpt:ro` if present |
| X11 | `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw --net=host` (matches what we used in PR-1's `--viz kit` smoke) |
| Isaac Sim cache | `~/.cache/compass/${container}/kit` bind-mounted to writable kit cache dirs (parallel to `robotic_grounding/workflow/run.sh:184-187`) |
| Credentials | `WANDB_API_KEY` and `HF_TOKEN` forwarded from env (or `~/.wand_api_key` fallback per robotic_grounding pattern) |
| EULA | `-e ACCEPT_EULA=Y -e OMNI_KIT_ALLOW_ROOT=1` (already in Dockerfile.rl as ENV; redundant flags are harmless and explicit) |
| UID/GID mapping | Custom `/etc/passwd` + `/etc/group` written to a tmp file and bind-mounted (lifted from `robotic_grounding/workflow/run.sh:165-174`) so files written from inside the container are owned by the host user |
| Entry | `docker run -d ... bash` then `docker exec -it ... bash` to drop user into a shell |

`assets` does (extends `ros2_deployment/prepare_assets.sh` patterns: token check, file integrity, wget/curl fallback, colored output):

```bash
./workflow/run.sh assets [--hf-token TOK] [--cache-dir ./assets]
```

- Reads `$HF_TOKEN` or `--hf-token`; errors with the same instructional message style as `prepare_assets.sh:48-63` if missing
- Downloads `compass_usds.zip` from `https://huggingface.co/nvidia/COMPASS/resolve/main/compass_usds.zip` to `./assets/compass_usds.zip`, validates non-empty, unzips to `./assets/usd/` (skips re-download if the unpacked dir exists)
- Downloads `https://huggingface.co/nvidia/X-Mobility/resolve/main/x_mobility-nav2-semantic_action_path.ckpt` to `./assets/x_mobility.ckpt` (skips if present and non-empty)
- Prints a summary like `prepare_assets.sh:128-151` with cache paths and sizes

`build` does:

```bash
docker build --network=host -f docker/Dockerfile.rl -t compass-rl:${tag} .
```

Adding a `.dockerignore` entry for `./assets/` so it doesn't bloat the build context.

## Convenience inside the container

Add to `docker/Dockerfile.rl` (one new line):

```dockerfile
RUN ln -sf ${ISAACLAB_PATH}/isaaclab.sh /usr/local/bin/python-il && \
    ln -sf ${ISAACLAB_PATH}/isaaclab.sh /usr/local/bin/python
```

(`robotic_grounding/workflow/Dockerfile:102-107` pattern — lets users type `python run.py ...` instead of `${ISAACLAB_PATH}/isaaclab.sh -p run.py ...` once inside the container.)

This is a small Dockerfile.rl tweak. Lands in this PR (not amended into PR-1) so PR-1 stays a clean bucket-A migration.

## Updated `README.md` quickstart

Add a new section **before** the existing "Installation" details, marked as the recommended path:

````markdown
## Quick Start (Docker, recommended)

1. `./workflow/run.sh assets` — download USDs + X-Mobility ckpt (one-time, ~5 min)
2. `./workflow/run.sh build` — build the dev image (one-time, ~10 min on first run)
3. `./workflow/run.sh start` — open a shell in the container

Inside the container:
```bash
python run.py -c configs/train_config.gin -o /tmp/out -b /workspace/x_mobility.ckpt --enable_camera
```

That's it. See `workflow/README.md` for subcommands and customization (multi-GPU, viz mode, etc.).

## Manual install

If you can't use Docker, follow the bare-metal installation in `compass/rl_env/README.md`.
````

The existing collapsed `<details>` block stays — keep both paths — but the Docker path becomes the primary recommendation.

## `workflow/README.md` content

- Prerequisites: docker, NVIDIA Container Toolkit, GPU, an HF account + token
- One-page reference for each `run.sh` subcommand with examples
- Multi-GPU / multi-container instructions (`./workflow/run.sh start latest 0`, `./workflow/run.sh start latest 1` simultaneously)
- VSCode/Cursor "Attach to Running Container" tip (lifted from `robotic_grounding/README:79-81`)

## Branch + commit strategy

- New branch `liuw/dev_environment` off `liuw/isaaclab_3.0_migration` (PR-1) — the dev script benefits from PR-1's "image bakes installs" change. Sibling to PR-2 and PR-3.
- Single squashed commit: "Add Docker-driven dev environment (`workflow/run.sh`)"

## Verification

1. **`./workflow/run.sh build`** completes; `docker images | grep compass-rl` shows the tag.
2. **`./workflow/run.sh assets`** with a fresh `./assets/` dir downloads + extracts both, exits with a summary; second invocation is a no-op (cache hit).
3. **`./workflow/run.sh start`** opens a bash shell where:
   - `pwd` = `/workspace`
   - `ls compass/rl_env/exts/mobility_es/mobility_es/usd/` shows the bind-mounted USDs
   - `ls /workspace/x_mobility.ckpt` shows the bind-mounted ckpt
   - `whoami` returns the host user, not `root` (UID/GID mapping working)
   - `nvidia-smi` shows the GPU
   - `python run.py --help` works without manually invoking `isaaclab.sh -p` (thanks to the Dockerfile symlink)
4. **End-to-end smoke**: from inside the shell, `python run.py -c configs/train_config.gin -o /tmp/out -b /workspace/x_mobility.ckpt --num_envs 1 --enable_camera --headless` reaches first PPO iteration (matches PR-1's local smoke test path).
5. **`./workflow/run.sh exec latest 0 -- python run.py --help`** runs from outside the container.
6. **`./workflow/run.sh shell`** re-enters the same container after `Ctrl-D` exit.
7. **`./workflow/run.sh stop`** cleanly tears the container down.
8. **OSMO compatibility** — `osmo/run_osmo.py train ... --image <built-from-workflow>` still works. Same image, different launcher.

## Decisions baked in (flag if any are wrong)

1. Layout name: `workflow/` (matches robotic_grounding). Alternative: `dev/`. Picking `workflow/` for team-pattern familiarity.
2. Reuse `docker/Dockerfile.rl` (no `Dockerfile.dev`); only addition is the `python` symlink for in-container ergonomics.
3. Keep both quick-start paths in README (Docker primary, bare-metal kept under "Manual install").
4. `workflow/run.sh` is bash, not Python. Python would be cleaner but bash matches `ros2_deployment/build_compass_docker.sh` house style and avoids pulling Python into the host bootstrap.
5. The `python` symlink lives in this PR, not amended into PR-1. PR-1 stays a clean bucket-A migration.

## Out of scope

- Replacing `osmo/run_osmo.py` — different concern, lives on top of the same image.
- Removing the bare-metal install instructions in `compass/rl_env/README.md` — keep them for users who can't use Docker.
- Adding `docker compose` — `docker run` with `run.sh` wrapping is enough; compose adds a dependency without UX gain for our case.

## Where this fits in the release tracker

Add as a **new item #7 (P1)** in `release_tracker.md` — quality-of-life, not a release blocker. Doesn't change the dependency graph (parallel to #1, #4, #5, #6).
