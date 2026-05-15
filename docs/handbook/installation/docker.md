# Docker-as-venv install

`./docker/run.sh` + `source ./docker/activate` give you a Python-venv-like dev
loop that runs commands inside the COMPASS container while keeping your editor,
terminal, and prompt on the host. Edit code with whatever editor you like; the
bind-mount means changes hot-reload.

:::{note}
Docker is the recommended path because the image bakes Isaac Lab, the
X-Mobility wheel, and the `mobility_es` extension at build time — no host
Python / Vulkan SDK to manage. Bare-metal install is possible but not
documented end-to-end in the handbook; see the
[Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta1/source/setup/installation/index.html)
and `${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e compass/rl_env/exts/mobility_es`.
:::

## How it works

```
Host shell  →  shim PATH  →  docker exec  →  daemon container
(editor,       (set by         (translates      (compass-rl image,
 terminal,      `source         host CWD        repo bind-mounted
 prompt)        docker/         to container    at /workspace/COMPASS)
                activate`)      path)
```

`source ./docker/activate` brings up the container if it isn't already running,
prepends a tmp shim dir to `PATH`, and rewrites your prompt to `(compass-rl)`.
The shims (`python`, `pip`, `tensorboard`, `pytest`, `yapf`, `pylint`,
`pre-commit`, …) each `docker exec` the same-named binary inside the container,
with the host CWD translated to the container path. `deactivate` reverts.

## Prerequisites

- Docker (Engine 24+ recommended)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) so `--gpus all` works
- An NVIDIA GPU + driver that satisfy the upstream [Isaac Lab system requirements](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta/source/setup/installation/index.html#system-requirements)
- A HuggingFace account with access to [nvidia/COMPASS](https://huggingface.co/nvidia/COMPASS) and a [token](https://huggingface.co/settings/tokens)
- An X server (only if you want `--viz kit` to render — headless training works fine without)

For the three-command quick-start path, see [Quick start](../quickstart.md).
The rest of this page is the subcommand reference, multi-checkout / multi-GPU
notes, the git workflow, and troubleshooting.

## Subcommand reference

```
./docker/run.sh build [tag]               # docker build -f docker/Dockerfile.rl
./docker/run.sh assets [--hf-token TOK]   # download USDs + ckpt to ./assets
./docker/run.sh up                        # idempotent: start daemon container
./docker/run.sh down                      # docker rm -f the container
./docker/run.sh exec <cmd> [...]          # one-shot exec inside the container
./docker/run.sh shell                     # escape hatch: drop into bash
./docker/run.sh status                    # show container + image state
```

### `build`

Builds `compass-rl:latest` (override the tag with `./docker/run.sh build mytag`).
Re-running is fast because Docker layer-caches.

### `assets`

Downloads:

| File | URL | Lands at |
|------|-----|---------|
| `compass_usds.zip` (extracted) | `huggingface.co/nvidia/COMPASS/...` | `./assets/usd/` |
| `x_mobility-nav2-semantic_action_path.ckpt` | `huggingface.co/nvidia/X-Mobility/...` | `./assets/x_mobility.ckpt` |

Cache-aware: skips files that already exist. `--force` to redownload everything.
`--cache-dir DIR` to put assets somewhere other than `./assets/`.

`./assets/` is part of the repo bind-mount, so inside the container it's at
`/workspace/COMPASS/assets/` — relative paths from the repo root work in both
host and container shells.

### `up` / `down`

`up` is idempotent: starts the container if not running, or no-op if already up.
`down` stops and removes it. The container's name is
`compass-<your-user>-<8-hex-of-repo-path>`, so multiple checkouts of COMPASS on
the same machine coexist without colliding.

### `exec` / `shell`

`exec` runs one command inside the container (`./docker/run.sh exec python --version`).
`shell` drops you into a bash inside the container — escape hatch for poking at
the image, not for daily dev (use `source ./docker/activate` for that).

### `status`

Shows whether the image is built, whether the container is running, and the
relevant paths. Run this first when something feels off.

## Multiple checkouts of COMPASS

The container name hashes the absolute repo path:

```bash
# ~/COMPASS/dev → container compass-liuw-a7f3b2c8
# ~/COMPASS/main → container compass-liuw-9d1e44f0
```

Both can run concurrently and never confuse each other. Each gets its own
bind-mount, its own pre-commit cache, its own kit shader cache.

## Multi-GPU

The container starts with `--gpus all` by default. To pin to specific GPUs,
export `NVIDIA_VISIBLE_DEVICES` before bringing the container up:

```bash
NVIDIA_VISIBLE_DEVICES=0 ./docker/run.sh up   # only GPU 0 visible inside the container
```

(Note: this currently requires editing `_compass_run_args` in
[`docker/run.sh`](https://github.com/NVlabs/COMPASS/blob/main/docker/run.sh)
to honor the env var — track via `release_tracker.md` if you need it; not
implemented in v1.)

## VSCode / Cursor

Editor + container are decoupled: VSCode runs natively on the host, opens the
repo at `~/Projects/COMPASS`, and you run `python run.py …` from VSCode's
integrated terminal **after** sourcing `./docker/activate` in that terminal. No
"Reopen in Container" needed.

## Git workflow

`git status / add / commit -s / push` run on the host as normal — no shim. SSH
keys, GPG signing, DCO sign-off all work without extra plumbing.

The only wrinkle is **pre-commit hooks** (yapf, pylint, clang-format). The
hooks live in the container, not on the host. So:

- **Activate before committing.** `source ./docker/activate` adds `pre-commit`
  to `PATH`, and the shim routes the hook invocation into the container.
- **Hook env cache** lives at `./.cache/pre-commit/` (inside the bind-mount), so
  `down` + `up` doesn't force a 30s rebuild. `.cache/` is gitignored.

## Troubleshooting

| Symptom | Likely cause | Fix |
|--------|--------------|-----|
| `Image compass-rl:latest not found` | Haven't built yet | `./docker/run.sh build` |
| `permission denied while trying to connect to the Docker daemon` | User not in `docker` group | `sudo usermod -aG docker $USER` then re-login |
| `Vulkan/X11: cannot open display` from inside container | X server not forwarded | `xhost +SI:localuser:$(id -un)` (run.sh tries this automatically; may need manual on systems where it fails silently) |
| Isaac Sim shader cold-compile every run | Shader cache wiped | Check `~/.cache/compass/kit/` is writable; reuses across runs |
| `pre-commit: command not found` on `git commit` | Forgot to activate | `source ./docker/activate` |
| Container won't `up`: "another container with the same name" | Stale container from a crash | `./docker/run.sh down` then `./docker/run.sh up` |

## What lives where

```
docker/
├── Dockerfile.rl            # Image: Isaac Lab 3.0-beta1 + COMPASS deps + python wrapper
├── Dockerfile.distillation  # Used by docker-only distillation runs (unrelated to dev env)
├── run.sh                   # build / assets / up / down / exec / shell / status
├── activate                 # source me: shim PATH + (compass-rl) prompt prefix
└── prepare_assets.sh        # invoked by `run.sh assets`

~/.cache/compass/kit/        # writable Isaac Sim shader cache (survives container teardown)
./assets/                    # USDs + X-Mobility ckpt (gitignored; populated by `run.sh assets`)
./.cache/                    # pip / pre-commit / huggingface caches (gitignored)
```
