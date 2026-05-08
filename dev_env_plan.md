# Docker-as-venv dev environment plan

> Quality-of-life follow-up to the COMPASS 2.0 release. Live shape of the work
> is in this file; tracker entry is in `release_tracker.md` § 7.

## Context

Pre-PR-7 first-run UX is **6–8 manual steps and ~30–60 min**: clone Isaac Lab,
checkout `v3.0.0-beta1`, `./isaaclab.sh --install`, set `ISAACLAB_PATH`, create
a venv, `pip install -r requirements.txt` through the wrapper, install the
X-Mobility wheel, download the X-Mobility checkpoint, download `compass_usds.zip`
and unzip. The bare-metal path also breaks on the wrong host Python / stale
Vulkan SDK.

PR-1 (`liuw/isaaclab_3.0_migration`) already bakes those installs into
`docker/Dockerfile.rl`, so the *image* is one-step ready. PR-7 fills in the
host-facing UX.

**Goal:** cut first-run UX to **"3 commands, ~3 min after the image build"**
**and** make the steady-state dev loop feel like a Python venv — host-side
editor, host-side shell, but every `python`/`pip`/`tensorboard` invocation
transparently routed through the container.

## Three-layer dev model

```
Host shell  ──►  shim PATH  ──►  docker exec  ──►  daemon container
(your editor,    (auto-set by                       (compass-rl image,
 your terminal,  `source docker/activate`)          repo bind-mounted at
 your prompt)                                        /workspace/COMPASS)
```

`source docker/activate` brings up a long-running daemon container if needed,
prepends a tmp shim dir to `PATH`, and rewrites `PS1` to show `(compass-rl)`.
The shims (`python`, `pip`, `isaaclab.sh`, `tensorboard`, `pytest`, `yapf`,
`pylint`, `pre-commit`) each `docker exec` the same-named binary inside the
container, with the host CWD translated to the container path. `deactivate`
reverts PATH/PS1 and removes the shim dir; the container keeps running until
`./docker/run.sh down`.

## Layout

```
COMPASS/
├── docker/
│   ├── Dockerfile.rl         # +5 lines vs PR-1: WORKDIR /workspace/COMPASS, python wrapper
│   ├── Dockerfile.distillation
│   ├── run.sh                # build/assets/up/down/exec/shell/status
│   ├── activate              # source me; shim PATH + (compass-rl) prompt
│   ├── prepare_assets.sh     # HF downloader (USDs + X-Mobility ckpt)
│   └── README.md             # subcommand reference + troubleshooting
├── osmo/                     # unchanged (uses the same image, different launcher)
└── README.md                 # Quick Start leads with ./docker/run.sh
```

Everything Docker-related lives under `docker/`. No new top-level dir, no
`run.sh`-vs-`run.py` confusion at the repo root.

## Elegance: one bind-mount covers everything

Single source of truth for mount + env args is `_compass_run_args()` inside
`docker/run.sh`. The complete list:

```bash
-v "${REPO_ROOT}:/workspace/COMPASS"               # repo (covers code + configs + ./assets/)
-v "/tmp/.X11-unix:/tmp/.X11-unix:rw"              # X server socket
-v "${HOME}/.cache/compass/kit:/isaac-sim/kit/cache"  # writable shader cache
-e HOME=/workspace/COMPASS                         # routes pip / pre-commit / hf caches into the bind-mount
-e DISPLAY=$DISPLAY
-e WANDB_API_KEY=$WANDB_API_KEY
-e HF_TOKEN=$HF_TOKEN
-e ACCEPT_EULA=Y
-e OMNI_KIT_ALLOW_ROOT=1
--gpus all
--net=host
--user "$(id -u):$(id -g)"
--passwd-entry "$(id -un):x:$(id -u):$(id -g):COMPASS dev:/workspace/COMPASS:/bin/bash"
--workdir /workspace/COMPASS
```

That's it: **three bind-mounts, six env vars**. Compare to
`robotic_grounding/workflow/run.sh:101-218` which mounts ~10 paths individually
because it isn't structured around a single repo root.

### Why `/workspace/COMPASS` instead of `/workspace`

The base image (`nvcr.io/nvidia/isaac-lab:3.0.0-beta1`) already places Isaac Lab
at `/workspace/isaaclab`. Bind-mounting `$(pwd) → /workspace` would shadow it.
Mounting at `/workspace/COMPASS` keeps `/workspace/isaaclab` accessible at its
expected `${ISAACLAB_PATH}` path while still putting the host repo at a stable
location inside the container.

## Container naming

```bash
COMPASS_CONTAINER="compass-$(id -un)-$(printf '%s' "${REPO_ROOT}" | sha1sum | cut -c1-8)"
```

Hash of the absolute repo path → multiple checkouts of COMPASS coexist without
colliding, and the same checkout always lands on the same container name.

## `python` wrapper (Dockerfile.rl change)

`isaaclab.sh` is a Python CLI (it `exec`s `python -c "from isaaclab.cli import cli; cli()" "$@"`),
so symlinking `python → isaaclab.sh` would put you inside the CLI rather than at
a Python prompt. Instead the Dockerfile installs a real wrapper:

```dockerfile
RUN printf '#!/usr/bin/env bash\nexec "${ISAACLAB_PATH}/_isaac_sim/python.sh" "$@"\n' \
        > /usr/local/bin/python \
 && chmod +x /usr/local/bin/python \
 && ln -sf /usr/local/bin/python /usr/local/bin/python3
```

Now `python run.py` inside the container resolves directly to Isaac Sim's
bundled `python.sh`, bypassing the CLI overhead and matching what
`isaaclab.sh -p` would have done.

## Git workflow

`git` itself runs **on the host** — no shim. Repo is on host, SSH/GPG keys are
on host, `git commit -s` (DCO) and `git push` work without extra plumbing.
File ownership stays consistent because the container runs as
`--user "$(id -u):$(id -g)"`.

The wrinkle: pre-commit hooks (yapf, pylint, clang-format) live in the
container's Python, not on the host. So:

- **Activate before committing.** `pre-commit` is in the shim list, so when
  host `git commit -s` invokes it via PATH, the shim routes the hook into the
  container.
- **Hook-env cache** lives at `./.cache/pre-commit/` (inside the bind-mount),
  surviving `down` + `up`. `.cache/` is gitignored.

## Files

| Path | Status | Why |
|------|--------|-----|
| `docker/run.sh` | NEW | Subcommand wrapper around `docker {build,run,exec,rm}`; single source of truth for mount/env args |
| `docker/activate` | NEW | Sourced; shim PATH, PS1 prefix, `deactivate` |
| `docker/prepare_assets.sh` | NEW | HF downloader (USDs + X-Mobility ckpt) |
| `docker/README.md` | NEW | Subcommand reference + troubleshooting |
| `docker/Dockerfile.rl` | MODIFIED | `WORKDIR /workspace/COMPASS`; install paths under `/workspace/COMPASS`; `python`/`python3` wrapper |
| `.dockerignore` | MODIFIED | Add `./assets/`, `./.cache/`, `./.git/` |
| `.gitignore` | MODIFIED | Add `/assets/`, `/.cache/` |
| `README.md` | MODIFIED | Quick Start leads with `./docker/run.sh` + `source ./docker/activate`; bare-metal moves under "Manual install" |
| `release_tracker.md` | MODIFIED | Item #7 status flipped 🟡; checklist progress |

## Branch + commit

- Branch `liuw/dev_environment`, off `liuw/auto_omap_from_usd` (PR-5 head, latest
  in the stack).
- Single squashed DCO-signed commit titled
  "Add Docker-as-venv dev environment (`docker/run.sh` + `docker/activate`)".

## Verification

1. **`./docker/run.sh build`** — image builds; `docker images | grep compass-rl` shows the tag.
2. **`./docker/run.sh assets`** — fresh `./assets/` dir gets `usd/` and `x_mobility.ckpt`; second run is a no-op.
3. **`./docker/run.sh up`** — daemon container starts; `./docker/run.sh status` reports `running` with the path-hashed name.
4. **`source ./docker/activate`** — prompt becomes `(compass-rl) …`; `which python` resolves to the shim dir; `python -V` reports Isaac Sim's bundled Python.
5. **CWD translation** — `cd compass/rl_env/exts/mobility_es && python -c 'import os; print(os.getcwd())'` prints `/workspace/COMPASS/compass/rl_env/exts/mobility_es`.
6. **End-to-end smoke** — from the activated shell:
   `python run.py -c configs/train_config.gin -o /tmp/out -b ./assets/x_mobility.ckpt --num_envs 1 --enable_camera --headless` reaches first PPO iteration.
7. **`deactivate`** — prompt + PATH revert; shim dir removed from `/tmp`.
8. **`./docker/run.sh down`** — container removed; `docker ps -a | grep compass-` returns nothing.
9. **OSMO compatibility** — `osmo/run_osmo.py train ... --image <built-via-docker/run.sh>` still works.
10. **Pre-commit clean (activated shell)** — `pre-commit run --files docker/run.sh docker/activate docker/prepare_assets.sh docker/README.md docker/Dockerfile.rl README.md`.
11. **Git commit signs cleanly** — `git commit -s -m "test" --allow-empty` from an activated shell completes; resulting commit shows DCO trailer.

## Out of scope

- Replacing `osmo/run_osmo.py` — different concern; same image.
- Removing the bare-metal install in `compass/rl_env/README.md` — kept for non-Docker users.
- `docker compose` — single container, single image; compose adds a dep without UX gain.
- VSCode `.devcontainer` — orthogonal; works alongside the activate model.
- Pushing branches / opening PRs — held per current standing instruction.
