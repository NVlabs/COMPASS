#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Manage the COMPASS docker dev container.
#
# Usage:
#   ./docker/run.sh build [tag]               # build the image (default tag: latest)
#   ./docker/run.sh assets [--hf-token TOK]   # download USDs + X-Mobility ckpt to ./assets
#   ./docker/run.sh up                        # idempotent: start daemon container if not running
#   ./docker/run.sh down                      # docker rm -f the container
#   ./docker/run.sh exec <cmd> [...]          # one-shot exec inside the running container
#   ./docker/run.sh shell                     # escape hatch: drop into bash inside the container
#   ./docker/run.sh status                    # show container + image state
#
# Companion: `source ./docker/activate` for a Python-venv-like dev experience.

set -eu

# Repo root: the dir containing this script's parent (docker/run.sh → COMPASS/).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Image tag. Override via `./docker/run.sh build mytag` then `COMPASS_IMAGE_TAG=mytag ./docker/run.sh up`.
IMAGE_TAG="${COMPASS_IMAGE_TAG:-latest}"
IMAGE_NAME="compass-rl:${IMAGE_TAG}"

# Container name = "compass-<user>-<8-hex of repo path>". Hashing the absolute repo
# path means multiple checkouts of COMPASS coexist without colliding, and the same
# checkout always lands on the same container across shells / reboots.
_REPO_HASH="$(printf '%s' "${REPO_ROOT}" | sha1sum | cut -c1-8)"
CONTAINER_NAME="compass-$(id -un)-${_REPO_HASH}"

# Writable shader cache lives outside the repo so a `git clean` doesn't nuke it.
KIT_CACHE_DIR="${HOME}/.cache/compass/kit"

# ──────────────────────────────────────────────────────────────────────────────
# Output helpers (lifted from ros2_deployment/prepare_assets.sh).
# ──────────────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARNING]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step()    { echo -e "${BLUE}[STEP]${NC} $*"; }

# ──────────────────────────────────────────────────────────────────────────────
# Single source of truth for `docker run` flags. Used by `up`.
# Keep the mount list minimal — the repo bind-mount covers code + configs +
# assets, so we only add X11 socket forwarding and the kit shader cache.
# ──────────────────────────────────────────────────────────────────────────────
CONT_REPO_DIR="/workspace/COMPASS"

_compass_run_args() {
    mkdir -p "${KIT_CACHE_DIR}"
    local args=(
        --name "${CONTAINER_NAME}"
        --gpus all
        --net=host
        --user "$(id -u):$(id -g)"
        # Isaac Sim's install at /isaac-sim is mode drwxr-x--- with group
        # `isaac-sim` (GID 1234 in the base image). Adding our host user to
        # that group at runtime is the minimal way to grant access without
        # modifying the base image.
        --group-add 1234
        # Repo bind-mount overrides the COPY in Dockerfile.rl so host edits hot-reload.
        # Path is /workspace/COMPASS to leave /workspace/isaaclab (Isaac Lab installation,
        # from the base image) intact and accessible.
        -v "${REPO_ROOT}:${CONT_REPO_DIR}"
        -v "/tmp/.X11-unix:/tmp/.X11-unix:rw"
        -v "${KIT_CACHE_DIR}:/isaac-sim/kit/cache"
        # HOME=/workspace/COMPASS puts pip / pre-commit / huggingface caches under
        # the bind-mount so they survive `down` + `up`. Caches land in ./.cache/
        # which is gitignored.
        -e "HOME=${CONT_REPO_DIR}"
        -e "DISPLAY=${DISPLAY:-}"
        -e "WANDB_API_KEY=${WANDB_API_KEY:-}"
        -e "HF_TOKEN=${HF_TOKEN:-}"
        # Already ENV in Dockerfile.rl; redundant flags are harmless and explicit.
        -e "ACCEPT_EULA=Y"
        -e "OMNI_KIT_ALLOW_ROOT=1"
        # Bind-mount host /etc/passwd + /etc/group read-only so the container
        # resolves $(id -u)/$(id -g) to a real username. Without this, NSS
        # lookups inside the container (huggingface_hub, getpass.getuser, …)
        # fall over with "no passwd entry for uid …".
        -v "/etc/passwd:/etc/passwd:ro"
        -v "/etc/group:/etc/group:ro"
        --workdir "${CONT_REPO_DIR}"
        --detach
    )
    printf '%s\n' "${args[@]}"
}

# ──────────────────────────────────────────────────────────────────────────────
# Subcommands.
# ──────────────────────────────────────────────────────────────────────────────
cmd_build() {
    local tag="${1:-${IMAGE_TAG}}"
    step "Building ${IMAGE_NAME/:*/:${tag}} from docker/Dockerfile.rl"
    cd "${REPO_ROOT}"
    docker build --network=host -f docker/Dockerfile.rl -t "compass-rl:${tag}" .
    info "Build complete: compass-rl:${tag}"
}

cmd_assets() {
    "${REPO_ROOT}/docker/prepare_assets.sh" "$@"
}

cmd_up() {
    if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
        info "Container already running: ${CONTAINER_NAME}"
        return 0
    fi
    if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
        # Stopped container with the same name — start it back up.
        step "Restarting existing container: ${CONTAINER_NAME}"
        docker start "${CONTAINER_NAME}" >/dev/null
        info "Container running: ${CONTAINER_NAME}"
        return 0
    fi

    if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
        error "Image ${IMAGE_NAME} not found. Run: ./docker/run.sh build"
        exit 1
    fi

    step "Starting container: ${CONTAINER_NAME} (image: ${IMAGE_NAME})"
    # Allow X11 connections from the docker user (no-op if no X server / xhost).
    xhost +SI:localuser:"$(id -un)" >/dev/null 2>&1 || true

    # Override the image's entrypoint (Isaac Sim's runheadless.sh) with `sleep
    # infinity` so the container stays alive as a passive shell host. We exec
    # actual commands via `docker exec`. `--entrypoint` accepts only the
    # executable; the image arg becomes the first positional argument.
    local args=()
    while IFS= read -r line; do args+=("$line"); done < <(_compass_run_args)
    args+=(--entrypoint /bin/sleep)
    docker run "${args[@]}" "${IMAGE_NAME}" infinity >/dev/null
    info "Container running: ${CONTAINER_NAME}"
}

cmd_down() {
    if ! docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
        info "No container to stop: ${CONTAINER_NAME}"
        return 0
    fi
    step "Removing container: ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null
    info "Container removed."
}

cmd_exec() {
    if [ "$#" -eq 0 ]; then
        error "Usage: ./docker/run.sh exec <cmd> [args...]"
        exit 2
    fi
    cmd_up >/dev/null    # Lazy up: don't make the user remember.
    # CWD translation: if the host CWD is inside the repo, mirror the same
    # subdirectory inside the container so relative paths work.
    local host_pwd cont_pwd
    host_pwd="$(pwd)"
    case "${host_pwd}" in
        "${REPO_ROOT}"|"${REPO_ROOT}"/*)
            cont_pwd="${CONT_REPO_DIR}${host_pwd#${REPO_ROOT}}"
            ;;
        *)
            cont_pwd="${CONT_REPO_DIR}"
            ;;
    esac
    # Add -t only when stdin AND stdout are a terminal — `docker exec -t`
    # against a non-tty stdin errors with "cannot attach stdin to a TTY-enabled".
    local exec_flags="-i"
    if [ -t 0 ] && [ -t 1 ]; then exec_flags="-it"; fi
    exec docker exec ${exec_flags} -w "${cont_pwd}" "${CONTAINER_NAME}" "$@"
}

cmd_shell() {
    cmd_up >/dev/null
    local exec_flags="-i"
    if [ -t 0 ] && [ -t 1 ]; then exec_flags="-it"; fi
    exec docker exec ${exec_flags} -w "${CONT_REPO_DIR}" "${CONTAINER_NAME}" bash
}

cmd_status() {
    echo "Image:     ${IMAGE_NAME}"
    if docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
        echo "Image OK:  $(docker image inspect "${IMAGE_NAME}" --format '{{.Id}} ({{.Created}})')"
    else
        echo "Image OK:  (not built — run ./docker/run.sh build)"
    fi
    echo "Container: ${CONTAINER_NAME}"
    if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
        echo "State:     running"
    elif docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
        echo "State:     stopped"
    else
        echo "State:     absent"
    fi
    echo "Repo root: ${REPO_ROOT}"
    echo "Kit cache: ${KIT_CACHE_DIR}"
}

usage() {
    sed -n '5,16p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    exit "${1:-0}"
}

# ──────────────────────────────────────────────────────────────────────────────
# Dispatch.
# ──────────────────────────────────────────────────────────────────────────────
if [ "$#" -eq 0 ]; then usage 2; fi
SUB="$1"; shift
case "${SUB}" in
    build)   cmd_build "$@"  ;;
    assets)  cmd_assets "$@" ;;
    up)      cmd_up "$@"     ;;
    down)    cmd_down "$@"   ;;
    exec)    cmd_exec "$@"   ;;
    shell)   cmd_shell "$@"  ;;
    status)  cmd_status "$@" ;;
    -h|--help|help) usage 0 ;;
    *)
        error "Unknown subcommand: ${SUB}"
        usage 2
        ;;
esac
