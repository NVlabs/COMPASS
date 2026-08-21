#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Download COMPASS host-side assets (USDs + X-Mobility checkpoint) into ./assets/.
# Optionally download NuRec scene folders into the mobility_es USD tree.
#
# Cache-aware: skips downloads when the target already exists and is non-empty.
# Idempotent: safe to re-run.
#
# Usage:
#   ./docker/prepare_assets.sh [--hf-token TOK] [--cache-dir DIR] [--force]
#                              [--nurec-scene SCENE]
#                              [--nurec-revision REV]
#
# Environment:
#   HF_TOKEN  HuggingFace token (https://huggingface.co/settings/tokens)
#             Required because the COMPASS HF repo is gated. The flag overrides.

set -eu

# ──────────────────────────────────────────────────────────────────────────────
# Output helpers (matches ros2_deployment/prepare_assets.sh).
# ──────────────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARNING]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step()  { echo -e "${BLUE}[STEP]${NC} $*"; }

# ──────────────────────────────────────────────────────────────────────────────
# Defaults.
# ──────────────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${REPO_ROOT}/assets"
HF_TOKEN_ARG=""
FORCE=false
NUREC_SCENES=()
NUREC_REVISION="${NUREC_REVISION:-refs/pr/34}"

USDS_URL="https://huggingface.co/nvidia/COMPASS/resolve/main/compass_usds.zip"
USDS_ZIP="${CACHE_DIR}/compass_usds.zip"
# Canonical install location — `mobility_es` config files reference USDs as
# `os.path.join(os.path.dirname(__file__), "../usd/...")`, so the tree must
# live next to `config/` inside the extension. Matches what the OSMO train
# workflow does (osmo/workflows/rl_es_train_workflow.yaml).
USDS_DIR="${REPO_ROOT}/compass/rl_env/exts/mobility_es/mobility_es/usd"

CKPT_URL="https://huggingface.co/nvidia/X-Mobility/resolve/main/x_mobility-nav2-semantic_action_path.ckpt"
CKPT_FILE="${CACHE_DIR}/x_mobility.ckpt"
NUREC_REPO="nvidia/PhysicalAI-Robotics-NuRec"

# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing.
# ──────────────────────────────────────────────────────────────────────────────
usage() {
    sed -n '5,18p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
    exit "${1:-0}"
}
while [ "$#" -gt 0 ]; do
    case "$1" in
        --hf-token) HF_TOKEN_ARG="$2"; shift 2 ;;
        --nurec-scene) NUREC_SCENES+=("$2"); shift 2 ;;
        --nurec-revision) NUREC_REVISION="$2"; shift 2 ;;
        --cache-dir)
            # --cache-dir only redirects the zip + ckpt cache. USDS_DIR is
            # fixed by the package layout (see definition above).
            CACHE_DIR="$2"
            USDS_ZIP="${CACHE_DIR}/compass_usds.zip"
            CKPT_FILE="${CACHE_DIR}/x_mobility.ckpt"
            shift 2
            ;;
        --force) FORCE=true; shift ;;
        -h|--help) usage 0 ;;
        *) error "Unknown flag: $1"; usage 2 ;;
    esac
done

HF_TOKEN_EFFECTIVE="${HF_TOKEN_ARG:-${HF_TOKEN:-}}"

# ──────────────────────────────────────────────────────────────────────────────
# Steps.
# ──────────────────────────────────────────────────────────────────────────────
check_hf_token() {
    if [ -z "${HF_TOKEN_EFFECTIVE}" ]; then
        error "HF_TOKEN not set. The COMPASS HuggingFace repo is gated."
        error ""
        error "To fix:"
        error "  1. Request access at https://huggingface.co/nvidia/COMPASS"
        error "  2. Get a token at  https://huggingface.co/settings/tokens"
        error "  3. export HF_TOKEN=hf_xxx  (or pass --hf-token hf_xxx)"
        exit 1
    fi
    info "HuggingFace token found."
}

check_downloader() {
    if command -v curl >/dev/null 2>&1; then
        DOWNLOADER=curl
    elif command -v wget >/dev/null 2>&1; then
        DOWNLOADER=wget
    else
        error "Neither curl nor wget is installed. Install one of them and re-run."
        exit 1
    fi
}

# Args: $1=URL  $2=destination path
hf_download() {
    local url="$1" dst="$2"
    case "${DOWNLOADER}" in
        curl)
            curl -L --fail --progress-bar \
                 -H "Authorization: Bearer ${HF_TOKEN_EFFECTIVE}" \
                 -o "${dst}" "${url}"
            ;;
        wget)
            wget --progress=bar:force \
                 --header="Authorization: Bearer ${HF_TOKEN_EFFECTIVE}" \
                 -O "${dst}" "${url}"
            ;;
    esac
}

prepare_dirs() {
    step "Creating ${CACHE_DIR}"
    mkdir -p "${CACHE_DIR}"
}

download_usds() {
    if [ "${FORCE}" = true ]; then
        rm -rf "${USDS_DIR}" "${USDS_ZIP}"
    fi
    if [ -d "${USDS_DIR}" ] && [ -n "$(ls -A "${USDS_DIR}" 2>/dev/null)" ]; then
        info "USDs already installed at ${USDS_DIR} — skipping."
        return 0
    fi

    step "Downloading compass_usds.zip → ${USDS_ZIP}"
    check_hf_token
    hf_download "${USDS_URL}" "${USDS_ZIP}"
    if [ ! -s "${USDS_ZIP}" ]; then
        error "Downloaded compass_usds.zip is empty. Check your HF_TOKEN and access."
        rm -f "${USDS_ZIP}"
        exit 1
    fi

    if ! command -v unzip >/dev/null 2>&1; then
        error "unzip is not installed. Install it (e.g. apt-get install unzip)."
        exit 1
    fi

    # The HF zip wraps everything in `groot_mobility_rl_es_usds/usd/<scene>/…`.
    # Stage to a temp dir, then move the inner `usd/` contents into USDS_DIR
    # (mirrors osmo/workflows/rl_es_train_workflow.yaml).
    local stage_dir
    stage_dir="$(mktemp -d "${CACHE_DIR}/usd_stage.XXXXXX")"
    step "Unzipping → ${stage_dir}"
    unzip -q -o "${USDS_ZIP}" -d "${stage_dir}"

    local inner="${stage_dir}/groot_mobility_rl_es_usds/usd"
    if [ ! -d "${inner}" ]; then
        error "Unexpected zip layout: ${inner} not found after extraction."
        error "Contents of ${stage_dir}:"
        ls -la "${stage_dir}" >&2
        rm -rf "${stage_dir}"
        exit 1
    fi

    step "Installing → ${USDS_DIR}"
    mkdir -p "${USDS_DIR}"
    mv "${inner}"/* "${USDS_DIR}/"
    rm -rf "${stage_dir}"
    info "USDs installed to ${USDS_DIR}"
}

download_ckpt() {
    if [ "${FORCE}" = true ]; then rm -f "${CKPT_FILE}"; fi
    if [ -s "${CKPT_FILE}" ]; then
        info "X-Mobility checkpoint already present at ${CKPT_FILE} — skipping."
        return 0
    fi

    step "Downloading X-Mobility ckpt → ${CKPT_FILE}"
    # X-Mobility ckpt isn't always gated, but pass the token anyway — works in both cases.
    if [ -z "${HF_TOKEN_EFFECTIVE}" ]; then
        warn "HF_TOKEN not set; trying anonymous download (may fail if gated)."
    fi
    if [ -n "${HF_TOKEN_EFFECTIVE}" ]; then
        hf_download "${CKPT_URL}" "${CKPT_FILE}"
    else
        case "${DOWNLOADER}" in
            curl) curl -L --fail --progress-bar -o "${CKPT_FILE}" "${CKPT_URL}" ;;
            wget) wget --progress=bar:force -O "${CKPT_FILE}" "${CKPT_URL}" ;;
        esac
    fi
    if [ ! -s "${CKPT_FILE}" ]; then
        error "Downloaded ckpt is empty. Check your HF_TOKEN and network."
        rm -f "${CKPT_FILE}"
        exit 1
    fi
    info "Downloaded $(du -h "${CKPT_FILE}" | cut -f1) → ${CKPT_FILE}"
}

download_nurec_scenes() {
    if [ "${#NUREC_SCENES[@]}" -eq 0 ]; then
        return 0
    fi

    check_hf_token

    local scene scene_dir
    for scene in "${NUREC_SCENES[@]}"; do
        scene_dir="${USDS_DIR}/${scene}"
        if [ "${FORCE}" = true ]; then
            rm -rf "${scene_dir}"
        fi
        if [ -d "${scene_dir}" ] && [ -n "$(ls -A "${scene_dir}" 2>/dev/null)" ]; then
            info "NuRec scene ${scene} already installed at ${scene_dir} — skipping."
            continue
        fi

        step "Downloading NuRec scene ${scene} from ${NUREC_REPO}@${NUREC_REVISION} → ${USDS_DIR}"
        HF_TOKEN="${HF_TOKEN_EFFECTIVE}" hf download "${NUREC_REPO}" \
            --repo-type dataset \
            --revision "${NUREC_REVISION}" \
            --local-dir "${USDS_DIR}" \
            --include "${scene}/**" \
            --exclude "**/raw_images.zip"
        info "NuRec scene installed to ${scene_dir}"
    done
}

show_summary() {
    step "Asset summary"
    echo "Cache dir: ${CACHE_DIR}"
    if [ -d "${USDS_DIR}" ] && [ -n "$(ls -A "${USDS_DIR}" 2>/dev/null)" ]; then
        echo "  ✓ USDs:           ${USDS_DIR} ($(du -sh "${USDS_DIR}" | cut -f1))"
    else
        echo "  ✗ USDs:           ${USDS_DIR} (missing)"
    fi
    if [ -s "${CKPT_FILE}" ]; then
        echo "  ✓ X-Mobility ckpt: ${CKPT_FILE} ($(du -h "${CKPT_FILE}" | cut -f1))"
    else
        echo "  ✗ X-Mobility ckpt: ${CKPT_FILE} (missing)"
    fi
    if [ "${#NUREC_SCENES[@]}" -gt 0 ]; then
        echo "  NuRec revision:   ${NUREC_REPO}@${NUREC_REVISION}"
        local scene
        for scene in "${NUREC_SCENES[@]}"; do
            echo "  ✓ NuRec scene:     ${USDS_DIR}/${scene}"
        done
    fi
    echo ""
    echo "The repo is bind-mounted at /workspace/COMPASS inside the container."
    echo "USDs are picked up automatically by the mobility_es extension."
    echo "Pass the X-Mobility checkpoint via -b/--base-policy-path, e.g.:"
    echo "  -b ${CKPT_FILE#${REPO_ROOT}/}"
}

main() {
    info "COMPASS asset preparation"
    info "========================="
    check_downloader
    prepare_dirs
    download_usds
    download_ckpt
    download_nurec_scenes
    show_summary
    info "Done."
}

main "$@"
