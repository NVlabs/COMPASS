#!/usr/bin/env bash
# compass_status.sh — diagnostic snapshot of a COMPASS dev environment.
#
# Usage:
#   ./compass_status.sh                # quick checks (~1s)
#   ./compass_status.sh --deep         # also runs Isaac Sim init test (~30s)
#   ./compass_status.sh --ckpt PATH    # also load a specific .pt file with torch
#
# Exit code: 0 if all required checks pass, 1 if any fail.
# WARN-level entries don't fail the run.

set -uo pipefail

DEEP=0
CKPT_PATH=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --deep) DEEP=1; shift ;;
        --ckpt) CKPT_PATH="${2:-}"; shift 2 ;;
        -h|--help)
            sed -n '4,9p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

# Resolve repo root (script lives at .claude/skills/compass-doctor/scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

PASS="✓"
FAIL="✗"
WARN="⚠"

declare -a TABLE
fail_count=0

row() {
    local status="$1" name="$2" detail="$3"
    TABLE+=("| ${status} | ${name} | ${detail} |")
    [ "${status}" = "${FAIL}" ] && fail_count=$((fail_count + 1))
}

# 1. Container running?
if ./docker/run.sh status 2>/dev/null | grep -qE "Up|running"; then
    row "${PASS}" "Container" "compass-rl up"
else
    row "${FAIL}" "Container" "down — run \`./docker/run.sh up\`"
fi

# 2. Activated shell? Detect by PATH containing the shim dir created by
# docker/activate (mktemp pattern: compass-shims.XXXXXX).
if echo "${PATH}" | grep -q "compass-shims\."; then
    row "${PASS}" "Activated shell" "shim dir on PATH"
else
    row "${WARN}" "Activated shell" "no — run \`source ./docker/activate\`"
fi

# 3. GPU
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_INFO="$(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null | head -1)"
    if [ -n "${GPU_INFO}" ]; then
        row "${PASS}" "GPU" "${GPU_INFO}"
    else
        row "${FAIL}" "GPU" "nvidia-smi returned empty (driver issue?)"
    fi
else
    row "${FAIL}" "GPU" "nvidia-smi not on PATH"
fi

# 4. Base policy ckpt
if [ -f "./assets/x_mobility.ckpt" ]; then
    SIZE="$(du -h ./assets/x_mobility.ckpt | cut -f1)"
    row "${PASS}" "Base ckpt" "./assets/x_mobility.ckpt (${SIZE})"
else
    row "${FAIL}" "Base ckpt" "missing — run \`./docker/run.sh assets\`"
fi

# 5. Built-in scene USDs
if [ -d "./assets/usd" ] && [ -n "$(ls -A ./assets/usd 2>/dev/null)" ]; then
    USD_COUNT="$(ls ./assets/usd | wc -l)"
    row "${PASS}" "USDs" "${USD_COUNT} entries in ./assets/usd/"
else
    row "${FAIL}" "USDs" "missing — run \`./docker/run.sh assets\`"
fi

# 6. Recent training log (informational; non-blocking)
LATEST="$(ls -t /tmp/isaaclab/logs/ 2>/dev/null | head -1 || true)"
if [ -n "${LATEST}" ]; then
    row "${PASS}" "Recent log" "/tmp/isaaclab/logs/${LATEST}"
else
    row "${WARN}" "Recent log" "none — no training has run yet on this machine"
fi

# 7. Deep check: Isaac Sim init (only with --deep)
if [ "${DEEP}" = "1" ]; then
    if python -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
app.close()
" >/dev/null 2>&1; then
        row "${PASS}" "Isaac Sim init" "headless app start/close OK"
    else
        row "${FAIL}" "Isaac Sim init" "FAILED — check container build / GPU access"
    fi
fi

# 8. Optional: load a specific ckpt
if [ -n "${CKPT_PATH}" ]; then
    if [ -f "${CKPT_PATH}" ] && python -c "
import torch
torch.load('${CKPT_PATH}', map_location='cpu')
" >/dev/null 2>&1; then
        SIZE="$(du -h "${CKPT_PATH}" | cut -f1)"
        row "${PASS}" "Ckpt load" "${CKPT_PATH} (${SIZE})"
    else
        row "${FAIL}" "Ckpt load" "${CKPT_PATH} (missing or not a valid torch ckpt)"
    fi
fi

# Print table
echo "| Status | Check | Detail |"
echo "|---|---|---|"
for line in "${TABLE[@]}"; do
    echo "${line}"
done

if [ "${fail_count}" -eq 0 ]; then
    echo ""
    echo "All checks passed."
    exit 0
else
    echo ""
    echo "${fail_count} check(s) failed. See https://nvlabs.github.io/COMPASS/docs/quickstart.html for setup."
    exit 1
fi
