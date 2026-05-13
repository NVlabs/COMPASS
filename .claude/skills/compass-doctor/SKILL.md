---
name: compass-doctor
description: >
  Diagnose why COMPASS isn't working: container, GPU, activated shell,
  assets, Isaac Sim init, checkpoint validity. Use whenever the user
  reports vague COMPASS errors — "training won't start", "something's
  wrong", "why is this failing" — even without the word "debug". Make
  sure to use this when the user mentions COMPASS isn't behaving and
  doesn't have a more specific intent.
allowed-tools:
  - Bash
  - Read
  - Grep
  - Glob
---

You produce a diagnostic snapshot of a COMPASS dev environment and report what's working and what's broken. Read-only by design — you report problems but never auto-fix them. Once you've identified the root cause, point the user at the specialty skill that owns the fix.

This skill is the place users land when training "just won't start" or something obscure breaks. The goal is to surface the actual root cause in one screen, not to chase the symptom.

## When NOT to use this skill

- The user knows what they want to do (train, evaluate, etc.) and just needs the workflow → `/compass`.
- The user wants to add a robot platform → `/compass-newembodiment`.
- A specific error message has a clear fix in the handbook → quote the relevant page; debug isn't needed.

---

## Workflow

### Step 1: Run the diagnostic script

The skill bundles `scripts/compass_status.sh` which runs all checks in parallel and prints a markdown table:

```bash
# dangerouslyDisableSandbox: true (script touches nvidia-smi)
bash <SKILL_BASE_DIR>/scripts/compass_status.sh
```

Output looks like:

```
| Status | Check | Detail |
|---|---|---|
| ✓ | Container | compass-rl up |
| ✓ | Activated shell | shim dir on PATH |
| ✓ | GPU | NVIDIA H100 80GB HBM3, 79980 MiB |
| ✓ | Base ckpt | ./assets/x_mobility.ckpt (1.2G) |
| ✓ | USDs | 8 entries in ./assets/usd/ |
| ✓ | Recent log | /tmp/isaaclab/logs/2026-05-08_14-23-02 |
```

For deeper diagnostics (slower, ~30s — runs Isaac Sim init headless to surface kit-init errors that don't show up in the quick check):

```bash
bash <SKILL_BASE_DIR>/scripts/compass_status.sh --deep
```

For checkpoint-specific issues (verifies a `.pt` file is a valid torch checkpoint and not corrupted):

```bash
bash <SKILL_BASE_DIR>/scripts/compass_status.sh --ckpt <PATH_TO_CKPT>
```

### Step 2: Interpret the output

The table is the report; just relay it to the user. Then add a one-paragraph interpretation: which row is the root cause, and what to do about it.

| Failed check | Most likely cause | Where to fix |
|---|---|---|
| Container | Container hasn't been started | `./docker/run.sh up` (or `./docker/run.sh build` if image missing) |
| Activated shell | User invoked Claude in a fresh shell | `source ./docker/activate` then retry |
| GPU | Driver issue OR sandbox blocking nvidia-smi | Check `dangerouslyDisableSandbox: true` was set; if persists, check host driver with `nvidia-smi` from the host shell |
| Base ckpt / USDs | Assets not downloaded yet | `./docker/run.sh assets` (needs `HF_TOKEN`) |
| Recent log | No training has run yet (informational) | Not a failure — just confirms a clean state |
| Isaac Sim init | Container build broken OR GPU not exposed | Try `./docker/run.sh down && ./docker/run.sh build` |
| Ckpt load | File corrupt OR wrong torch version | Re-download / re-train; if re-train, route to `/compass` |

### Step 3: Recommend the next skill

Don't try to fix the issue yourself — different fixes belong to different skills:

| Root cause class | Recommended next skill |
|---|---|
| Setup issue (container, assets, activated shell) | `/compass` (Setup COMPASS section) |
| Training-time issue (config, ckpt path, env name) | `/compass` (Train section) |
| New robot platform missing | `/compass-newembodiment` |

Anti-pattern guard: don't run fixes yourself. The user benefits from understanding what went wrong; running fixes blind hides root causes.

---

## Common patterns

### "Training crashed silently"

Look for the most-recent kit log:
```bash
find ~/.local/share/ov/pkg/isaac-sim-* -name "kit_*.log" 2>/dev/null | head -1 | xargs tail -100
```

Common silent-crash causes:
- GPU OOM (look for "out of memory" / "CUDA error" near the tail).
- USD asset missing (look for "Failed to load" / file-path lines).
- Quaternion convention mismatch on a custom embodiment (look for "wxyz" / "xyzw").

### "Pre-flight passes but `python run.py …` still errors"

Confirm the user is in an activated shell. The script checks PATH for the shim dir but a user can run scripts in any shell — verify with:
```bash
which python
# Should resolve to a /tmp/compass-shims.* path
```

If `which python` resolves to `/usr/bin/python` or similar, the shell isn't activated even if other env vars suggest it is.

### "OSMO submission fails on `osmo workflow submit`"

That's an OSMO-side issue, not a local one. Check:
```bash
test -n "${WANDB_API_KEY:-}" && echo "WANDB_API_KEY set" || echo "WANDB_API_KEY MISSING"
test -n "${HF_TOKEN:-}" && echo "HF_TOKEN set" || echo "HF_TOKEN MISSING"
test -n "${COMPASS_OSMO_REGISTRY:-}" && echo "registry set" || echo "registry MISSING (or pass --image)"
```

If all set, route the user to `/compass` (OSMO submission section in that skill body).

---

## Key File Locations

| File | Purpose |
|------|---------|
| `<SKILL_BASE_DIR>/scripts/compass_status.sh` | The diagnostic script; usable as a standalone CLI |
| `./docker/run.sh` | Container lifecycle (status / up / down / build / assets / shell) |
| `./docker/activate` | Shell activate script (sets up python/pip shims) |
| `./assets/` | Asset bind-mount (USDs + base ckpt) |
| `/tmp/isaaclab/logs/` | Training run logs (latest is most useful) |
| `~/.local/share/ov/pkg/isaac-sim-*/kit_*.log` | Kit logs for crash diagnostics |
