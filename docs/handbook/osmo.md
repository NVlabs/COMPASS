# OSMO cloud submission

Submit COMPASS training, evaluation, recording, or distillation jobs to
NVIDIA's [OSMO](https://docs.nvidia.com/osmo/) cluster via the bundled
Python launcher [`osmo/run_osmo.py`](https://github.com/NVlabs/COMPASS/blob/main/osmo/run_osmo.py).

## Prerequisites

- An OSMO account with a logged-in CLI: `osmo login`
- Write access to a docker registry that OSMO workers can pull from (set as `--registry-prefix` or via the `COMPASS_OSMO_REGISTRY` env var, e.g. `nvcr.io/<org>/<team>`)
- A wandb account; export `WANDB_API_KEY` (or pass `--prompt`)
- A HuggingFace token with read access to [`nvidia/COMPASS`](https://huggingface.co/nvidia/COMPASS) (gated) and [`nvidia/X-Mobility`](https://huggingface.co/nvidia/X-Mobility); export `HF_TOKEN` (or pass `--prompt`). Distillation does not need this.
- Any resume / residual / distillation checkpoints you reference uploaded as wandb artifacts that the workflow can `wandb artifact get`.

The COMPASS USDs (`compass_usds.zip`) and the X-Mobility base-policy
checkpoint are downloaded inside the workflow directly from HuggingFace
on every run — no OSMO-dataset or wandb-artifact mirroring required.

## Quick start

```bash
# One-time, in your shell
export WANDB_API_KEY=<your-wandb-key>
export HF_TOKEN=<your-hf-token>
export COMPASS_OSMO_REGISTRY=nvcr.io/<org>/<team>

# Submit residual RL training; build+push the image automatically.
python osmo/run_osmo.py train \
    --experiment-name pilot \
    --wandb-project compass-rl
```

To inspect the would-be `osmo workflow submit` invocation without actually
submitting, add `--dry-run`.

## Subcommands

| Subcommand | Workflow YAML | Purpose |
|------------|---------------|---------|
| `train` | [`osmo/workflows/rl_es_train_workflow.yaml`](https://github.com/NVlabs/COMPASS/blob/main/osmo/workflows/rl_es_train_workflow.yaml) | Residual RL specialist training (auto-runs an eval at the end) |
| `eval` | [`osmo/workflows/rl_es_eval_workflow.yaml`](https://github.com/NVlabs/COMPASS/blob/main/osmo/workflows/rl_es_eval_workflow.yaml) | Residual RL specialist or generalist evaluation |
| `record` | [`osmo/workflows/rl_es_record_workflow.yaml`](https://github.com/NVlabs/COMPASS/blob/main/osmo/workflows/rl_es_record_workflow.yaml) | Roll out a specialist policy to collect HDF5 data for distillation |
| `distill` | [`osmo/workflows/distillation_train_workflow.yaml`](https://github.com/NVlabs/COMPASS/blob/main/osmo/workflows/distillation_train_workflow.yaml) | Train the generalist distillation policy |

A separate launcher [`osmo/run_benchmark.py`](https://github.com/NVlabs/COMPASS/blob/main/osmo/run_benchmark.py) reuses the eval workflow above to fire a no-regression benchmark sweep across scenes for a given embodiment — see [Benchmark](#benchmark) below.

### `train`

```bash
python osmo/run_osmo.py train \
    --experiment-name <name> \
    --wandb-project <wandb-project> \
    [--resume-ckpt <wandb-artifact>] \
    [--no-residual] \
    [--image <pre-built-image>]
```

### `eval`

```bash
python osmo/run_osmo.py eval \
    --experiment-name <name> \
    --wandb-project <wandb-project> \
    --checkpoint <residual-wandb-artifact> \
    [--distillation-ckpt <wandb-artifact>] \
    [--no-residual] \
    [--embodiment h1|spot|carter|g1|digit] \
    [--environment <scene-name>] \
    [--image <pre-built-image>]
```

### `record`

```bash
python osmo/run_osmo.py record \
    --experiment-name <name> \
    --dataset-name <output-osmo-dataset> \
    [--image <pre-built-image>]
```

### `distill`

```bash
python osmo/run_osmo.py distill \
    --experiment-name <name> \
    --wandb-project <wandb-project> \
    --dataset-name <input-osmo-dataset> \
    [--checkpoint <wandb-artifact>] \
    [--train-config distillation_config] \
    [--image <pre-built-image>]
```

## Benchmark

Use [`osmo/run_benchmark.py`](https://github.com/NVlabs/COMPASS/blob/main/osmo/run_benchmark.py) to sweep a checkpoint across multiple scenes for one embodiment in a single invocation. Each `--environments` entry fires one `rl_es_eval_workflow.yaml` submission; results land in W&B under `bm_<embodiment>_<environment>_<experiment-name>` with the usual `eval/goal_reached_rate`, `eval/fall_down_rate`, `eval/total_travel_time`, and `eval/weighted_travel_time` metrics. Re-run with different `--embodiment` values for full matrix coverage.

```bash
python osmo/run_benchmark.py \
    --experiment-name <name> \
    --wandb-project-name <wandb-project> \
    --checkpoint-artifact <residual-wandb-artifact> \
    [--distillation-ckpt-artifact <wandb-artifact>] \
    [--embodiment h1|spot|carter|g1|digit] \
    [--environments simple_office warehouse_single_rack ...] \
    [--image-name <pre-built-image>]
```

Default sweep covers `simple_office`, `warehouse_single_rack`, `warehouse_multi_rack`, `combined_single_rack`, `combined_multi_rack` with `--embodiment h1`. Prerequisites and `--registry-prefix` / `--image-name` / `--dry-run` / `--prompt` behavior are the same as `run_osmo.py`.

## Image build behavior

If you do not pass `--image`, `run_osmo.py` will:

1. Build the appropriate Dockerfile ([`docker/Dockerfile.rl`](https://github.com/NVlabs/COMPASS/blob/main/docker/Dockerfile.rl) for `train` / `eval` / `record`, [`docker/Dockerfile.distillation`](https://github.com/NVlabs/COMPASS/blob/main/docker/Dockerfile.distillation) for `distill`) tagged as `<registry-prefix>/compass_<experiment-name>:<short-uuid>`.
2. `docker push` that tag.
3. Pass the resulting image into the `osmo workflow submit` command.

For repeated runs against the same code, build the image once and pass it via
`--image` to skip the build/push.

## Troubleshooting

- **`ERROR: $WANDB_API_KEY is not set`** — export it, or pass `--prompt` to be asked interactively.
- **`ERROR: --image not given and --registry-prefix is empty`** — either supply `--image <pre-built>` or set `--registry-prefix` / `$COMPASS_OSMO_REGISTRY`.
- **`manifest unknown` from `osmo workflow submit`** — the image hasn't pushed yet (or you don't have read access). Re-run `docker push <image>` and retry.
- **Workflow logs** — visit your OSMO console; the workflow ID printed by `osmo workflow submit` is the lookup key.
- **HuggingFace download fails inside the workflow** — verify `HF_TOKEN` was exported and that your account has access to [`nvidia/COMPASS`](https://huggingface.co/nvidia/COMPASS) and [`nvidia/X-Mobility`](https://huggingface.co/nvidia/X-Mobility).
- **Dataset not found** (distillation only) — confirm your distillation input dataset is uploaded and visible to the workflow's run-as account.
