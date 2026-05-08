# OSMO cloud submission

Submit COMPASS training, evaluation, recording, or distillation jobs to
NVIDIA's [OSMO](https://docs.nvidia.com/osmo/) cluster via the bundled
Python launcher [`osmo/run_osmo.py`](https://github.com/NVlabs/COMPASS/blob/main/osmo/run_osmo.py).

## Prerequisites

- An OSMO account with a logged-in CLI: `osmo login`
- Write access to a docker registry that OSMO workers can pull from (set as `--registry-prefix` or via the `COMPASS_OSMO_REGISTRY` env var, e.g. `nvcr.io/<org>/<team>`)
- A wandb account; export `WANDB_API_KEY` (or pass `--prompt`)
- A HuggingFace token with read access to gated assets if your run needs them; export `HF_TOKEN` (or pass `--prompt`). Distillation does not need this.
- An OSMO dataset uploaded with the unzipped contents of `compass_usds.zip` named `groot_mobility_rl_es_usds`. The RL workflows mount this dataset and copy the USDs into `compass/rl_env/exts/mobility_es/mobility_es/usd` at startup.
- The base-policy and any resume / residual / distillation checkpoints uploaded as wandb artifacts that the workflow can `wandb artifact get`.

## Quick start

```bash
# One-time, in your shell
export WANDB_API_KEY=<your-wandb-key>
export HF_TOKEN=<your-hf-token>
export COMPASS_OSMO_REGISTRY=nvcr.io/<org>/<team>

# Submit residual RL training; build+push the image automatically.
python osmo/run_osmo.py train \
    --experiment-name pilot \
    --wandb-project compass_train \
    --base-policy-ckpt my-wandb-entity/my-project/x_mobility:v1
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

### `train`

```bash
python osmo/run_osmo.py train \
    --experiment-name <name> \
    --wandb-project <wandb-project> \
    --base-policy-ckpt <wandb-artifact> \
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
    --base-policy-ckpt <wandb-artifact> \
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
    --base-policy-ckpt <wandb-artifact> \
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
- **Dataset not found** — confirm the OSMO dataset `groot_mobility_rl_es_usds` (RL workflows) or your distillation dataset is uploaded and visible to the workflow's run-as account.
