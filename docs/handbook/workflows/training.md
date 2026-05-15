# Training residual RL specialists

`run.py` trains a residual RL policy on top of the X-Mobility base policy for a
given embodiment + scene. The residual head adapts X-Mobility's
language-conditioned navigation behaviour to embodiment-specific dynamics.

## Default training run

```bash
python run.py \
    -c configs/train_config.gin \
    -o <output_dir> \
    -b <path/to/x_mobility_ckpt> \
    --enable_cameras
```

(Inside an [activated dev shell](../installation/docker.md), `python` already
points at Isaac Sim's bundled Python; on a bare-metal install, prefix with
`${ISAACLAB_PATH}/isaaclab.sh -p`.)

## Evaluating a trained specialist

```bash
python run.py \
    -c configs/eval_config.gin \
    -o <output_dir> \
    -b <path/to/x_mobility_ckpt> \
    -p <path/to/residual_policy_ckpt> \
    --enable_cameras \
    --video \
    --video_interval <video_interval>
```

## GPU memory and `--num_envs`

GPU memory scales linearly with the parallel-env count: **32 envs ≈ 30 GB**.
Drop `--num_envs` to fit the host. `--num_envs 1` is the canonical smoke-test
setting and reaches PPO iteration 0 in a few minutes.

## Picking embodiment / scene

Override the gin defaults via CLI:

```bash
python run.py -c configs/train_config.gin \
    --embodiment {h1,carter,spot,g1,digit} \
    --environment <scene_name> \
    -o <output_dir> -b <ckpt>
```

The current set lives in `EmbodimentEnvCfgMap` and `EnvSceneAssetCfgMap` in
[`run.py`](https://github.com/NVlabs/COMPASS/blob/main/run.py). To register a
new one, see [Adding a new embodiment / scene](../extending.md).

## Logging

TensorBoard at `<output_dir>/tensorboard/` by default. For Weights & Biases
add:

```bash
--logger wandb \
--wandb-project-name <name> \
--wandb-run-name <name> \
--wandb-entity-name <entity>
```

## Multi-GPU training

Pass `--distributed` and launch under `torch.distributed.run` to fan out across
GPUs. Each rank runs its own Isaac Sim instance + env; gradients, KL, and
metrics sync via manual all-reduce in `PPO.update`. Rank 0 owns the logger,
checkpoints, video, and episode-log writes.

```bash
${ISAACLAB_PATH}/isaaclab.sh -p -m torch.distributed.run \
    --nproc_per_node=8 \
    run.py --distributed \
    -c configs/train_config.gin \
    -o <output_dir> -b <x_mobility_ckpt> \
    --enable_cameras --num_envs 32
```

Total parallel envs = `nproc_per_node × num_envs`. The trainer's distributed
code paths are `world_size`-aware, so `--nproc_per_node=1` (or just plain
`python run.py --distributed`) is also a valid single-rank fallback. On OSMO,
`osmo/run_osmo.py train --num-gpus {2,8}` routes to the matching
distributed-workflow YAML.

## Submitting to OSMO

Train on the OSMO cluster instead of locally. The X-Mobility base ckpt and
COMPASS USDs are downloaded inside the workflow from HuggingFace, so you
don't need to pass the base-policy checkpoint or USDs locally.

`osmo/run_osmo.py` is **host-side** (it shells out to `docker` and `osmo` CLIs).
The activate shim auto-routes it to host Python via the `# COMPASS_HOST_SIDE`
marker, so plain `python osmo/run_osmo.py …` works from the activated shell —
see the host-side callout in [OSMO cloud submission](../osmo.md) for the full
explanation and fallbacks.

```bash
export COMPASS_OSMO_REGISTRY=nvcr.io/<org>/<team>
export WANDB_API_KEY=...
export HF_TOKEN=...

python osmo/run_osmo.py train \
    --experiment-name <name> \
    --wandb-project <wandb-project> \
    [--num-gpus 8] \
    [--image <pre-built>]
```

Full reference: [OSMO cloud submission](../osmo.md).
