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

## Submitting to OSMO

Train on the OSMO cluster instead of locally:

```bash
python osmo/run_osmo.py train \
    --experiment-name <name> \
    --image <pre-built-image-or-empty-to-build> \
    --base-policy-ckpt <wandb-artifact-path>
```

Full reference: [OSMO cloud submission](../osmo.md).
