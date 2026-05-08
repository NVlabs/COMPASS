# Distilling the generalist

Once you have an HDF5 distillation dataset (see [Recording distillation data](recording.md)),
`distillation_train.py` trains a single generalist policy that imitates the
per-embodiment specialists. PyTorch Lightning under the hood; pure-PyTorch
script (no Isaac-Lab Python wrapper required).

## Train a generalist

```bash
python3 distillation_train.py \
    --config-files configs/distillation_config.gin \
    --dataset-path <path/to/specialists_dataset> \
    --output-dir <output_dir>
```

Hyperparameters live in
[`configs/distillation_config.gin`](https://github.com/NVlabs/COMPASS/blob/main/configs/distillation_config.gin).
Override on the CLI by repeating `--config-files` or via gin
`--gin_bindings` overrides if you need to tweak a single value.

## Evaluate the generalist

The same `run.py` driver evaluates a generalist checkpoint when given via `-d`:

```bash
python run.py \
    -c configs/eval_config.gin \
    -o <output_dir> \
    -b <path/to/x_mobility_ckpt> \
    -d <path/to/generalist_policy_ckpt> \
    --enable_cameras \
    --video \
    --video_interval <video_interval>
```

(Inside an [activated dev shell](../installation/docker.md), `python` is
already wired; on bare-metal install, prefix with `${ISAACLAB_PATH}/isaaclab.sh -p`.)

## Submitting to OSMO

```bash
python osmo/run_osmo.py distill \
    --experiment-name <name> \
    --dataset-name <hdf5-dataset-name>
```

Full reference: [OSMO cloud submission](../osmo.md).

## Where to go next

- [Export](export.md) the generalist to ONNX / TensorRT for inference.
- [GR00T post-training](gr00t_finetuning.md) — use the same dataset to
  fine-tune the Isaac-GR00T VLA model.
