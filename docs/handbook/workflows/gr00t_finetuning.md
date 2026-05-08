# GR00T post-training (VLA fine-tuning)

COMPASS distillation datasets can fine-tune VLA models like
[NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) to bolt on
navigation capabilities. Three steps: convert, fine-tune, evaluate.

## Step 1 — Convert HDF5 → GR00T LeRobot format

You need an HDF5 distillation dataset first; see
[Recording distillation data](recording.md). Then convert with the bundled
script:

```bash
python scripts/hdf5_to_lerobot_episodic.py \
    --hdf5-dir <path/to/hdf5/directory> \
    --output-path <path/to/lerobot/format>
```

The script is pure-Python (no Isaac Lab) — runs anywhere with the standard
COMPASS Python environment. It walks the HDF5 dataset, repacks per-episode
into LeRobot's parquet format, and emits the chunk + metadata layout that
GR00T's training pipeline expects.

## Step 2 — Post-train GR00T

Follow the post-training instructions in the
[Isaac-GR00T getting-started guide](https://github.com/NVIDIA/Isaac-GR00T/tree/main/getting_started).

A ready-to-use navigation data configuration lives on this branch:
[`liuw/nav_fine_tune`](https://github.com/NVIDIA/Isaac-GR00T/compare/main...liuw/nav_fine_tune).

## Step 3 — Evaluate the post-trained GR00T model in COMPASS

Launch the GR00T inference server (see the Isaac-GR00T repo) on
**port 8888**, with the same data configuration you used during training.
Then evaluate from COMPASS:

```bash
python run.py \
    -c configs/eval_config.gin \
    -o <output_dir> \
    -b <path/to/x_mobility_ckpt> \
    --enable_cameras \
    --gr00t-policy
```

`--gr00t-policy` tells `run.py` to dispatch action queries to the inference
server instead of loading a local checkpoint. Eval parameters (scene,
embodiment, episode count) live in
[`configs/eval_config.gin`](https://github.com/NVlabs/COMPASS/blob/main/configs/eval_config.gin).
