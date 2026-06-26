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

## Step 3 — Evaluate the post-trained GR00T model in COMPASS (closed loop)

> Requires the [`liuw/gr00t-n16-eval`](https://github.com/NVlabs/COMPASS/tree/liuw/gr00t-n16-eval)
> branch (GR00T N1.6 inference-protocol + 480×640 camera fixes) — `git checkout` it first.

Eval runs two processes over ZeroMQ **port 8888**: the GR00T inference server
(serves the fine-tuned policy) and the COMPASS sim (queries it each step).

**1. Serve the checkpoint** (in the Isaac-GR00T repo):

```bash
python gr00t/eval/run_gr00t_server.py \
    --model-path <path/to/checkpoint> \
    --embodiment-tag NEW_EMBODIMENT \
    --device cuda:0 --host 0.0.0.0 --port 8888
```

**2. Run the closed-loop eval** (in COMPASS):

```bash
python run.py -c configs/eval_config.gin --enable_cameras --gr00t-policy \
    -b ./assets/x_mobility.ckpt -o /tmp/gr00t_eval \
    --embodiment g1 --environment combined_single_rack --num_envs 10
```

`--gr00t-policy` queries the server at `0.0.0.0:8888` instead of loading a local
checkpoint (no `-p` needed); the success rate is reported as
`eval/goal_reached_rate`. Add `--viz kit --num_envs 1` to watch one robot in the
viewer.
