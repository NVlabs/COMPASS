# Recording distillation data

`record.py` rolls out trained specialists across embodiments / scenes and
writes the trajectories as HDF5 files. The resulting dataset feeds the
[generalist distillation step](distillation.md).

## Run a recording

1. **Configure which specialists to record from**: edit
   [`configs/distillation_dataset_config_template.yaml`](https://github.com/NVlabs/COMPASS/blob/main/configs/distillation_dataset_config_template.yaml)
   to point at your per-embodiment specialist checkpoints and split ratios.
   Save it as `configs/distillation_dataset_config.yaml` (or anywhere; pass
   the path on the command line).

2. **Roll out**:

   ```bash
   python record.py \
       -c configs/distillation_dataset_config.yaml \
       -o <output_dir> \
       -b <path/to/x_mobility_ckpt> \
       --dataset-name <dataset_name>
   ```

The dataset name appears in OSMO references and in the HDF5 file metadata. It
is the same identifier used by [`scripts/hdf5_to_lerobot_episodic.py`](https://github.com/NVlabs/COMPASS/blob/main/scripts/hdf5_to_lerobot_episodic.py)
for [GR00T post-training](gr00t_finetuning.md).

## Output layout

`record.py` writes one HDF5 file per shard, plus a manifest, under
`<output_dir>/`. Each HDF5 file contains episodic data with the standard
COMPASS observation keys (camera, lidar, proprio, …) and the action emitted
by the specialist. The shape convention is `[T, S, ...]` (timesteps × shards).

## On OSMO

The X-Mobility base ckpt and COMPASS USDs are downloaded inside the workflow
from HuggingFace, so you don't need to pass the base-policy checkpoint or
USDs locally.

```bash
export COMPASS_OSMO_REGISTRY=nvcr.io/<org>/<team>
export WANDB_API_KEY=...
export HF_TOKEN=...

python osmo/run_osmo.py record \
    --experiment-name <name> \
    --dataset-name <hdf5-dataset-name> \
    [--image <pre-built>]
```

Full reference: [OSMO cloud submission](../osmo.md).
