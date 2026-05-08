# Auto OMap from USDs

Occupancy maps gate collision-free pose sampling during training. Historically
they were hand-authored in the Isaac Sim editor — slow, error-prone, and a
blocker for [SAGE-driven](agentic.md) scene generation. The
[`scripts/generate_omap_from_usd.py`](https://github.com/NVlabs/COMPASS/blob/main/scripts/generate_omap_from_usd.py)
CLI generates an OMap directly from a USD scene in one shot.

## Generate an OMap

```bash
python scripts/generate_omap_from_usd.py \
    compass/rl_env/exts/mobility_es/mobility_es/usd/<scene>/<scene>.usd
```

(Inside an [activated dev shell](installation/docker.md), `python` already
points at Isaac Sim's bundled Python; on bare-metal install, prefix with
`${ISAACLAB_PATH}/isaaclab.sh -p`.)

By default the script writes the PNG + ROS YAML pair to `<usd_dir>/omap/`
next to the input USD — exactly where
[`OccupancyMapCollisionChecker`](https://github.com/NVlabs/COMPASS/blob/main/compass/rl_env/exts/mobility_es/mobility_es/utils/occupancy_map.py)
auto-discovers it when no explicit `OMAP_PATHS` entry exists.

## Common flags

| Flag | Default | Meaning |
|------|--------|---------|
| `-o, --out-dir` | `<usd_dir>/omap/` | Where the PNG + YAML land |
| `--cell-size` | `0.05` | meters / pixel |
| `--z-min` / `--z-max` | `0.1` / `0.62` | Height slab the rasterizer projects |
| `--bounds` | derived from USD bbox + 1 m padding | World-space `xmin xmax ymin ymax` |
| `--padding` | `1.0` | Padding (m) around the auto-derived bbox |
| `--map-name` | `<usd_basename>` | PNG stem |

Run `python scripts/generate_omap_from_usd.py --help` for the complete list.

## Loader auto-discovery

`OccupancyMapCollisionChecker` (`compass/rl_env/exts/.../utils/occupancy_map.py`)
checks `OMAP_PATHS` first; if it has no entry for the scene, it falls back to
`<dirname(spawn.usd_path)>/omap/occupancy_map.yaml` — the same path the
generator writes to. **No `OMAP_PATHS` registration is needed for new
scenes.**

## Verifying a generated OMap

A small standalone verifier in `/tmp/verify_omap.py` (used during PR-5
development) loads the generated YAML, samples a uniform grid of poses,
runs `is_in_collision`, and writes annotated PNGs:

- All 300 samples — green = free, red = collision (or out-of-bounds).
- Free-only — every dot lands on an unoccupied (white) cell.

The verification checked three representative scenes (`office`,
`combined_simple_warehouse`, `sample_small_footprint_one_rack_obst_sdg`);
all passed with free-pose samples cleanly avoiding obstacles.

## Slab tuning for taller embodiments

The default `[0.1, 0.62]` height slab works for Carter, Spot, G1. Tall
humanoids (H1) clear about 1.3 m; bump `--z-max 1.4` so the rasterizer
captures higher obstacles. The slab is the only embodiment-specific knob
and is exposed at the CLI rather than baked into per-embodiment maps.
