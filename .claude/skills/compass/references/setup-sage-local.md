# Setup SAGE (local install)

Most COMPASS users do **not** need this. The default flow uses
`nvidia/SAGE-10k` (10,000 pre-generated indoor scenes) via the bundled
`scripts/sage10k_search.py` and `scripts/sage10k_to_usd.py` — no SAGE
install required, no extra deps beyond Isaac Sim.

You only need a local SAGE install when you want to **generate fully
custom scenes** that aren't in SAGE-10k — e.g., novel layouts, custom
object distributions, or VLM-driven asset placement.

## Why this is gated behind a reference file

- Heavy deps: pytorch3d, TRELLIS, VLM servers (Llava / similar).
- Multi-GPU recommended for the generation pipeline (the VLM and the
  3D-asset generator both want a GPU).
- Cold install can take 30–60 minutes and conflicts with parts of the
  COMPASS env (different torch/cuda combinations).

For day-to-day SAGE-driven training, prefer the SAGE-10k flow in the
main `SKILL.md` (the "Search SAGE-10k" section).

## How to install

Follow the canonical instructions in the SAGE repo: <https://github.com/NVlabs/sage>.

The repo README is the source of truth — version-pinning, environment
setup, and dataset paths change there faster than this reference can
keep up. Cross-link from there back to COMPASS once SAGE is installed
and you have a working `sage` CLI.

## After SAGE is installed

The `compass` skill's SAGE-10k flow remains the recommended path for
most scenes. If you specifically want SAGE to generate a novel layout:

1. Run SAGE's generator to produce a layout (PLY meshes + `layout.json`).
2. Convert with the same `scripts/sage10k_to_usd.py` the SAGE-10k flow
   uses — the converter is layout-source-agnostic.
3. Continue with **Register Scene** → **Train** in the main `SKILL.md`.
