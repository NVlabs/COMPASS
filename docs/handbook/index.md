# COMPASS Handbook

```{raw} html
<div class="hero-lead">
<strong>COMPASS</strong> is a cross-embodiment mobility policy framework that
combines imitation learning, residual reinforcement learning, and policy
distillation to ship one generalist navigation policy across humanoids,
quadrupeds, and wheeled robots. This handbook is the live reference for
installing, training, deploying, and extending it.
</div>
<div class="workflow-pills">
  <span class="workflow-pill">Residual RL</span>
  <span class="workflow-pill">Policy Distillation</span>
  <span class="workflow-pill">Isaac Lab 3.0</span>
  <span class="workflow-pill">ROS2 deployment</span>
  <span class="workflow-pill">OSMO cloud</span>
  <span class="workflow-pill">VLA fine-tuning</span>
</div>
```

[**↗ Project page**](https://nvlabs.github.io/COMPASS/) ·
[arXiv](https://arxiv.org/abs/2502.16372) ·
[Source on GitHub](https://github.com/NVlabs/COMPASS)

---

## End-to-end pipeline

Each stage has a top-level entry-point script in the repo:

| # | Stage | Script |
|---|------|--------|
| 1 | Train residual RL specialist (or evaluate any policy) | [`run.py`](https://github.com/NVlabs/COMPASS/blob/main/run.py) |
| 2 | Roll out specialists to collect HDF5 distillation data | [`record.py`](https://github.com/NVlabs/COMPASS/blob/main/record.py) |
| 3 | Distil specialists into one generalist | [`distillation_train.py`](https://github.com/NVlabs/COMPASS/blob/main/distillation_train.py) |
| 4 | Export to ONNX / JIT | [`onnx_conversion.py`](https://github.com/NVlabs/COMPASS/blob/main/onnx_conversion.py) |
| 5 | Convert ONNX → TensorRT engine | [`trt_conversion.py`](https://github.com/NVlabs/COMPASS/blob/main/trt_conversion.py) |
| 6 | Deploy via ROS2 in Isaac Sim or on real robots | [`ros2_deployment/`](https://github.com/NVlabs/COMPASS/tree/main/ros2_deployment) |

Built on Isaac Lab 3.0 / Isaac Sim 4.5 and on top of NVIDIA's
[X-Mobility](https://github.com/NVlabs/X-MOBILITY) base policy.

---

## Where to start

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 🚀 Quick start
:link: quickstart
:link-type: doc

Three commands to a training shell.
:::

:::{grid-item-card} 📦 Installation
:link: installation/docker
:link-type: doc

Docker-as-venv: three commands and you're in a training shell.
:::

:::{grid-item-card} 🎓 Training a specialist
:link: workflows/training
:link-type: doc

How the residual RL loop works, how to override embodiment / scene.
:::

:::{grid-item-card} 🛰️ ROS2 Deployment
:link: deployment/ros2
:link-type: doc

ROS2 nodes consuming a TensorRT engine — Isaac Sim, sim2real, or
object navigation.
:::

:::{grid-item-card} ☁️ OSMO cloud submission
:link: osmo
:link-type: doc

Submit training and evaluation runs to NVIDIA's OSMO cluster.
:::

:::{grid-item-card} 🤖 GR00T post-training
:link: workflows/gr00t_finetuning
:link-type: doc

Use COMPASS distillation datasets to fine-tune NVIDIA's
[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) VLA model.
:::

::::

---

## What's where in the repo

```text
COMPASS/
├── compass/
│   ├── residual_rl/         # Actor-critic that adds a residual on top of X-Mobility
│   ├── distillation/        # Generalist distillation (PyTorch Lightning trainer)
│   ├── rl_env/              # Isaac Lab `mobility_es` extension (embodiments + scenes)
│   └── utils/               # Shared helpers
├── configs/                 # Gin configs (train / eval / record / distillation / shared)
├── docker/                  # Docker-as-venv dev environment + Dockerfiles
├── osmo/                    # OSMO workflow YAMLs + Python launcher
├── ros2_deployment/         # ROS2 packages for Isaac Sim / real-robot deployment
├── scripts/                 # Standalone tools (HDF5→LeRobot, omap generator, …)
└── x_mobility/              # Vendored X-Mobility wheel
```

For deeper layout details, see [CLAUDE.md](https://github.com/NVlabs/COMPASS/blob/main/CLAUDE.md).

```{toctree}
:caption: Installation
:maxdepth: 1
:hidden:

quickstart
installation/docker
agentic
extending
```

```{toctree}
:caption: Workflows
:maxdepth: 1
:hidden:

workflows/training
workflows/recording
workflows/distillation
workflows/export
workflows/gr00t_finetuning
```

```{toctree}
:caption: ROS2 Deployment
:maxdepth: 1
:hidden:

deployment/ros2
deployment/isaac_sim
```

```{toctree}
:caption: Reference
:maxdepth: 1
:hidden:

osmo
omap
contributing
```
