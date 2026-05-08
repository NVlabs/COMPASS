# Exporting to ONNX / TensorRT

Two scripts cover the export path. Both are pure-PyTorch (no Isaac Lab
required).

## ONNX / JIT

### Specialist (residual + base)

```bash
python3 onnx_conversion.py \
    -b <path/to/x_mobility_ckpt> \
    -r <path/to/residual_policy_ckpt> \
    -o <path/to/output_onnx_file> \
    -j <path/to/output_jit_file>
```

### Generalist

```bash
python3 onnx_conversion.py \
    -b <path/to/x_mobility_ckpt> \
    -g <path/to/generalist_policy_ckpt> \
    -e <embodiment_type> \
    -o <path/to/output_onnx_file> \
    -j <path/to/output_jit_file>
```

`-e` selects which embodiment-specific head to export. Pass one of
`{h1, carter, spot, g1, digit}` — must match an entry in `EmbodimentEnvCfgMap`
in [`run.py`](https://github.com/NVlabs/COMPASS/blob/main/run.py).

## TensorRT

```bash
python3 trt_conversion.py \
    -o <path/to/onnx_file> \
    -t <path/to/trt_engine_file>
```

Generates a TensorRT engine specialised for your GPU. The engine is then
consumed by the [ROS2 deployment node](../deployment/ros2.md).

:::{note} Engine portability
TensorRT engines are GPU-specific. An engine built on an L40 won't run
optimally (or at all) on a Jetson Orin. Build the engine on the same GPU
family as the deployment target, or use the
[ROS2 deployment](../deployment/ros2.md) container's runtime conversion
flow.
:::
