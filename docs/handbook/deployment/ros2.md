# ROS2 Deployment

COMPASS ships ROS2 nodes that consume a TensorRT engine and emit velocity
commands. The same nodes cover three integration paths:

- **Isaac Sim** — driving a simulated robot end-to-end via the ROS2 bridge.
  The detailed setup is on the [Isaac Sim setup](isaac_sim.md) sub-page.
- **Sim2Real** — deploying the same `compass_inference` node directly on a
  real robot, optionally pairing with visual SLAM (e.g. cuVSLAM) for state
  estimation.
- **Object navigation** — wiring an object-localization module (e.g.
  Locate3D) into the bundled `obj_target_generator` node so the robot can
  approach named objects.

The packages live under
[`ros2_deployment/`](https://github.com/NVlabs/COMPASS/tree/main/ros2_deployment).

## Provided ROS2 nodes

- **`compass_inference`** — consumes camera images, target poses, and robot
  speed inputs, then outputs velocity commands by running TensorRT inference
  with COMPASS engines.
- **`obj_target_generator`** — receives object localization bounding boxes
  and generates navigation target poses for the COMPASS navigator.

These nodes enable a variety of integration workflows; the three below are
the bundled examples.

## 1. Isaac Sim integration

The [`compass_navigator`](https://github.com/NVlabs/COMPASS/tree/main/ros2_deployment/compass_navigator)
package is fully compatible with [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim),
leveraging its robotics environments and ROS2 bridge for seamless simulation.

For detailed setup instructions and a step-by-step Isaac Sim integration guide,
see the [Isaac Sim setup](isaac_sim.md) sub-page.

## 2. Zero-shot sim2real transfer

The `compass_inference` node can also be deployed directly on real robots,
enabling a seamless transition from simulation to real-world operation. By
integrating visual SLAM solutions such as
[cuVSLAM](https://nvidia-isaac-ros.github.io/concepts/visual_slam/cuvslam/index.html)
for robot state estimation, the COMPASS model can support zero-shot sim2real
transfer.

<https://github.com/user-attachments/assets/141af4f6-c915-4254-b6a0-00706e4aea5f>

## 3. Object navigation integration

By integrating an object localization module (e.g.,
[Locate3D](https://locate3d.atmeta.com/)), we can also enable object navigation
with COMPASS. The `obj_target_generator` node can convert localized object
bounding boxes into navigation goals, allowing the robot to autonomously
approach specified objects.

<https://github.com/user-attachments/assets/3928e9fd-f78d-4b8e-8bbc-9932b386ae6b>
