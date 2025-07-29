# COMPASS ROS2 Deployment

This directory contains ROS2 packages for deploying **COMPASS** across both simulation and real-world environments.

## Provided ROS2 Nodes

- **compass_inference**: Consumes camera images, target poses, and robot speed inputs, then outputs velocity commands by running TensorRT inference with COMPASS engines.
- **obj_target_generator**: Receives object localization bounding boxes and generates navigation target poses for the COMPASS navigator.

These nodes enable a variety of integration workflows, including the following examples:

---

## 1. Isaac Sim Integration

The `compass_navigator` package is fully compatible with [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim), leveraging its robotics environments and ROS2 bridge for seamless simulation.

For detailed setup instructions and a step-by-step Isaac Sim integration guide, see [ISAACSIM_README.md](./ISAACSIM_README.md) in this directory.

---

## 2. Zero-Shot Sim2Real Transfer

The **compass_inference** node can also be deployed directly on real robots, enabling a seamless transition from simulation to real-world operation. By integrating visual SLAM solutions such as [cuVSLAM](https://nvidia-isaac-ros.github.io/concepts/visual_slam/cuvslam/index.html) for robot state estimation, the COMPASS model can support zero-shot Sim2Real transfer.

https://github.com/user-attachments/assets/9fd51703-f59c-47a2-9398-aba135036b55

---

## 3. Object Navigation Integration


By integrating an object localization module (e.g., [Locate3D](https://locate3d.atmeta.com/)), we can also enable object navigation with COMPASS. The **obj_target_generator** node can convert localized object bounding boxes into navigation goals, allowing the robot to autonomously approach specified objects.

https://github.com/user-attachments/assets/3928e9fd-f78d-4b8e-8bbc-9932b386ae6b



---
