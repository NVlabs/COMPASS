# Isaac Sim setup

The deeper-dive companion to the [ROS2 Deployment](ros2.md) overview —
container build, ROS2 bridge configuration, and the Isaac Sim launch
recipe.

## Overview

Standalone ROS2 deployment for the COMPASS navigator with Isaac Sim
integration.

### What's included

[`ros2_deployment/compass_container.py`](https://github.com/NVlabs/COMPASS/blob/main/ros2_deployment/compass_container.py)
is the container management script. It:

- Manages **two separate Docker containers** — COMPASS Navigator and Isaac Sim
- Controls container lifecycle: start, stop, enter, monitor both containers
- Handles TensorRT model conversion automatically
- Launches the Isaac Sim container with the ROS2 bridge
- Provides easy container entry and debugging

[`ros2_deployment/docker/Dockerfile.deploy`](https://github.com/NVlabs/COMPASS/tree/main/ros2_deployment/docker)
creates a container with:

- ROS2 Humble + CUDA 11.8 + TensorRT
- the COMPASS Navigator package and its dependencies
- a non-root user setup for development
- proper FastDDS configuration for inter-container communication

## Quick setup & run

### 1. Build the container

```bash
export COMPASS_PATH=/path/to/compass
export HUGGINGFACE_TOKEN=hf_your_token_here
cd $COMPASS_PATH/ros2_deployment
./build_compass_docker.sh
```

### 2. Launch Isaac Sim (terminal 1)

```bash
./compass_container.py launch_isaac_sim
```

- **In Isaac Sim GUI:** Go to `Windows` → `Examples` → `Robotics Examples` → `ROS2` → `Navigation`.
- **Load a robot scene** (e.g., Carter).
- **Press Play (▶️)** to start simulation.

### 3. Launch COMPASS Navigator with GUI (terminal 2)

```bash
./compass_container.py --rviz launch_nav
```

- Automatically converts ONNX → TensorRT (first time only)
- Builds the ROS2 workspace and launches the navigator
- Opens RViz2 with robot visualization

### 3.b Navigate the robot

- **In RViz2:** use the **2D Nav Goal** tool in the toolbar.
- **Click anywhere on the map** to set navigation targets.
- **Robot navigates autonomously** using AI-powered path planning.

## 4. Alternative: headless mode (no GUI)

### 4.a Launch COMPASS Navigator

```bash
./compass_container.py launch_nav
```

- Runs the navigator without RViz2 GUI (saves resources).
- All ROS2 topics still available for external tools.

### 4.b Send navigation goals

```bash
# Enter the compass container
./compass_container.py enter

# Inside container, publish a goal pose
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped "
header:
  frame_id: 'odom'
pose:
  position:
    x: 2.0
    y: 0.0
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0
"

# Monitor navigation progress
ros2 topic echo /cmd_vel --once       # robot commands
ros2 topic echo /chassis/odom --once  # robot position
# (or visualize in Isaac Sim)
```

## Container commands

```bash
# COMPASS Navigator container lifecycle
./compass_container.py start                                # start headless (background)
./compass_container.py --rviz start                         # start with RViz GUI
./compass_container.py enter                                # enter running container
./compass_container.py stop                                 # stop container
./compass_container.py logs                                 # view logs
./compass_container.py status                               # status

# Isaac Sim container (separate)
./compass_container.py launch_isaac_sim                     # interactive

# Combined launch options
./compass_container.py launch_nav                           # quick headless launch
./compass_container.py --rviz launch_nav                    # quick launch with RViz GUI
./compass_container.py launch_nav --force-trt-conversion    # rebuild TRT engine (headless)
./compass_container.py --rviz launch_nav --force-trt-conversion

# Development
./compass_container.py enter                                # enter headless container
./compass_container.py --rviz enter                         # enter with RViz support

# Troubleshooting
./compass_container.py debug_network_setup                  # debug ROS2 network setup
```

## Container lifecycle

### COMPASS Navigator container

- **Starts in detached mode** (runs in background).
- **Persists after `launch_nav` exits** — the container keeps running.
- **Runs ROS2 nodes and RViz2** inside the container.
- **Must be manually stopped** with `./compass_container.py stop`.

```bash
./compass_container.py launch_nav
# Command exits, but the container + ROS2 nodes continue running.

./compass_container.py status   # check if still running
./compass_container.py enter    # access running container
./compass_container.py stop     # stop container
```

### Isaac Sim container

- **Starts in interactive mode** (blocks the terminal).
- **Container exits when the Isaac Sim GUI closes**.
- **Auto-removes itself** when terminated.
- **Command blocks** until you close Isaac Sim.

```bash
./compass_container.py launch_isaac_sim
# Command blocks, shows the Isaac Sim GUI.
# Close Isaac Sim → container exits → command exits.
# The container is automatically removed.
```

### Typical workflow

```bash
# Terminal 1: start Isaac Sim (interactive, blocks)
./compass_container.py launch_isaac_sim

# Terminal 2: start COMPASS (detached, returns immediately)
./compass_container.py launch_nav

# Use both systems...

# Terminal 1: close the Isaac Sim GUI (container auto-exits)
# Terminal 2: stop the COMPASS container
./compass_container.py stop
```

## Troubleshooting

**No GUI / black screen**

```bash
# X11 permissions are checked automatically when launching Isaac Sim.
# If launch fails with X11 errors, fix permissions:
xhost +local:docker
export DISPLAY=:0
```

**Topics not visible between containers**

```bash
# Verify ROS2 domain consistency
echo $ROS_DOMAIN_ID  # Should be the same in both containers (0 by default)
```

**Package not found errors**

```bash
# Force a workspace rebuild
./compass_container.py enter
# Inside the container:
rm -rf build install log && colcon build --symlink-install
```

**Model updated (need TensorRT conversion)**

```bash
# If you manually changed the ONNX model in assets/models/
./compass_container.py launch_nav --force-trt-conversion
```

**Manual workspace build (inside container)**

```bash
cd /home/compassuser/compass_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

**Build issues — `HUGGINGFACE_TOKEN` not set**

```bash
# Get a token from https://huggingface.co/settings/tokens after requesting
# access to nvidia/COMPASS.
export HUGGINGFACE_TOKEN=hf_your_token_here
./build_compass_docker.sh
```

**TensorRT conversion fails**

```bash
# Check GPU memory usage
nvidia-smi
# Stop other GPU processes if needed
./compass_container.py stop
./compass_container.py launch_nav --force-trt-conversion
```

**ROS2 communication issues — robot doesn't respond to navigation commands**

```bash
# Verify Isaac Sim is running and the scene is playing (Play ▶️ pressed).
# Inside the COMPASS Navigator container:
./compass_container.py enter
# Inside the container:
ros2 topic list | grep -E "(cmd_vel|odom|image)"
ros2 topic echo /cmd_vel --once  # should show navigation commands when goals are set
```

**Network troubleshooting**

```bash
# Automated diagnostics (recommended)
./compass_container.py debug_network_setup
# This checks env vars, network mode, and ROS2 topic discovery in both containers.

# Manual checks (if needed):
docker inspect compass-ros2-navigator | grep NetworkMode  # Should show: "NetworkMode": "host"
```
