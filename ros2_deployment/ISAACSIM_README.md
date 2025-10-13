# Compass Navigator ROS2 Deployment in ISAAC SIM

## Overview

**Standalone ROS2 deployment** for Compass Navigator with Isaac Sim integration.

### What's Included

**`compass_container.py`** - Container management script that:
- Manages **two separate Docker containers**: Compass Navigator and Isaac Sim
- Controls container lifecycle: start, stop, enter, and monitor both containers
- Handles TensorRT model conversion automatically
- Launches Isaac Sim container with ROS2 bridge
- Provides easy container entry and debugging

**`Dockerfile.deploy`** - Creates a container with:
- ROS2 Humble + CUDA 11.8 + TensorRT
- Compass Navigator package and dependencies
- Non-root user setup for development
- Proper FastDDS configuration for inter-container communication

## Quick Setup & Run

### 1. Build the Container
```bash
export COMPASS_PATH=/path/to/compass
export HUGGINGFACE_TOKEN=hf_your_token_here
cd $COMPASS_PATH/ros2_deployment
./build_compass_docker.sh
```

### 2. Launch Isaac Sim (Terminal 1)
```bash
./compass_container.py launch_isaac_sim
```
- **In Isaac Sim GUI:** Go to `Windows` → `Examples` → `Robotics Examples` → `ROS2` → `Navigation`
- **Load a robot scene** (e.g., Carter)
- **Press Play (▶️)** to start simulation

### 3. Launch Compass Navigator with GUI (Terminal 2)
```bash
./compass_container.py --rviz launch_nav
```
- Automatically converts ONNX to TensorRT (first time only)
- Builds ROS2 workspace and launches navigator
- Opens RViz2 with robot visualization

### 3.b Navigate the Robot
- **In RViz2:** Use the **2D Nav Goal** tool in the toolbar
- **Click anywhere on the map** to set navigation targets
- **Robot navigates autonomously** using AI-powered path planning

## 4. Alternative: Headless Mode (No GUI)

### 4.a Launch Compass Navigator
```bash
./compass_container.py launch_nav
```
- Runs navigator without RViz2 GUI (saves resources)
- All ROS2 topics still available for external tools

### 4.b Send Navigation Goals
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
ros2 topic echo /cmd_vel --once  # Check robot commands
ros2 topic echo /chassis/odom --once  # Check robot position
or visualize in Isaac-Sim
```

## Container Commands

```bash
# Compass Navigator container lifecycle
./compass_container.py start                    # Start compass container headless (background)
./compass_container.py --rviz start              # Start compass container with RViz GUI
./compass_container.py enter                    # Enter running compass container
./compass_container.py stop                     # Stop compass container
./compass_container.py logs                     # View compass container logs
./compass_container.py status                   # Check compass container status

# Isaac Sim container (separate)
./compass_container.py launch_isaac_sim         # Start Isaac Sim container (interactive)

# Combined launch options
./compass_container.py launch_nav               # Quick headless launch
./compass_container.py --rviz launch_nav         # Quick launch with RViz GUI
./compass_container.py launch_nav --force-trt-conversion  # Rebuild TensorRT engine (headless)
./compass_container.py --rviz launch_nav --force-trt-conversion  # Rebuild engine + GUI

# Development
./compass_container.py enter                    # Enter headless container
./compass_container.py --rviz enter              # Enter container with RViz support

# Troubleshooting
./compass_container.py debug_network_setup        # Debug ROS2 network setup
```

## Container Lifecycle

### Compass Navigator Container
- **Starts in detached mode** (runs in background)
- **Persists after `launch_nav` command exits** - container keeps running
- **Runs ROS2 nodes and RViz2** inside the container
- **Must be manually stopped** with `./compass_container.py stop`

```bash
./compass_container.py launch_nav
# Command exits, but container + ROS2 nodes continue running

./compass_container.py status    # Check if still running
./compass_container.py enter     # Access running container
./compass_container.py stop      # Stop container
```

### Isaac Sim Container
- **Starts in interactive mode** (blocks terminal)
- **Container exits when Isaac Sim GUI closes**
- **Auto-removes itself** when terminated
- **Command blocks until you close Isaac Sim**

```bash
./compass_container.py launch_isaac_sim
# Command blocks, shows Isaac Sim GUI
# Close Isaac Sim → container exits → command exits
# Container is automatically removed
```

### Typical Workflow
```bash
# Terminal 1: Start Isaac Sim (interactive, blocks)
./compass_container.py launch_isaac_sim

# Terminal 2: Start Compass (detached, returns immediately)
./compass_container.py launch_nav

# Use both systems...

# Terminal 1: Close Isaac Sim GUI (container auto-exits)
# Terminal 2: Stop Compass container
./compass_container.py stop
```

## Troubleshooting

**No GUI / Black screen:**
```bash
# X11 permissions are checked automatically when launching Isaac Sim
# If launch fails with X11 errors, fix permissions:
xhost +local:docker
export DISPLAY=:0
```

**Topics not visible between containers:**
```bash
# Verify ROS2 domain consistency
echo $ROS_DOMAIN_ID  # Should be same (0 by default)
```

**Package not found errors:**
```bash
# Force workspace rebuild
./compass_container.py enter
# Inside container: rm -rf build install log && colcon build --symlink-install
```

**Model updated (need TensorRT conversion):**
```bash
# If you manually changed the ONNX model in assets/models/
./compass_container.py launch_nav --force-trt-conversion
```

**Manual workspace build (inside container):**
```bash
cd /home/compassuser/compass_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

**Build Issues:**

**HUGGINGFACE_TOKEN not set:**
```bash
# Get token from https://huggingface.co/settings/tokens after requesting access to nvidia/COMPASS
export HUGGINGFACE_TOKEN=hf_your_token_here
./build_compass_docker.sh
```

**TensorRT conversion fails:**
```bash
# Check GPU memory usage
nvidia-smi
# Stop other GPU processes if needed
./compass_container.py stop
./compass_container.py launch_nav --force-trt-conversion
```

**ROS2 Communication Issues:**

**Robot doesn't respond to navigation commands:**
```bash
# Verify Isaac Sim is running and scene is playing (Play ▶️ button pressed)
# Inside compass navigator container:
./compass_container.py enter
# Inside container:
ros2 topic list | grep -E "(cmd_vel|odom|image)"
ros2 topic echo /cmd_vel --once  # Should show navigation commands when you set goals
```

**Network troubleshooting:**
```bash
# Automated network diagnostics (recommended)
./compass_container.py debug_network_setup
# This checks environment variables, network mode, and ROS2 topic discovery in both containers

# Manual checks (if needed):
docker inspect compass-ros2-navigator | grep NetworkMode  # Should show: "NetworkMode": "host"
```
