#!/bin/bash
# =============================================================================
# Compass Navigator Launch Helper Script
# =============================================================================
#
# DESCRIPTION:
#   This script launches the Compass navigation system for ROS2. It handles:
#   - ROS2 workspace building and sourcing
#   - Compass navigator node startup with rviz2 visualization
#
#   NOTE: TensorRT conversion should be handled before calling this script.
#
# EXAMPLES:
#   launch_compass.sh                              # Basic launch
#   launch_compass.sh use_sim_time:=true          # Launch with sim time
#   launch_compass.sh runtime_path:=/custom/path  # Custom model path
#
# =============================================================================

set -e

# Configuration
MODELS_DIR="/tmp/models"
ENGINES_DIR="/tmp/engines"
ONNX_FILE="${MODELS_DIR}/compass_carter.onnx"
TRT_ENGINE="${ENGINES_DIR}/compass_carter.engine"
COMPASS_WS="/home/compassuser/compass_ws"

echo "=== Compass Navigator Launch Helper ==="

# Check if ONNX file exists
if [ ! -f "${ONNX_FILE}" ]; then
    echo "Error: ONNX file not found at ${ONNX_FILE}"
    echo "Please ensure the ONNX model is mounted into the container"
    exit 1
fi

# Check if TensorRT engine exists
if [ ! -f "${TRT_ENGINE}" ]; then
    echo "Error: TensorRT engine not found at ${TRT_ENGINE}"
    echo "Please run TensorRT conversion first to generate the engine"
    echo "You can use: python3 /usr/local/bin/trt_conversion.py --onnx-path ${ONNX_FILE} --trt-path ${TRT_ENGINE}"
    exit 1
fi

echo "Found ONNX model: ${ONNX_FILE} ($(du -h "${ONNX_FILE}" | cut -f1))"
echo "Found TensorRT engine: ${TRT_ENGINE} ($(du -h "${TRT_ENGINE}" | cut -f1))"

# Build compass workspace if not already built
cd "${COMPASS_WS}"
if [ ! -f "install/setup.bash" ]; then
    echo "Building compass workspace..."
    source /opt/ros/humble/setup.bash
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
fi

# Source compass workspace (ROS2 base already sourced automatically)
echo "Sourcing compass workspace..."
source "${COMPASS_WS}/install/setup.bash"

# Launch compass navigator
echo "Launching compass navigator..."
ros2 launch compass_navigator compass_navigator.launch.py \
    runtime_path:="${TRT_ENGINE}" \
    "$@"
