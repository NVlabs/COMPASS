#!/bin/bash

# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

# Default values
SKIP_ASSETS=false
DOCKER_TAG="latest"
CACHE_DIR="${PWD}/assets"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Function to display help
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Build script for Compass ROS2 Navigation Deployment Container.

This script builds a standalone Docker container with ROS2 Humble, CUDA 11.8,
and all dependencies needed for Compass navigation.

OPTIONAL OPTIONS:
    --skip-assets               Skip asset preparation (download ONNX model)
    --cache-dir DIR             Directory to store assets (default: ./assets)
    --docker-tag TAG            Docker tag to use (default: latest)
    --help                      Show this help message and exit

REQUIREMENTS:
    1. Docker with NVIDIA runtime support
    2. NVIDIA GPU drivers
    3. For asset download: HuggingFace token for gated model access

WORKFLOW:
    1. Validate Docker and NVIDIA runtime
    2. Prepare assets (download COMPASS ONNX model with HuggingFace auth)
    3. Build Compass ROS2 deployment container
    4. Create usage instructions

EXAMPLES:
    # Full build with asset preparation
    export HUGGINGFACE_TOKEN=hf_your_token_here
    $0

    # Build without asset preparation (ONNX model already exists)
    $0 --skip-assets

    # Custom assets directory and tag
    $0 --cache-dir /tmp/compass_assets --docker-tag v1.0

EOF
}

# Function to validate Docker and NVIDIA runtime
validate_docker() {
    print_step "Validating Docker setup"

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        print_error "Please install Docker first: https://docs.docker.com/get-docker/"
        exit 1
    fi

    # Check if Docker is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        print_error "Please start Docker daemon"
        exit 1
    fi

    # Check NVIDIA runtime
    if docker info 2>/dev/null | grep -q "nvidia"; then
        print_info "NVIDIA Docker runtime detected"
    else
        print_warning "NVIDIA Docker runtime not detected"
        print_warning "Make sure nvidia-docker2 or Docker with NVIDIA runtime is installed"
        print_warning "See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
    fi

    print_info "Docker validation completed"
}

# Function to validate paths and setup
validate_setup() {
    if [ -z "$COMPASS_PATH" ]; then
        print_error "COMPASS_PATH environment variable is not set"
        print_error "Set it with: export COMPASS_PATH=/path/to/compass"
        exit 1
    fi

    if [ ! -d "$COMPASS_PATH" ]; then
        print_error "COMPASS_PATH directory does not exist: $COMPASS_PATH"
        exit 1
    fi

    # Check if Dockerfile exists
    local dockerfile="$COMPASS_PATH/ros2_deployment/docker/Dockerfile.deploy"
    if [ ! -f "$dockerfile" ]; then
        print_error "Dockerfile.deploy not found: $dockerfile"
        exit 1
    fi

    print_info "Setup validation completed"
}

# Function to prepare assets
prepare_assets() {
    if [ "$SKIP_ASSETS" = true ]; then
        print_info "Skipping asset preparation (--skip-assets flag)"

        # Check if ONNX model exists
        local onnx_file="${CACHE_DIR}/models/compass_carter.onnx"
        if [ ! -f "$onnx_file" ] || [ ! -s "$onnx_file" ]; then
            print_error "ONNX model not found or empty: $onnx_file"
            print_error "Run without --skip-assets to download it, or manually place the ONNX model file"
            exit 1
        fi

        return 0
    fi

    print_step "Preparing Compass navigation assets"

    # Check if prepare_assets.sh exists
    local prepare_script="$COMPASS_PATH/ros2_deployment/prepare_assets.sh"
    if [ ! -f "$prepare_script" ]; then
        print_error "Asset preparation script not found: $prepare_script"
        exit 1
    fi

    # Make sure it's executable
    chmod +x "$prepare_script"

    # Run asset preparation with the specified cache directory
    "$prepare_script" --cache-dir "$CACHE_DIR"

    print_info "Asset preparation completed"
}

# Function to build Docker image
build_docker_image() {
    print_step "Building Compass ROS2 deployment container"

    # Check if Dockerfile exists
    local dockerfile="$COMPASS_PATH/ros2_deployment/docker/Dockerfile.deploy"
    if [ ! -f "$dockerfile" ]; then
        print_error "Dockerfile.deploy not found: $dockerfile"
        exit 1
    fi

    # Build the deployment container
    print_info "Building compass-ros2-deploy:$DOCKER_TAG..."
    docker build \
        -t compass-ros2-deploy:$DOCKER_TAG \
        -f "$dockerfile" \
        "$COMPASS_PATH"

    print_info "Docker image built successfully: compass-ros2-deploy:$DOCKER_TAG"
}

# Function to display usage instructions
show_usage_instructions() {
    print_step "Build Summary"
    echo ""
    echo "✓ ONNX model prepared in: ${CACHE_DIR}/models/"
    echo "✓ Docker image built: compass-ros2-deploy:${DOCKER_TAG}"
    echo ""
    echo "To run the compass navigator:"
    echo ""
    echo "1. Start the container:"
    echo "   docker run -it --rm --gpus all \\"
    echo "     -v ${CACHE_DIR}/models:/tmp/models:ro \\"
    echo "     -v /tmp:/tmp \\"
    echo "     --name compass-navigator \\"
    echo "     compass-ros2-deploy:${DOCKER_TAG}"
    echo ""
    echo "2. Inside the container, convert ONNX to TensorRT (first time only):"
    echo "   python3 /usr/local/bin/trt_conversion.py \\"
    echo "     --onnx-path /tmp/models/compass_carter.onnx \\"
    echo "     --trt-path /tmp/engines/compass_carter.engine"
    echo ""
    echo "3. Build the ROS2 workspace (if needed):"
    echo "   cd /home/compassuser/compass_ws"
    echo "   source /opt/ros/humble/setup.bash"
    echo "   colcon build --symlink-install"
    echo ""
    echo "4. Launch the compass navigator:"
    echo "   launch_compass.sh"
    echo ""
    echo "Optional parameters for launch_compass.sh:"
    echo "   launch_compass.sh use_sim_time:=true log_level:=debug"
    echo ""
    echo "Note: TensorRT conversion happens once and creates an optimized engine for your GPU."
    echo "The conversion may take 1-5 minutes depending on your hardware."
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-assets)
            SKIP_ASSETS=true
            shift
            ;;
        --cache-dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --docker-tag)
            DOCKER_TAG="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_info "Compass ROS2 Deployment Docker Build Script"
    print_info "============================================"
    print_info "Compass Path: $COMPASS_PATH"
    print_info "Cache Directory: $CACHE_DIR"
    print_info "Docker Tag: $DOCKER_TAG"
    print_info "Skip Assets: $SKIP_ASSETS"
    if [ -n "$HUGGINGFACE_TOKEN" ]; then
        print_info "HuggingFace Token: Set (${HUGGINGFACE_TOKEN:0:10}...)"
    else
        print_warning "HuggingFace Token: Not set (required for asset download)"
    fi
    echo ""

    # Validate Docker setup
    validate_docker

    # Validate paths and setup
    validate_setup

    # Prepare assets
    prepare_assets

    # Build Docker image
    build_docker_image

    # Show usage instructions
    show_usage_instructions

    print_info "Build completed successfully!"
}

# Run main function
main "$@"
