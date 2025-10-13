#!/bin/bash

# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

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

# Default paths
CACHE_DIR="./cache"
MODELS_DIR="${CACHE_DIR}/models"
ONNX_FILE="${MODELS_DIR}/compass_carter.onnx"

# HuggingFace model URLs
COMPASS_ONNX_URL="https://huggingface.co/nvidia/COMPASS/resolve/main/carter_w_goal_heading_alignment_isaac_sim.onnx"

# Function to create directory structure
create_dirs() {
    print_step "Creating directory structure"
    mkdir -p "${MODELS_DIR}"
    mkdir -p "${CACHE_DIR}/logs"
}

# Function to check HuggingFace authentication
check_huggingface_auth() {
    if [ -z "$HUGGINGFACE_TOKEN" ]; then
        print_error "HUGGINGFACE_TOKEN environment variable is not set"
        print_error "The NVIDIA COMPASS model is gated and requires authentication"
        print_error ""
        print_error "To fix this:"
        print_error "1. Go to https://huggingface.co/nvidia/COMPASS"
        print_error "2. Request access to the model"
        print_error "3. Get your HuggingFace token from https://huggingface.co/settings/tokens"
        print_error "4. Set the token: export HUGGINGFACE_TOKEN=your_token_here"
        print_error "5. Run this script again"
        exit 1
    fi

    print_info "HuggingFace token found, proceeding with authenticated download"
}

# Function to download model ONNX file
download_onnx_model() {
    print_step "Downloading COMPASS ONNX model"

    if [ -f "${ONNX_FILE}" ] && [ -s "${ONNX_FILE}" ]; then
        print_info "ONNX model already exists: ${ONNX_FILE}"
        local file_size=$(du -h "${ONNX_FILE}" | cut -f1)
        print_info "File size: ${file_size}"
        return 0
    fi

    # Remove any existing 0-byte file
    if [ -f "${ONNX_FILE}" ]; then
        print_warning "Removing existing 0-byte file: ${ONNX_FILE}"
        rm -f "${ONNX_FILE}"
    fi

    # Check HuggingFace authentication
    check_huggingface_auth

    print_info "Downloading COMPASS ONNX model from HuggingFace (authenticated)..."

    # Download with authentication
    if command -v wget &> /dev/null; then
        print_info "Using wget for download..."
        wget --progress=bar:force \
             --header="Authorization: Bearer ${HUGGINGFACE_TOKEN}" \
             -O "${ONNX_FILE}" \
             "${COMPASS_ONNX_URL}"
    elif command -v curl &> /dev/null; then
        print_info "Using curl for download..."
        curl -L --progress-bar \
             -H "Authorization: Bearer ${HUGGINGFACE_TOKEN}" \
             -o "${ONNX_FILE}" \
             "${COMPASS_ONNX_URL}"
    else
        print_error "Neither wget nor curl is available. Please install one of them."
        exit 1
    fi

    # Verify download
    if [ ! -f "${ONNX_FILE}" ] || [ ! -s "${ONNX_FILE}" ]; then
        print_error "Failed to download ONNX file or file is empty"
        print_error "This could be due to:"
        print_error "1. Invalid HuggingFace token"
        print_error "2. No access to the NVIDIA COMPASS model"
        print_error "3. Network connectivity issues"
        print_error ""
        print_error "Please check your token and model access at:"
        print_error "https://huggingface.co/nvidia/COMPASS"

        # Remove empty file if it exists
        if [ -f "${ONNX_FILE}" ]; then
            rm -f "${ONNX_FILE}"
        fi
        exit 1
    fi

    local file_size=$(du -h "${ONNX_FILE}" | cut -f1)
    print_info "Successfully downloaded: ${ONNX_FILE} (${file_size})"
}

# Function to display file information
show_asset_info() {
    print_step "Asset Summary"
    echo "Asset cache directory: ${CACHE_DIR}"
    echo "Models directory: ${MODELS_DIR}"
    echo ""

    if [ -f "${ONNX_FILE}" ] && [ -s "${ONNX_FILE}" ]; then
        echo "✓ ONNX Model: ${ONNX_FILE} ($(du -h "${ONNX_FILE}" | cut -f1))"
    elif [ -f "${ONNX_FILE}" ]; then
        echo "⚠ ONNX Model: ${ONNX_FILE} (0 bytes - download failed)"
    else
        echo "✗ ONNX Model: ${ONNX_FILE} (missing)"
    fi

    echo ""
    echo "Note: The ONNX model will be converted to TensorRT at runtime for optimal GPU-specific performance."
    echo "This ensures the best performance on your target hardware."

    if [ -z "$HUGGINGFACE_TOKEN" ]; then
        echo ""
        echo "⚠ HUGGINGFACE_TOKEN not set - required for downloading the gated model"
        echo "Set it with: export HUGGINGFACE_TOKEN=your_token_here"
    fi
}

# Function to display help
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Prepare assets for Compass ROS2 navigation deployment.

This script downloads the COMPASS ONNX model from HuggingFace.
The model will be converted to TensorRT at container runtime for optimal performance.

REQUIREMENTS:
    The NVIDIA COMPASS model is gated and requires authentication.
    You must:
    1. Request access at: https://huggingface.co/nvidia/COMPASS
    2. Get your token from: https://huggingface.co/settings/tokens
    3. Set the token: export HUGGINGFACE_TOKEN=your_token_here

OPTIONS:
    --cache-dir DIR         Directory to store cached assets (default: ./cache)
    --force                 Force re-download even if files exist
    --info                  Show information about existing assets and exit
    --help                  Show this help message and exit

ENVIRONMENT VARIABLES:
    HUGGINGFACE_TOKEN       Required for downloading the gated model

EXAMPLES:
    # Set token and download ONNX model
    export HUGGINGFACE_TOKEN=hf_your_token_here
    $0

    # Custom cache directory
    $0 --cache-dir /tmp/compass_cache

    # Force re-download everything
    $0 --force

    # Show what assets are currently available
    $0 --info

NOTE:
    The ONNX model will be optimized to TensorRT during container startup
    for best performance on your specific GPU hardware.

EOF
}

# Parse command line arguments
FORCE_MODE=false
INFO_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cache-dir)
            CACHE_DIR="$2"
            MODELS_DIR="${CACHE_DIR}/models"
            ONNX_FILE="${MODELS_DIR}/compass_carter.onnx"
            shift 2
            ;;
        --force)
            FORCE_MODE=true
            shift
            ;;
        --info)
            INFO_MODE=true
            shift
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
    print_info "Compass Navigation Asset Preparation"
    print_info "===================================="

    # Show info and exit if requested
    if [ "$INFO_MODE" = true ]; then
        show_asset_info
        exit 0
    fi

    # Create directory structure
    create_dirs

    # Force mode: remove existing files
    if [ "$FORCE_MODE" = true ]; then
        print_warning "Force mode enabled - removing existing files"
        rm -f "${ONNX_FILE}"
    fi

    # Download ONNX model
    download_onnx_model

    # Show final summary
    show_asset_info

    print_info "Asset preparation complete!"
    print_info "The ONNX model is ready for deployment."
    print_info "You can now run the Docker build script to create the compass deployment container."
}

# Run main function
main "$@"
