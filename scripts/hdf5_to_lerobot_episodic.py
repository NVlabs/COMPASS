#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
HDF5 to LeRobot Data Format Converter for Episodic Data

This script converts HDF5 data with episodic structure [T, S, ...] to the GR00T LeRobot format.
Where T = number of episodes, S = number of steps per episode.

Expected HDF5 structure:
- image: [T, S, H, W, C] - Image sequences
- speed: [T, S, 1] - Speed data  
- route: [T, S, 10, 4] - Route data
- goal_heading: [T, S, 2] - Goal heading data
- action: [T, S, 6] - Action data

Output structure:
.
├─meta 
│ ├─episodes.jsonl
│ ├─modality.json
│ ├─info.json
│ └─tasks.jsonl
├─videos
│ └─chunk-000
│   └─observation.images.ego_view
│     └─episode_000000.mp4
│     └─episode_000001.mp4
└─data
  └─chunk-000
    ├─episode_000000.parquet
    └─episode_000001.parquet
"""

import argparse
import logging

import cv2
import h5py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_hdf5_structure(hdf5_path: str) -> Dict[str, Any]:
    """Inspect the structure of an HDF5 file to understand the data format."""
    structure = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        def visit_func(name, obj):
            if isinstance(obj, h5py.Dataset):
                structure[name] = {
                    'shape': obj.shape,
                    'dtype': str(obj.dtype),
                    'size': obj.size
                }
        
        f.visititems(visit_func)
    
    return structure

def create_directory_structure(output_path: str, video_key: str = "observation.images.ego_view") -> None:
    """Create the required directory structure for GNx LeRobot format."""
    base_path = Path(output_path)
    
    # Create main directories
    (base_path / "meta").mkdir(parents=True, exist_ok=True)
    
    # Create chunk directories
    # GNx: only support one chunk folder for now
    chunk_name = "chunk-000"
    (base_path / "data" / chunk_name).mkdir(parents=True, exist_ok=True)
    (base_path / "videos" / chunk_name / video_key).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created directory structure at {output_path}")

def extract_video_from_images(images: np.ndarray, output_path: str, fps: float = 20.0) -> None:
    """Convert image sequence [S, H, W, C] to MP4 video."""
    if len(images.shape) != 4:  # Expected: [S, H, W, C]
        raise ValueError(f"Expected 4D image array [S, H, W, C], got shape {images.shape}")
    
    S, H, W, C = images.shape
    
    # Ensure images are uint8
    if images.dtype != np.uint8:
        if images.dtype in [np.float32, np.float64]:
            # Assuming images are in [0, 1] range
            if images.max() <= 1.0:
                images = (images * 255).astype(np.uint8)
            else:
                images = np.clip(images, 0, 255).astype(np.uint8)
        else:
            images = np.clip(images, 0, 255).astype(np.uint8)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    for s in range(S):
        frame = images[s]
        # Convert RGB to BGR if needed (OpenCV uses BGR)
        if C == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    
    video_writer.release()
    logger.info(f"Created video: {output_path}")

def create_modality_json(state_keys: List[str], action_keys: List[str], 
                        video_key: str = "observation.images.ego_view") -> Dict[str, Any]:
    """Create the modality.json file content based on data structure."""
    
    modality = {
        "state": {},
        "action": {},
        "video": {
            "ego_view": {
                "original_key": video_key
            }
        },
        "annotation": {
            "human.action.task_description": {
                "original_key": "annotation.human.action.task_description"
            },
            "human.validity": {
                "original_key": "annotation.human.validity"
            }
        }
    }
    
    # Add state modalities
    current_idx = 0
    for key in state_keys:
        if key == "speed":
            modality["state"]["speed"] = {"start": current_idx, "end": current_idx + 1}
            current_idx += 1
        elif key == "route":
            # Assuming routes is [S, 10, 4], flatten to [S, 40]
            modality["state"]["route"] = {"start": current_idx, "end": current_idx + 40}
            current_idx += 40
        elif key == "goal_heading":
            modality["state"]["goal_heading"] = {"start": current_idx, "end": current_idx + 2}
            current_idx += 2
    
    # Add action modalities  
    action_idx = 0
    for key in action_keys:
        if key == "action":
            # Get action dimension from the key
            modality["action"]["vel_cmd"] = {"start": action_idx, "end": action_idx + 3}  # Assuming 3D actions
            action_idx += 3
    
    return modality

def create_info_json(total_episodes: int, total_frames: int, state_dim: int, action_dim: int,
                    fps: float = 20.0, image_shape: Tuple[int, int, int] = (320, 512, 3),
                    video_key: str = "observation.images.ego_view") -> Dict[str, Any]:
    """Create the info.json file content."""
    
    # Build features dictionary
    features = {
        "observation.state": {
            "dtype": "float64",
            "shape": [state_dim],
            "names": [f"state_{i}" for i in range(state_dim)]
        },
        "action": {
            "dtype": "float64", 
            "shape": [action_dim],
            "names": [f"action_{i}" for i in range(action_dim)]
        },
        "timestamp": {
            "dtype": "float64",
            "shape": [1]
        },
        "annotation.human.action.task_description": {
            "dtype": "int64",
            "shape": [1]
        },
        "task_index": {
            "dtype": "int64",
            "shape": [1]
        },
        "annotation.human.validity": {
            "dtype": "int64",
            "shape": [1]
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [1]
        },
        "index": {
            "dtype": "int64", 
            "shape": [1]
        },
        video_key: {
            "dtype": "video",
            "shape": list(image_shape),
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": fps,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        }
    }
    
    info = {
        "codebase_version": "v2.0",
        "robot_type": "custom", 
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 2,
        "total_videos": total_episodes,
        "total_chunks": 0,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features
    }
    
    return info

def convert_episodic_hdf5_to_lerobot(
    hdf5_paths: List[str], 
    output_path: str,
    image_key: str = "images",
    speed_key: str = "speed", 
    route_key: str = "route",
    goal_heading_key: str = "goal_heading",
    action_key: str = "action",
    task_description: str = "Robot navigation task",
    fps: float = 20.0,
    video_key: str = "observation.images.ego_view"
) -> None:
    """
    Convert episodic HDF5 data [T, S, ...] to LeRobot format with multiple chunks.
    
    Args:
        hdf5_paths: List of paths to HDF5 files (each becomes a chunk)
        output_path: Output directory for LeRobot format
        image_key: Key for image data in HDF5
        speed_key: Key for speed data in HDF5
        route_key: Key for route data in HDF5 
        goal_heading_key: Key for goal heading data in HDF5
        action_key: Key for action data in HDF5
        task_description: Description of the task
        fps: Frames per second for video encoding
        video_key: Output video key name
    """
    
    logger.info(f"Converting {len(hdf5_paths)} HDF5 files to LeRobot format at {output_path}")
    
    # First, inspect the structure of the first HDF5 file to get schema info
    if not hdf5_paths:
        raise ValueError("No HDF5 files provided")
    
    structure = inspect_hdf5_structure(hdf5_paths[0])
    logger.info("HDF5 structure (from first file):")
    for key, info in structure.items():
        logger.info(f"  {key}: shape={info['shape']}, dtype={info['dtype']}")
    
    # Create directory structure for all chunks
    create_directory_structure(output_path, video_key)
    
    # Initialize totals for metadata
    total_episodes = 0
    total_frames = 0
    episodes_data = []
    global_episode_idx = 0
    global_observation_idx = 0  # Global index across all observations
    state_dim = None
    action_dim = None
    image_shape = None
    
    # Process each HDF5 file as a separate chunk
    for chunk_idx, hdf5_path in enumerate(tqdm(hdf5_paths, desc="Processing HDF5 files")):
        logger.info(f"Processing chunk {chunk_idx}: {hdf5_path}")
        
        with h5py.File(hdf5_path, 'r') as f:
            # Load all data for this chunk
            images = np.array(f[image_key])  # [T, S, H, W, C]
            speeds = np.array(f[speed_key])  # [T, S, 1]
            routes = np.array(f[route_key])  # [T, S, 10, 4] 
            actions = np.array(f[action_key])  # [T, S, A]
            goal_reached = np.array(f["goal_reached"])  # [T, 1]
            goal_headings = np.array(f[goal_heading_key])  # [T, S, 2]
            # Only process episodes where goal_reached is True
            # goal_reached: [T, 1] or [T], so flatten to [T]
            goal_reached_flat = goal_reached.flatten()
            valid_episode_indices = np.where(goal_reached_flat == True)[0]
            if len(valid_episode_indices) == 0:
                logger.info(f"No episodes with goal_reached=True in chunk {chunk_idx}, skipping chunk.")
                continue
            # Filter all arrays to only include valid episodes
            images = images[valid_episode_indices]
            speeds = speeds[valid_episode_indices]
            routes = routes[valid_episode_indices]
            actions = actions[valid_episode_indices]
            goal_headings = goal_headings[valid_episode_indices]
            T, S = images.shape[:2]  # Number of episodes, steps per episode
            
            logger.info(f"Chunk {chunk_idx} data shapes:")
            logger.info(f"  Images: {images.shape}")
            logger.info(f"  Speeds: {speeds.shape}")
            logger.info(f"  Routes: {routes.shape}")
            logger.info(f"  Actions: {actions.shape}")
            logger.info(f"  Goal heading: {goal_headings.shape}")
            logger.info(f"  Episodes in chunk: {T}, Steps per episode: {S}")
            
            # Calculate dimensions (use first chunk as reference)
            if state_dim is None:
                route_flat_dim = routes.shape[2] * routes.shape[3]  # 10 * 4 = 40
                goal_heading_dim = goal_headings.shape[2]  # 2
                state_dim = speeds.shape[2] + route_flat_dim + goal_heading_dim  # 1 + 40 + 2 = 43
                action_dim = actions.shape[2]  # A
                image_shape = images.shape[2:]  # [H, W, C]
                
                logger.info(f"Computed dimensions:")
                logger.info(f"  State dimension: {state_dim}")
                logger.info(f"  Action dimension: {action_dim}")
                logger.info(f"  Image shape: {image_shape}")
            
            # Update totals
            total_episodes += T
            total_frames += T * S
            
            # Process each episode in this chunk
            for local_episode_idx in tqdm(range(T), desc=f"Processing episodes in chunk {chunk_idx}", leave=False):
                # Extract episode data
                episode_images = images[local_episode_idx]  # [S, H, W, C]
                episode_speeds = speeds[local_episode_idx]  # [S, 1]
                episode_routes = routes[local_episode_idx]  # [S, 10, 4]
                episode_goal_headings = goal_headings[local_episode_idx]  # [S, 2]
                episode_actions = actions[local_episode_idx][:,(0,1,5)]  # [S, 3]
                # Flatten routes for each step
                episode_routes_flat = episode_routes.reshape(S, -1)  # [S, 40]
                
                # Combine state data
                episode_state = np.concatenate([episode_speeds, episode_routes_flat, episode_goal_headings], axis=1)  # [S, 43]
                
                # Create timestamps
                timestamps = np.arange(S, dtype=np.float64) / fps
                
                # Create parquet data with all required LeRobot fields
                parquet_data = {
                    # Core data
                    "observation.state": episode_state.tolist(),  # Concatenated state array per modality.json
                    "action": episode_actions.tolist(),  # Concatenated action array per modality.json
                    "timestamp": timestamps.tolist(),  # Timestamp from episode start

                    # Annotation system - indices to meta/tasks.jsonl
                    "annotation.human.action.task_description": [0] * S,  # Points to task_index 0 (main task)
                    "task_index": [0] * S,  # Main task index (same as above)
                    "annotation.human.validity": [1] * S,  # Points to task_index 1 ("valid")
                    
                    # Episode tracking
                    "episode_index": [global_episode_idx] * S,  # Episode number
                    "index": list(range(global_observation_idx, global_observation_idx + S)),  # GLOBAL obs index
                }
                
                # Update global observation index
                global_observation_idx += S
                
                # Save parquet file in the appropriate chunk directory
                # GNx: only support one chunk for now
                chunk_name = "chunk-000"
                df = pd.DataFrame(parquet_data)
                parquet_path = Path(output_path) / "data" / chunk_name / f"episode_{global_episode_idx:06d}.parquet"
                df.to_parquet(parquet_path, index=False)
                
                # Save video in the appropriate chunk directory
                video_path = Path(output_path) / "videos" / chunk_name / video_key / f"episode_{global_episode_idx:06d}.mp4"
                extract_video_from_images(episode_images, str(video_path), fps)
                
                # Add to episodes data  
                episodes_data.append({
                    "episode_index": global_episode_idx,
                    "tasks": [task_description, "valid"],  # Task description and validity
                    "length": S
                })
                
                global_episode_idx += 1
    
    # Create metadata files
    base_path = Path(output_path)
    
    # modality.json
    state_keys = [speed_key, route_key, goal_heading_key]
    action_keys = [action_key]
    modality_data = create_modality_json(state_keys, action_keys, video_key)
    with open(base_path / "meta" / "modality.json", 'w') as f:
        json.dump(modality_data, f, indent=2)
    
    # info.json  
    if state_dim is None or action_dim is None or image_shape is None:
        raise ValueError("Could not determine data dimensions from HDF5 files")
    
    info_data = create_info_json(total_episodes, total_frames, state_dim, action_dim, fps, image_shape, video_key)
    # Update info.json to reflect multiple chunks
    info_data["total_chunks"] = len(hdf5_paths)
    with open(base_path / "meta" / "info.json", 'w') as f:
        json.dump(info_data, f, indent=2)
    
    # episodes.jsonl
    with open(base_path / "meta" / "episodes.jsonl", 'w') as f:
        for episode in episodes_data:
            f.write(json.dumps(episode) + '\n')
    
    # tasks.jsonl
    tasks_data = [
        {"task_index": 0, "task": task_description},
        {"task_index": 1, "task": "valid"}
    ]
    with open(base_path / "meta" / "tasks.jsonl", 'w') as f:
        for task in tasks_data:
            f.write(json.dumps(task) + '\n')
    
    logger.info(f"✅ Successfully converted {len(hdf5_paths)} HDF5 files to LeRobot format at {output_path}")
    logger.info(f"📊 Statistics:")
    logger.info(f"   Total chunks: {len(hdf5_paths)}")
    logger.info(f"   Total episodes: {total_episodes}")
    logger.info(f"   Total frames: {total_frames}")
    logger.info(f"   State dimension: {state_dim}")
    logger.info(f"   Action dimension: {action_dim}")
    logger.info(f"   Image shape: {image_shape}")

def main():
    parser = argparse.ArgumentParser(description="Convert episodic HDF5 data [T, S, ...] to LeRobot format with multi-chunk support")
    parser.add_argument("--hdf5-dir", type=str, default=None, help="Optional directory to search for all HDF5 files to process")
    parser.add_argument("--hdf5-paths", nargs='*', help="Paths to HDF5 files (each becomes a chunk)")
    parser.add_argument("--output-path", help="Output directory for LeRobot format")
    parser.add_argument("--image-key", default="image", help="Key for image data in HDF5")
    parser.add_argument("--speed-key", default="speed", help="Key for speed data in HDF5")
    parser.add_argument("--route-key", default="route", help="Key for route data in HDF5")
    parser.add_argument("--goal-heading-key", default="goal_heading", help="Key for goal heading data in HDF5")
    parser.add_argument("--action-key", default="action", help="Key for action data in HDF5")
    parser.add_argument("--task-description", default="Robot navigation task", help="Description of the task")
    parser.add_argument("--fps", type=float, default=20.0, help="Frames per second for video encoding")
    parser.add_argument("--video-key", default="observation.images.ego_view", help="Output video key name")
    parser.add_argument("--inspect-only", action='store_true', help="Only inspect HDF5 structure without conversion")
    
    args = parser.parse_args()
    
    if args.hdf5_dir:
        hdf5_paths = list(Path(args.hdf5_dir).glob("*.h5"))
    else:
        hdf5_paths = args.hdf5_paths

    if args.inspect_only:
        print(f"Inspecting {len(hdf5_paths)} HDF5 files:")
        for i, hdf5_path in enumerate(hdf5_paths):
            print(f"\n--- File {i}: {hdf5_path} ---")
            structure = inspect_hdf5_structure(hdf5_path)
            for key, info in structure.items():
                print(f"  {key}: shape={info['shape']}, dtype={info['dtype']}")
        return


    convert_episodic_hdf5_to_lerobot(
        hdf5_paths,
        args.output_path,
        args.image_key,
        args.speed_key,
        args.route_key,
        args.goal_heading_key,
        args.action_key,
        args.task_description,
        args.fps,
        args.video_key
    )

if __name__ == "__main__":
    main()
