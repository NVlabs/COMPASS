# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os

import math
import numpy as np
import yaml

import torch
from PIL import Image
from scipy import ndimage

from mobility_es.config.environments import OMAP_PATHS


class OccupancyMapCollisionChecker:
    """Convert a occupancy map to grid map for collision checking.
        Note: This collision checker only works when the following conditions are met:
            1. The USD world frame's origin has x facing right, y facing up.
            2. The occupancy map must be generated in Isaac Sim with origin as (0,0,0),
               so the occupancy map's origin is measured in the USD's world frame.
            3. The occupancy map's origin is placed at the top-left of the PNC file with
               x facing right, and y facing down.
    """

    def __init__(self, cfg, precompute_valid_poses=False):    # pylint: disable=unused-argument
        self.__grid_map = None
        self.__map_meta = None
        self.__start_pose_valid_locations = None
        self.__goal_pose_valid_locations = None
        self.__start_pose_valid_orientations = None    # Store valid orientations per position
        self.__goal_pose_valid_orientations = None    # Store valid orientations per position
        # Store collision distances for potential future use
        self.__start_collision_distance = None    # pylint: disable=unused-private-member
        self.__goal_collision_distance = None    # pylint: disable=unused-private-member

        # Initialize the grid map using the occupancy map yaml file path.
        env_prim_path = cfg.environment.prim_path
        yaml_file_path = None
        for key, val in OMAP_PATHS.items():
            if env_prim_path.endswith(key):
                yaml_file_path = val
                break

        if yaml_file_path and os.path.exists(yaml_file_path):
            print(f'Loaded omap for env {env_prim_path}')
            self._load_grid_map(yaml_file_path)

    def _load_grid_map(self, yaml_file_path):
        """
        Reads the map image and generates the grid map.\n
        Grid map is a 2D boolean matrix where True=>occupied space & False=>Free space.
        """
        with open(yaml_file_path, encoding="utf-8") as f:
            file_content = f.read()
        self.__map_meta = yaml.safe_load(file_content)
        img_file_path = os.path.join(os.path.dirname(yaml_file_path), self.__map_meta["image"])

        img = Image.open(img_file_path)
        img = np.array(img)

        # Anything greater than free_thresh is considered as occupied
        if self.__map_meta["negate"]:
            res = np.where((img / 255)[:, :, 0] > self.__map_meta["free_thresh"])
        else:
            res = np.where(((255 - img) / 255)[:, :, 0] > self.__map_meta["free_thresh"])

        self.__grid_map = np.zeros(shape=(img.shape[:2]), dtype=bool)
        for i in range(res[0].shape[0]):
            self.__grid_map[res[0][i], res[1][i]] = True

    def __transform_to_image_coordinates(self, point):
        p_x = point[:, 0]
        p_y = point[:, 1]
        # NOTE: this assumes the occupancy map's origin with x facing right, and y facing down, and
        # the world frame's origin with x facing right, y facing up.
        i_x = torch.floor(
            (p_x - self.__map_meta["origin"][0]) / self.__map_meta["resolution"]).reshape(
                (point.shape[0], 1))
        i_y = torch.floor(
            (self.__map_meta["origin"][1] - p_y) / self.__map_meta["resolution"]).reshape(
                (point.shape[0], 1))
        return torch.cat((i_x, i_y), dim=1).int()

    def __is_obstacle_in_distance(self, img_point, distance_in_pixel):
        row_start_idx = img_point[:, 1] - distance_in_pixel
        col_start_idx = img_point[:, 0] - distance_in_pixel
        row_end_idx = img_point[:, 1] + distance_in_pixel
        col_end_idx = img_point[:, 0] + distance_in_pixel

        # image point acts as the center of the square, where each side of square is of size
        # 2xdistance
        in_collision = torch.zeros_like(row_start_idx).int()
        for i in range(row_end_idx.shape[0]):
            # Consider out of bounds points as in collision.
            if row_start_idx[i] < 0 or col_start_idx[i] < 0 or row_end_idx[
                    i] >= self.__grid_map.shape[0] or col_end_idx[i] >= self.__grid_map.shape[1]:
                in_collision[i] = 1
                continue
            patch = self.__grid_map[row_start_idx[i]:row_end_idx[i],
                                    col_start_idx[i]:col_end_idx[i]]
            if np.any(patch):
                in_collision[i] = 1

        return in_collision

    def is_in_collision(self, points, distance=0.75):
        assert points.shape[1] == 2
        img_points = self.__transform_to_image_coordinates(points)
        distance_in_pixel = math.ceil(distance / self.__map_meta["resolution"])
        in_collision = self.__is_obstacle_in_distance(img_points, distance_in_pixel)
        return in_collision

    def is_initialized(self):
        return self.__grid_map is not None

    def precompute_valid_poses(self, start_collision_distance, goal_collision_distance):
        """Precompute valid pose locations for start and goal poses.

        Args:
            start_collision_distance (float): Collision distance for start pose sampling (in meters)
            goal_collision_distance (float): Collision distance for goal pose sampling (in meters)
        """
        if not self.is_initialized():
            print("Warning: Cannot precompute valid poses - collision checker not initialized")
            return

        print(f"Precomputing valid pose locations (start_distance={start_collision_distance}m, "
              f"goal_distance={goal_collision_distance}m)...")

        # Store collision distances for orientation checking (may be used in future)
        self.__start_collision_distance = start_collision_distance    # pylint: disable=unused-private-member
        self.__goal_collision_distance = goal_collision_distance    # pylint: disable=unused-private-member

        # Buffer obstacles for start pose
        start_buffered_map = self._buffer_obstacles(self.__grid_map, start_collision_distance)
        self.__start_pose_valid_locations = self._extract_free_space_locations(
            start_buffered_map, self.__map_meta)

        # Precompute valid orientations for each start pose location
        # Check free space in different directions with 2x collision distance
        # free_space_map should be True=free, False=occupied (inverse of buffered map)
        print("Computing valid orientations for start poses...")
        # Invert: True=free, False=occupied
        free_space_map = np.logical_not(start_buffered_map)
        self.__start_pose_valid_orientations = self._compute_valid_orientations(
            self.__start_pose_valid_locations, free_space_map, 2.0 * start_collision_distance)

        # Buffer obstacles for goal pose
        goal_buffered_map = self._buffer_obstacles(self.__grid_map, goal_collision_distance)
        self.__goal_pose_valid_locations = self._extract_free_space_locations(
            goal_buffered_map, self.__map_meta)

        # Precompute valid orientations for each goal pose location
        # Check free space in different directions with 2x collision distance
        print("  Computing valid orientations for goal poses...")
        # Invert: True=free, False=occupied
        goal_free_space_map = np.logical_not(goal_buffered_map)
        self.__goal_pose_valid_orientations = self._compute_valid_orientations(
            self.__goal_pose_valid_locations, goal_free_space_map, 2.0 * goal_collision_distance)

        print(f"  Start pose valid locations: {len(self.__start_pose_valid_locations):,}")
        print(f"  Goal pose valid locations: {len(self.__goal_pose_valid_locations):,}")

        # Print memory usage
        start_memory = self.__start_pose_valid_locations.nbytes
        goal_memory = self.__goal_pose_valid_locations.nbytes
        start_orientation_memory = sum(
            len(orientations) * 8 for orientations in self.__start_pose_valid_orientations
        ) if self.__start_pose_valid_orientations else 0
        goal_orientation_memory = (sum(
            len(orientations) * 8 for orientations in self.__goal_pose_valid_orientations)
                                   if self.__goal_pose_valid_orientations else 0)
        total_memory = (start_memory + goal_memory + start_orientation_memory +
                        goal_orientation_memory)

        def format_bytes(bytes_size):
            for unit in ['B', 'KB', 'MB', 'GB']:
                if bytes_size < 1024.0:
                    return f"{bytes_size:.2f} {unit}"
                bytes_size /= 1024.0
            return f"{bytes_size:.2f} TB"

        print(f"  Memory: start={format_bytes(start_memory)}, goal={format_bytes(goal_memory)}, "
              f"start_orientations={format_bytes(start_orientation_memory)}, "
              f"goal_orientations={format_bytes(goal_orientation_memory)}, "
              f"total={format_bytes(total_memory)}")

    def _buffer_obstacles(self, grid_map, collision_distance_meters):
        """Buffer obstacles by collision distance to create a safety zone.

        Args:
            grid_map (np.ndarray): boolean array (True=occupied, False=free)
            collision_distance_meters (float): collision distance in meters

        Returns:
            np.ndarray: buffered grid map (True=occupied or buffered, False=free)
        """
        resolution = self.__map_meta["resolution"]
        buffer_distance_pixels = collision_distance_meters / resolution

        if buffer_distance_pixels > 0:
            # Create circular structuring element for buffering
            size = int(2 * buffer_distance_pixels) + 1
            center = size // 2

            # Create circular mask with radius = buffer_distance_pixels
            y, x = np.ogrid[:size, :size]
            mask = (x - center)**2 + (y - center)**2 <= buffer_distance_pixels**2
            struct = mask.astype(bool)

            # Dilate obstacles (True values) to create buffer
            occupied = grid_map.astype(np.uint8)
            buffered_occupied = ndimage.binary_dilation(occupied, structure=struct)

            return buffered_occupied.astype(bool)

        return grid_map

    def _check_free_space_in_direction(self, world_pos, yaw_rad, free_space_map,
                                       min_distance_meters):
        """Check if there's free space in a specific direction from a position.

        Args:
            world_pos (np.ndarray): World position [x, y]
            yaw_rad (float): Yaw angle in radians (0 = +x direction, pi/2 = +y direction)
            free_space_map (np.ndarray): boolean array (True=free, False=occupied)
            min_distance_meters (float): Minimum distance of free space required

        Returns:
            bool: True if there's at least min_distance_meters of free space in the direction
        """
        resolution = self.__map_meta["resolution"]
        min_distance_pixels = int(min_distance_meters / resolution)

        # Convert world position to image coordinates
        origin = np.array(self.__map_meta["origin"])
        img_x = int((world_pos[0] - origin[0]) / resolution)
        img_y = int((origin[1] - world_pos[1]) / resolution)    # Flip y axis

        # Check bounds
        if img_x < 0 or img_x >= free_space_map.shape[
                1] or img_y < 0 or img_y >= free_space_map.shape[0]:
            return False

        # Check if starting position is free
        if not free_space_map[img_y, img_x]:
            return False

        # Direction vector in image coordinates
        # In image: x is column (right), y is row (down)
        # In world: x is right, y is up
        # So: world x -> image x (same), world y -> image y (flipped)
        dir_x = np.cos(yaw_rad)    # World x direction
        dir_y = -np.sin(yaw_rad)    # World y direction (flipped for image)

        # Check along the direction
        for step in range(1, min_distance_pixels + 1):
            check_x = int(img_x + step * dir_x)
            check_y = int(img_y + step * dir_y)

            # Check bounds
            if check_x < 0 or check_x >= free_space_map.shape[
                    1] or check_y < 0 or check_y >= free_space_map.shape[0]:
                return False

            # Check if occupied (free_space_map: True=free, False=occupied)
            if not free_space_map[check_y, check_x]:
                return False    # Hit an obstacle

        return True    # All checks passed

    def _compute_valid_orientations(self, world_positions, free_space_map, min_distance_meters):
        """Compute valid orientations for each position based on free space in different directions.

        Args:
            world_positions (np.ndarray): Array of shape (N, 2) with world positions
            free_space_map (np.ndarray): boolean array (True=free, False=occupied)
            min_distance_meters (float): Minimum distance of free space required in a direction

        Returns:
            list: List of length N, each element is a list of valid yaw angles in radians.
                  If no valid orientations found, returns [0.0] as fallback.
        """
        valid_orientations = []
        # Check 16 directions: 0°, 22.5°, 45°, 67.5°, 90°, 112.5°, 135°, 157.5°,
        #                      180°, 202.5°, 225°, 247.5°, 270°, 292.5°, 315°, 337.5°
        test_yaws = np.linspace(0, 2 * np.pi, 16, endpoint=False)

        for pos in world_positions:
            valid_yaws = []
            for yaw in test_yaws:
                if self._check_free_space_in_direction(pos, yaw, free_space_map,
                                                       min_distance_meters):
                    valid_yaws.append(yaw)
            # If no valid orientations found, default to 0° (will trigger retry during sampling)
            if len(valid_yaws) == 0:
                valid_yaws = [0.0]    # Default to 0° if no valid direction
            valid_orientations.append(np.array(valid_yaws))

        return valid_orientations

    def _extract_free_space_locations(self, grid_map, map_meta):
        """Extract all free space (unoccupied) locations and convert to world coordinates.

        Args:
            grid_map (np.ndarray): boolean array (True=occupied, False=free)
            map_meta (dict): map metadata with 'origin' and 'resolution'

        Returns:
            np.ndarray: numpy array of shape (N, 2) with [x_world, y_world] for all free space
                locations
        """
        # Find all free space pixels (where grid_map is False)
        free_pixels = np.argwhere(~grid_map)    # Shape: (N, 2) with [row, col]

        if len(free_pixels) == 0:
            return np.array([]).reshape(0, 2)

        # Convert from [row, col] to [x_pixel, y_pixel]
        # In image coordinates: x is column, y is row
        img_coords = free_pixels[:, [1, 0]]    # Swap to [x, y]

        # Convert to world coordinates
        origin = np.array(map_meta["origin"])
        resolution = map_meta["resolution"]
        world_x = origin[0] + img_coords[:, 0] * resolution
        world_y = origin[1] - img_coords[:, 1] * resolution    # Flip y axis

        return np.stack([world_x, world_y], axis=1)

    def sample_start_pose(self, num_samples):
        """Sample start poses from precomputed valid locations.

        Args:
            num_samples (int): Number of poses to sample

        Returns:
            np.ndarray: numpy array of shape (num_samples, 2) with [x_world, y_world] coordinates
        """
        if self.__start_pose_valid_locations is None or len(self.__start_pose_valid_locations) == 0:
            raise ValueError("Start pose valid locations not precomputed or empty")

        indices = np.random.choice(len(self.__start_pose_valid_locations),
                                   num_samples,
                                   replace=True)
        return self.__start_pose_valid_locations[indices]

    def sample_start_pose_with_orientation(self, num_samples, max_iteration=100):
        """Sample start poses with valid orientations from precomputed locations.

        For each sample, if the sampled pose has no valid orientations (only [0.0] fallback),
        retries with a different pose up to max_iteration times.

        Args:
            num_samples (int): Number of poses to sample
            max_iteration (int): Maximum number of iterations to find a pose with valid orientations

        Returns:
            tuple: (positions, orientations) where:
                - positions: numpy array of shape (num_samples, 2) with [x_world, y_world]
                - orientations: numpy array of shape (num_samples,) with yaw angles in radians

        Raises:
            ValueError: If no valid orientations found after max_iteration attempts
        """
        if self.__start_pose_valid_locations is None or len(self.__start_pose_valid_locations) == 0:
            raise ValueError("Start pose valid locations not precomputed or empty")
        if self.__start_pose_valid_orientations is None:
            raise ValueError("Start pose valid orientations not precomputed")

        positions = np.zeros((num_samples, 2))
        orientations = np.zeros(num_samples)

        for i in range(num_samples):
            # Try to find a pose with valid orientations (not just [0.0] fallback)
            for _ in range(max_iteration):    # pylint: disable=unused-variable
                idx = np.random.choice(len(self.__start_pose_valid_locations))
                valid_yaws = self.__start_pose_valid_orientations[idx]

                # Check if this pose has valid orientations (not just the [0.0] fallback)
                # A pose has valid orientations if:
                # - It has more than 1 orientation (multiple valid directions found)
                # - It has exactly 1 orientation that is not 0.0 (one valid direction found)
                # A pose only has [0.0] fallback if len==1 and value==0.0
                has_valid_orientations = len(valid_yaws) > 1 or (len(valid_yaws) == 1 and
                                                                 not np.isclose(valid_yaws[0], 0.0))

                if has_valid_orientations:
                    positions[i] = self.__start_pose_valid_locations[idx]
                    orientations[i] = np.random.choice(valid_yaws)
                    break
            else:
                # If we exhausted max_iteration, raise exception to trigger fallback
                raise ValueError(f"No valid orientations found after {max_iteration} "
                                 f"iterations for sample {i}")

        return positions, orientations

    def sample_goal_pose(self, num_samples):
        """Sample goal poses from precomputed valid locations.

        Args:
            num_samples (int): Number of poses to sample

        Returns:
            np.ndarray: numpy array of shape (num_samples, 2) with [x_world, y_world] coordinates
        """
        if self.__goal_pose_valid_locations is None or len(self.__goal_pose_valid_locations) == 0:
            raise ValueError("Goal pose valid locations not precomputed or empty")

        indices = np.random.choice(len(self.__goal_pose_valid_locations), num_samples, replace=True)
        return self.__goal_pose_valid_locations[indices]

    def sample_goal_pose_with_orientation(self, num_samples, max_iteration=100):
        """Sample goal poses with valid orientations from precomputed locations.

        For each sample, if the sampled pose has no valid orientations (only [0.0] fallback),
        retries with a different pose up to max_iteration times.

        Args:
            num_samples (int): Number of poses to sample
            max_iteration (int): Maximum number of iterations to find a pose with valid orientations

        Returns:
            tuple: (positions, orientations) where:
                - positions: numpy array of shape (num_samples, 2) with [x_world, y_world]
                - orientations: numpy array of shape (num_samples,) with yaw angles in radians

        Raises:
            ValueError: If no valid orientations found after max_iteration attempts
        """
        if self.__goal_pose_valid_locations is None or len(self.__goal_pose_valid_locations) == 0:
            raise ValueError("Goal pose valid locations not precomputed or empty")
        if self.__goal_pose_valid_orientations is None:
            raise ValueError("Goal pose valid orientations not precomputed")

        positions = np.zeros((num_samples, 2))
        orientations = np.zeros(num_samples)

        for i in range(num_samples):
            # Try to find a pose with valid orientations (not just [0.0] fallback)
            for _ in range(max_iteration):    # pylint: disable=unused-variable
                idx = np.random.choice(len(self.__goal_pose_valid_locations))
                valid_yaws = self.__goal_pose_valid_orientations[idx]

                # Check if this pose has valid orientations (not just the [0.0] fallback)
                # A pose has valid orientations if:
                # - It has more than 1 orientation (multiple valid directions found)
                # - It has exactly 1 orientation that is not 0.0 (one valid direction found)
                # A pose only has [0.0] fallback if len==1 and value==0.0
                has_valid_orientations = len(valid_yaws) > 1 or (len(valid_yaws) == 1 and
                                                                 not np.isclose(valid_yaws[0], 0.0))

                if has_valid_orientations:
                    positions[i] = self.__goal_pose_valid_locations[idx]
                    orientations[i] = np.random.choice(valid_yaws)
                    break
            else:
                # If we exhausted max_iteration, raise exception to trigger fallback
                raise ValueError(f"No valid orientations found after {max_iteration} "
                                 f"iterations for sample {i}")

        return positions, orientations

    def has_precomputed_start_poses(self):
        """Check if start pose locations are precomputed."""
        return self.__start_pose_valid_locations is not None and len(
            self.__start_pose_valid_locations) > 0

    def has_precomputed_goal_poses(self):
        """Check if goal pose locations are precomputed."""
        return self.__goal_pose_valid_locations is not None and len(
            self.__goal_pose_valid_locations) > 0
