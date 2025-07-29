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

    def __init__(self, cfg):
        self.__grid_map = None

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
