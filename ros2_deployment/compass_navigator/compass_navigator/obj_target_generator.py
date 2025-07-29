#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from vision_msgs.msg import BoundingBox3DArray, BoundingBox3D
from tf2_ros import TransformListener, Buffer
from tf2_ros import TransformException
from tf_transformations import euler_from_quaternion
import tf2_geometry_msgs


class ObjectTargetGenerator(Node):
    """ROS 2 node that produce navigation goal from object bounding box data."""

    def __init__(self):
        super().__init__('obj_target_generator')

        # Declare parameters
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('goal_frame', 'odom')
        self.declare_parameter('approach_distance', 1.0)

        # Get parameters
        self.robot_frame = self.get_parameter('robot_frame').value
        self.goal_frame = self.get_parameter('goal_frame').value
        self.approach_distance = self.get_parameter('approach_distance').value

        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscriber for bounding box data (using BoundingBox3D as bbox)
        self.object_bbox_subscriber = self.create_subscription(BoundingBox3DArray, '/object_bbox',
                                                               self.object_bbox_callback, 10)

        # Publisher for goal pose
        self.goal_pose_publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)

        self.get_logger().info('ObjectTargetGenerator node initialized')
        self.get_logger().info(f'Robot frame: {self.robot_frame}')
        self.get_logger().info(f'Goal frame: {self.goal_frame}')
        self.get_logger().info(f'Approach distance: {self.approach_distance}m')

    def object_bbox_callback(self, object_bbox_msg: BoundingBox3DArray):
        if not object_bbox_msg.boxes:
            self.get_logger().warn('No bounding boxes in message')
            return

        # Assume there is only one object in the array
        object_bbox = object_bbox_msg.boxes[0]

        # Transform bounding box to robot frame
        bbox_in_robot_frame = self.transform_bbox_to_robot_frame(object_bbox,
                                                                 object_bbox_msg.header)

        if bbox_in_robot_frame is None:
            return

        # Find closest edge and generate goal pose
        goal_pose = self.find_closest_edge_goal(bbox_in_robot_frame)
        if goal_pose is None:
            return

        # Publish goal pose
        self.publish_goal_pose(goal_pose)

    def transform_bbox_to_robot_frame(self, bbox: BoundingBox3D, header: Header):
        # Create a PoseStamped for the bbox center
        bbox_center_pose = PoseStamped()
        bbox_center_pose.header = header
        bbox_center_pose.pose = bbox.center

        # Transform to robot frame
        try:
            transform = self.tf_buffer.lookup_transform(self.robot_frame, header.frame_id,
                                                        header.stamp)
        except TransformException as ex:
            self.get_logger().error(f'Failed to transform bbox to robot frame: {ex}')
            return None

        # Transform the pose and create a new bbox with transformed center
        transformed_pose = tf2_geometry_msgs.do_transform_pose(bbox_center_pose.pose, transform)

        bbox_in_robot_frame = BoundingBox3D()
        bbox_in_robot_frame.center = transformed_pose
        bbox_in_robot_frame.size = bbox.size

        return bbox_in_robot_frame

    def find_closest_edge_goal(self, bbox_data: BoundingBox3D):
        """
        Find the closest edge of the bounding box and generate a goal pose.

        Args:
            bbox_data: BoundingBox3D message with transformed bbox data

        Returns:
            PoseStamped message with goal pose
        """
        # Extract yaw from bbox orientation (assuming pitch=0, roll=0)
        bbox_quat = bbox_data.center.orientation
        _, _, bbox_yaw = euler_from_quaternion([bbox_quat.x, bbox_quat.y, bbox_quat.z, bbox_quat.w])

        # Calculate bbox edges (assuming robot is at origin in robot frame)
        robot_position = np.array([0.0, 0.0])    # Robot position in its own frame
        bbox_center_2d = np.array([bbox_data.center.position.x, bbox_data.center.position.y])

        # Create rotation matrix for bbox yaw
        cos_yaw = math.cos(bbox_yaw)
        sin_yaw = math.sin(bbox_yaw)
        rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

        # Edge offset vectors in bbox local frame
        half_size_x = bbox_data.size.x / 2.0
        half_size_y = bbox_data.size.y / 2.0
        edge_offsets = {
            'front': np.array([half_size_x, 0.0]),    # +x edge in bbox frame
            'back': np.array([-half_size_x, 0.0]),    # -x edge in bbox frame
            'right': np.array([0.0, -half_size_y]),    # -y edge in bbox frame
            'left':
                np.array([0.0, half_size_y])    # +y edge in bbox frame
        }

        # Transform edge offsets to robot frame and calculate edge positions
        edges = {}
        edge_distances = {}
        for edge_name, offset in edge_offsets.items():
            edges[edge_name] = bbox_center_2d + rotation_matrix @ offset
            edge_distances[edge_name] = np.linalg.norm(edges[edge_name] - robot_position)

        # Find closest edge
        closest_edge_name = min(edge_distances, key=edge_distances.get)
        closest_edge_pos = edges[closest_edge_name]

        # Generate goal position at approach distance from the closest edge
        # Direction from bbox center to closest edge
        edge_direction = closest_edge_pos - bbox_center_2d
        if np.linalg.norm(edge_direction) > 0:
            edge_direction = edge_direction / np.linalg.norm(edge_direction)
        else:
            edge_direction = np.array([1.0, 0.0])    # Default direction

        # Goal position: move approach_distance away from bbox center towards the edge
        goal_position_2d = closest_edge_pos + edge_direction * self.approach_distance

        # Create goal pose
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = self.robot_frame
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = float(goal_position_2d[0])
        goal_pose.pose.position.y = float(goal_position_2d[1])
        goal_pose.pose.position.z = bbox_data.center.position.z

        # Set orientation to face the bbox center (yaw only, pitch=0, roll=0)
        yaw = math.atan2(bbox_center_2d[1] - goal_position_2d[1],
                         bbox_center_2d[0] - goal_position_2d[0])
        goal_pose.pose.orientation.x = 0.0
        goal_pose.pose.orientation.y = 0.0
        goal_pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_pose.pose.orientation.w = math.cos(yaw / 2.0)

        return goal_pose

    def publish_goal_pose(self, goal_pose: PoseStamped):
        try:
            transform = self.tf_buffer.lookup_transform(self.goal_frame, self.robot_frame, Time())
            transformed_goal = tf2_geometry_msgs.do_transform_pose_stamped(goal_pose, transform)
            transformed_goal.header.frame_id = self.goal_frame
            self.goal_pose_publisher.publish(transformed_goal)
        except TransformException as ex:
            self.get_logger().error(f'Failed to transform goal pose to {self.goal_frame}: {ex}')
            return


def main(args=None):
    rclpy.init(args=args)
    obj_target_generator = ObjectTargetGenerator()
    rclpy.spin(obj_target_generator)


if __name__ == '__main__':
    main()
