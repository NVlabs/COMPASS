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

import numpy as np
import pycuda.autoinit    # pylint: disable=unused-import
import pycuda.driver as cuda
import rclpy
import tensorrt as trt
import onnxruntime as ort
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_pose
from tf_transformations import euler_from_quaternion

IMAGE_TOPIC_NAME = '/front_stereo_camera/left/image_raw'
ODOM_TOPIC_NAME = '/chassis/odom'
CMD_TOPIC_NAME = '/cmd_vel'
ROUTE_TOPIC_NAME = '/route'
GOAL_TOPIC_NAME = '/goal_pose'
PATH_TOPIC_NAME = '/compass_route'
RUNTIME_PATH = 'runtime_path'
MAPLESS_FLAG = 'is_mapless'

NUM_ROUTE_POINTS = 11
# Route vector with 4 values representing start and end positions
ROUTE_VECTOR_SIZE = 4
ROBOT_FRAME = 'base_link'


# Upsample the points between start and goal.
def upsample_points(start, goal, max_segment_length):
    x1, y1 = start
    x2, y2 = goal

    # Calculate the Euclidean distance between the two points
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Handle the case where the start and goal are too close (or identical)
    if distance <= max_segment_length:
        return [start, goal]

    # Determine the number of segments based on the maximum segment length
    num_segments = max(1, int(np.ceil(distance / max_segment_length)))

    # Generate the interpolated points
    interpolated_points = [(x1 + (i / num_segments) * (x2 - x1),
                            y1 + (i / num_segments) * (y2 - y1)) for i in range(num_segments + 1)]

    return interpolated_points


class CompassNavigator(Node):
    '''Compass Navigator ROS Node
    '''

    def __init__(self):
        super().__init__('compass_navigator')
        # Parameters
        self.declare_parameter(RUNTIME_PATH, '/tmp/compass.engine')
        self.declare_parameter(MAPLESS_FLAG, True)

        # Subscriber
        self.image_subscriber = self.create_subscription(Image, IMAGE_TOPIC_NAME,
                                                         self.image_callback, 10)
        self.odom_subscriber = self.create_subscription(Odometry, ODOM_TOPIC_NAME,
                                                        self.odom_callback, 10)
        self.goal_subscriber = self.create_subscription(PoseStamped, GOAL_TOPIC_NAME,
                                                        self.goal_callback, 10)

        # Publisher
        self.cmd_publisher = self.create_publisher(Twist, CMD_TOPIC_NAME, 10)
        self.path_publisher = self.create_publisher(Path, PATH_TOPIC_NAME, 10)

        # Timer
        self.timer = self.create_timer(0.2, self.inference)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Internal states
        self.action = np.zeros(6, dtype=np.float32)
        self.history = np.zeros((1, 1024), dtype=np.float32)
        self.sample = np.zeros((1, 512), dtype=np.float32)
        self.goal_heading = np.zeros(1, dtype=np.float32)
        self.camera_image = None
        self.route_vectors = None
        self.goal = None
        self.ego_speed = None
        self.runtime_context = None
        self.onnx_session = None
        self.stream = cuda.Stream()
        self.cv_bridge = CvBridge()

        # Runtime type
        self.runtime_type = None

    def load_model(self):
        self.get_logger().info('Loading model')
        runtime_path = self.get_parameter(RUNTIME_PATH).get_parameter_value().string_value

        # Detect runtime type from file extension if set to 'auto'
        if runtime_path.endswith('.engine'):
            self.runtime_type = 'tensorrt'
            self._load_tensorrt_model(runtime_path)
        elif runtime_path.endswith('.onnx'):
            self.runtime_type = 'onnx'
            self._load_onnx_model(runtime_path)
        else:
            raise ValueError(f"Could not determine runtime type from extension for {runtime_path}. "
                             "Using TensorRT as fallback.")

    def _load_tensorrt_model(self, runtime_path):
        with open(runtime_path, "rb") as f:
            engine_data = f.read()

        # Create a TensorRT runtime
        runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
        engine = runtime.deserialize_cuda_engine(engine_data)
        self.runtime_context = engine.create_execution_context()
        self.get_logger().info('TensorRT model loaded successfully')

    def _load_onnx_model(self, runtime_path):
        # Create ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.onnx_session = ort.InferenceSession(runtime_path, providers=providers)
        self.get_logger().info('ONNX model loaded successfully')

    def image_callback(self, image_msg):
        self.camera_image = self.process_image_msg(image_msg)

    def odom_callback(self, odom_msg):
        self.ego_speed = np.array(odom_msg.twist.twist.linear.x, dtype=np.float32)

    def goal_callback(self, goal_msg):
        self.goal = goal_msg
        # Reset history state
        self.history = np.zeros((1, 1024), dtype=np.float32)
        self.sample = np.zeros((1, 512), dtype=np.float32)

    def compose_mapless_route(self):
        if self.goal is None:
            return
        try:
            transform = self.tf_buffer.lookup_transform(ROBOT_FRAME, self.goal.header.frame_id,
                                                        Time())
        except TransformException as ex:
            self.get_logger().error(
                f'Could not transform {ROBOT_FRAME} to {self.goal.header.frame_id}: {ex}')
            return
        goal_in_robot_frame = do_transform_pose(self.goal.pose, transform)
        # Get the heading of the goal in robot frame.
        goal_orientation = goal_in_robot_frame.orientation
        quaternion = [
            goal_orientation.x, goal_orientation.y, goal_orientation.z, goal_orientation.w
        ]
        _, _, yaw = euler_from_quaternion(quaternion)
        self.goal_heading[0] = yaw
        # Upsample the points between start and goal.
        route_poses = upsample_points(
            [0.0, 0.0], [goal_in_robot_frame.position.x, goal_in_robot_frame.position.y], 1.0)
        num_poses = min(len(route_poses), NUM_ROUTE_POINTS)
        # Return if route is empty.
        if num_poses == 0:
            return
        # Select the first NUM_ROUTE_POINTS and append the last route point as needed.
        indices = [idx for idx in range(num_poses)]
        indices.extend([num_poses - 1] * (NUM_ROUTE_POINTS - len(indices)))
        # Extract the x and y position in robot frame.
        selected_route_positions = []
        for idx in indices:
            selected_route_positions.append(route_poses[idx])
        self.route_vectors = np.zeros((NUM_ROUTE_POINTS - 1, ROUTE_VECTOR_SIZE), np.float32)
        for idx in range(NUM_ROUTE_POINTS - 1):
            self.route_vectors[idx] = np.concatenate(
                (selected_route_positions[idx], selected_route_positions[idx + 1]), axis=0)

    def inference(self):
        # Load model if not ready
        if not self.runtime_context and not self.onnx_session:
            self.load_model()

        # Compose a simple route in mapless mode.
        if self.get_parameter(MAPLESS_FLAG).get_parameter_value().bool_value:
            self.compose_mapless_route()

        # Sanity checks of the inputs.
        # TODO: Sync the msgs.
        missing_inputs = []
        if self.camera_image is None:
            missing_inputs.append(f'camera_image (topic: {IMAGE_TOPIC_NAME})')
        if self.route_vectors is None:
            missing_inputs.append(f'route_vectors (goal not set or transform failed)')
        if self.ego_speed is None:
            missing_inputs.append(f'ego_speed (topic: {ODOM_TOPIC_NAME})')

        if missing_inputs:
            self.get_logger().info(f'Inputs are not ready. Missing: {", ".join(missing_inputs)}')
            return

        # Determine which inference method to use
        if self.runtime_type == 'onnx':
            self._onnx_inference()
        else:
            self._trt_inference()

        self.publish_action()
        self.publish_route()

    def _trt_inference(self):
        # Create a dictionary to map tensor names to their allocated memory
        input_bindings = {}
        output_bindings = {}

        # Allocate device memory for inputs
        input_bindings['image'] = cuda.mem_alloc(self.camera_image.nbytes)
        input_bindings['route'] = cuda.mem_alloc(self.route_vectors.nbytes)
        input_bindings['speed'] = cuda.mem_alloc(self.ego_speed.nbytes)
        input_bindings['goal_heading'] = cuda.mem_alloc(self.goal_heading.nbytes)
        input_bindings['action_input'] = cuda.mem_alloc(self.action.nbytes)
        input_bindings['history_input'] = cuda.mem_alloc(self.history.nbytes)
        input_bindings['sample_input'] = cuda.mem_alloc(self.sample.nbytes)
        # Allocate device memory for outputs
        output_bindings['action_output'] = cuda.mem_alloc(self.action.nbytes)
        output_bindings['history_output'] = cuda.mem_alloc(self.history.nbytes)
        output_bindings['sample_output'] = cuda.mem_alloc(self.sample.nbytes)

        # Copy inputs to device
        cuda.memcpy_htod(input_bindings['image'], self.camera_image)
        cuda.memcpy_htod(input_bindings['route'], self.route_vectors)
        cuda.memcpy_htod(input_bindings['speed'], self.ego_speed)
        cuda.memcpy_htod(input_bindings['goal_heading'], self.goal_heading)
        cuda.memcpy_htod(input_bindings['action_input'], self.action)
        cuda.memcpy_htod(input_bindings['history_input'], self.history)
        cuda.memcpy_htod(input_bindings['sample_input'], self.sample)

        # Create bindings array based on the engine's expected order
        engine = self.runtime_context.engine
        num_bindings = engine.num_io_tensors
        bindings = [None] * num_bindings

        # Map tensor names to their corresponding allocated memory
        for i in range(num_bindings):
            name = engine.get_tensor_name(i)
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if is_input:
                if name in input_bindings:
                    bindings[i] = int(input_bindings[name])
                else:
                    self.get_logger().error(f"Unknown input tensor name: {name}")
            else:
                if name in output_bindings:
                    bindings[i] = int(output_bindings[name])
                else:
                    self.get_logger().error(f"Unknown output tensor name: {name}")

        # Run inference
        self.runtime_context.execute_v2(bindings)

        # Copy outputs back to host
        cuda.memcpy_dtoh(self.action, output_bindings['action_output'])
        cuda.memcpy_dtoh(self.history, output_bindings['history_output'])
        cuda.memcpy_dtoh(self.sample, output_bindings['sample_output'])

    def _onnx_inference(self):
        # Prepare input dictionary for ONNX Runtime
        input_dict = {
            'image':
                self.camera_image.reshape(1, 1, self.camera_image.shape[0],
                                          self.camera_image.shape[1], self.camera_image.shape[2]),
            'route':
                self.route_vectors.reshape(1, 1, self.route_vectors.shape[0],
                                           self.route_vectors.shape[1]),
            'speed':
                np.array([[[self.ego_speed]]], dtype=np.float32),
            'goal_heading':
                self.goal_heading.reshape(1, 1),
            'action_input':
                self.action.reshape(1, 6),
            'history_input':
                self.history,
            'sample_input':
                self.sample
        }

        # Run inference
        outputs = self.onnx_session.run(['action_output', 'history_output', 'sample_output'],
                                        input_dict)

        # Update state with outputs
        self.action = outputs[0][0][0]
        self.history = outputs[1]
        self.sample = outputs[2]

    def publish_action(self):
        cmd_vel = Twist()
        cmd_vel.linear.x = float(self.action[0])
        cmd_vel.angular.z = float(self.action[5])
        self.cmd_publisher.publish(cmd_vel)

    def publish_route(self):
        route = Path()
        route.header.frame_id = ROBOT_FRAME
        route.header.stamp = self.get_clock().now().to_msg()
        for idx in range(len(self.route_vectors)):
            path_pose = PoseStamped()
            path_pose.header = route.header
            path_pose.pose.position.x = float(self.route_vectors[idx][0])
            path_pose.pose.position.y = float(self.route_vectors[idx][1])
            route.poses.append(path_pose)
        self.path_publisher.publish(route)

    def process_image_msg(self, image_msg):
        image_channels = int(image_msg.step / image_msg.width)
        image = np.array(image_msg.data).reshape(
            (image_msg.height, image_msg.width, image_channels))
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.ascontiguousarray(image)


def main(args=None):
    rclpy.init(args=args)
    compass_navigator = CompassNavigator()
    rclpy.spin(compass_navigator)


if __name__ == '__main__':
    main()
