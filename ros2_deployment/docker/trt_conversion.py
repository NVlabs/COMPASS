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

import argparse

import tensorrt as trt


def convert(onnx_path, trt_path):
    # Create a TensorRT builder and network with explicit batch flag
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    network = builder.create_network(explicit_batch)

    # Parse the ONNX model
    parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
    with open(onnx_path, 'rb') as onnx:
        if not parser.parse(onnx.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError("Failed to parse ONNX model!")

    # Configure builder options
    builder_config = builder.create_builder_config()

    # Build TensorRT engine
    engine = builder.build_serialized_network(network, builder_config)

    # Save the serialized TensorRT engine
    with open(trt_path, 'wb') as f:
        f.write(engine)


def main():
    # Parse the arguments.
    parser = argparse.ArgumentParser(description='Convert the X-Mobility ONNX to TRT engine.')
    parser.add_argument('--onnx-path',
                        '-o',
                        type=str,
                        required=True,
                        help='The path to the input onnx file.')
    parser.add_argument('--trt-path',
                        '-t',
                        type=str,
                        required=True,
                        help='The path to the output TRT engine.')

    args = parser.parse_args()

    # Run the conversion.
    convert(args.onnx_path, args.trt_path)


if __name__ == '__main__':
    main()
