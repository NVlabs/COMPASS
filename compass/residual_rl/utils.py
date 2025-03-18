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

import torch
import torch.nn.functional as F


@torch.jit.script
def preprocess_depth_images(batched_raw_depth_image: torch.Tensor,
                            target_height: int = 224,
                            target_width: int = 224):
    """ Preporcess the raw depth images, including
      1) replace the inf elements with the maximum depth value.
      2) resize the image to the target size.
      3) replicate the single channel to create a 3-channel tensor for resnet.
      4) normalize the tensor using ImageNet statistics.

      It will raise ValueError if the raw image has NaN.
  """

    if torch.isnan(batched_raw_depth_image).any():
        raise ValueError("Input tensor contains NaN values.")

    # Preprocess for infinity values.
    if torch.isinf(batched_raw_depth_image).any():
        batched_raw_depth_image[torch.isinf(batched_raw_depth_image)] = -1
        depth_max = batched_raw_depth_image.max()
        # Sanity check to make sure we don't get negative depth value.
        depth_max = torch.where(depth_max < 0, torch.tensor(1.0), depth_max)
        batched_raw_depth_image[batched_raw_depth_image == -1] = depth_max

    # Resize using interpolate
    resized_tensors = F.interpolate(batched_raw_depth_image,
                                    size=(target_height, target_width),
                                    mode='bilinear',
                                    align_corners=False)

    # Normalize
    # Compute minimum and maximum values
    depth_min = resized_tensors.min()
    depth_max = resized_tensors.max()

    # Check for the case where all elements are the same
    if depth_max == depth_min:
        # Return a zero tensor if no variation is present
        return torch.zeros_like(resized_tensors).expand(-1, 3, -1, -1)

    # Replicate the single channel to create a 3-channel tensor for resnet encoding.
    resized_tensors = resized_tensors.expand(-1, 3, -1, -1)
    # Normalize each image in the batch using ImageNet statistics
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(batched_raw_depth_image.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(batched_raw_depth_image.device)

    epsilon = 1e-6
    resized_tensors = (resized_tensors - mean) / (std + epsilon)

    return resized_tensors
