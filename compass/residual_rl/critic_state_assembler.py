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

from typing import Dict

import gin
import torch
from torch import nn
from torchvision import models

from compass.residual_rl.constants import INPUT_IMAGE_SIZE
from compass.residual_rl.utils import preprocess_depth_images


@gin.configurable
class DepthImageFeatureExtractor(nn.Module):
    """
    Feature extractor for depth images using pretrained ResNet.
    Assumes input images are three channels (via replication for example)
    and preprocessed to 224x224.

    Note: while the input_size can be different, we currently restrict it to
     this fixed size only as the ResNet is trained with this size. It is a
     common practice and widely used.
    """

    def __init__(self):
        super().__init__()
        # Initialize a pre-trained ResNet
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.input_size = [3, 224, 224]

        # Use the model's layers up to a certain point for feature extraction
        # Remove the fully connected layers, retain convolutional layers
        modules = list(resnet.children())[:-2]
        self.feature_extractor = nn.Sequential(*modules)

        # The output shape of the feature extractor is 512 x 7 x 7, we choose
        # output_size of the average pool as 1 to reduce the feature dimension.
        output_size = (1, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((output_size))

        # Freeze all parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.out_channels = modules[-1][-1].conv2.out_channels * output_size[0] * output_size[1]

    def forward(self, x):
        # Forward pass through feature extractor
        features = self.feature_extractor(x)

        # Apply global average pooling
        pooled_features = self.global_avg_pool(features)

        pooled_features = torch.flatten(pooled_features, start_dim=1)
        if pooled_features.numel() == 0:
            raise ValueError("Feature tensor is empty. Check input size and network layers.")
        return pooled_features


@gin.configurable
class CriticObservationEncoder(nn.Module):
    '''Encoder of the input observations
        Args:
            depth_encoder: encoder of the depth image
            route_feat_dim: feature dimension of the route encoder
            speed_feat_dim: feature dimension of the speed encoder

        Inputs:
            batch (Dict): dict of the input tensors:
                depth_image: (b, 1, h, w)
                speed_feat: (b, out_channels), encoded speed feature
                route_feat: (b, out_channels), encoded route feature

        Returns:
            embedding: A dict of 1D embedding of the observations
    '''

    def __init__(self, depth_encoder: nn.Module, route_feat_dim: int, speed_feat_dim: int):
        super().__init__()

        # Depth Image
        self.depth_encoder = depth_encoder()
        features_channels = self.depth_encoder.out_channels

        # Add dims from pretrained route encoder and speed encoder
        features_channels += route_feat_dim
        features_channels += speed_feat_dim

        self.embedding_dim = features_channels

    def forward(self, batch: Dict) -> torch.Tensor:
        # Only support one step for now.
        depth_image = batch['depth_image']

        # Image encoding
        depth_feat = self.depth_encoder(depth_image)

        speed_feat = batch["speed_feat"]
        route_feat = batch["route_feat"]

        # Final observation embedding.
        embedding = torch.cat([depth_feat, speed_feat, route_feat], dim=1)

        # Check dimension matches
        assert embedding.shape[1] == self.embedding_dim

        return embedding


@gin.configurable
class CriticStateAssemblerBase:
    """Abstract base class to assemble the critic state.

    The goal is to provide a universal interface type for different critic setups.
    """

    # pylint: disable=unused-argument
    def __init__(self, env=None, encoder=None, policy_state_dim=None):
        self._state_dim = -1

    @property
    def state_dim(self) -> int:
        return self._state_dim

    def __str__(self) -> str:
        raise NotImplementedError

    def compute_critic_state(self, policy_state=None, obs_dict=None, extras=None) -> torch.Tensor:
        raise NotImplementedError


@gin.configurable
class CriticStateDepthEncoderAssembler(CriticStateAssemblerBase):
    """ Use the encoded depth, speed and route feature for critic state.

    """

    def __init__(self, env=None, encoder=None, policy_state_dim=None):
        super().__init__(env, encoder, policy_state_dim)
        self.encoder = encoder
        self._state_dim = encoder.embedding_dim

    def __str__(self) -> str:
        return "CriticStateDepthEncoderAssembler"

    def compute_critic_state(self, policy_state=None, obs_dict=None, extras=None):
        # Preprocess depth image
        input_size = self.encoder.depth_encoder.input_size
        raw_depth_image = obs_dict["privileged"]["camera_depth_img"]
        b = raw_depth_image.shape[0]
        depth_image_processed = raw_depth_image.reshape(b, 1, INPUT_IMAGE_SIZE[0],
                                                        INPUT_IMAGE_SIZE[1]).float()

        depth_image_processed = preprocess_depth_images(depth_image_processed,
                                                        target_height=input_size[1],
                                                        target_width=input_size[2])

        critic_batch = {}
        critic_batch.update(extras)
        critic_batch['depth_image'] = depth_image_processed
        return self.encoder(critic_batch)


@gin.configurable
class CriticStateSymmetricAssembler(CriticStateAssemblerBase):
    """Use the policy state as the critic state for symmetric A2C.

    """

    def __init__(self, env=None, encoder=None, policy_state_dim=None):
        super().__init__(env, encoder, policy_state_dim)
        self._state_dim = policy_state_dim

    def __str__(self) -> str:
        return "CriticStateSymmetricAssembler"

    def compute_critic_state(self, policy_state=None, obs_dict=None, extras=None):
        return policy_state
