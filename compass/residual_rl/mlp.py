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

from torch import nn


def build_mlp_network(input_dim, hidden_dims, output_dim, activation_fn):
    """
    Builds a MLP feed-forward neural network.

    Args:
        input_dim (int): Dimension of the input layer.
        hidden_dims (list of int): Dimensions of the hidden layers.
        output_dim (int): Dimension of the output layer.
        activation_fn (nn.Module): Activation function to use between layers.

    Returns:
        nn.Sequential: A sequential container of layers representing the network.
    """
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dims[0]))
    layers.append(activation_fn)

    for i in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        layers.append(activation_fn)

    layers.append(nn.Linear(hidden_dims[-1], output_dim))

    return nn.Sequential(*layers)


def get_activation(act_name, inplace: bool = True):
    if act_name == "elu":
        return nn.ELU(inplace)
    elif act_name == "selu":
        return nn.SELU(inplace)
    elif act_name == "relu":
        return nn.ReLU(inplace)
    elif act_name == "crelu":
        return nn.CReLU(inplace)
    elif act_name == "lrelu":
        return nn.LeakyReLU(inplace)
    elif act_name == "tanh":
        return nn.Tanh(inplace)
    elif act_name == "sigmoid":
        return nn.Sigmoid(inplace)
    else:
        print("invalid activation function!")
        return None
