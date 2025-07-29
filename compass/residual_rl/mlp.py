# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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
