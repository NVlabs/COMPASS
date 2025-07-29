# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import gin
import torch
import torch.nn.functional as F
from torch import nn

from model.x_mobility.utils import pack_sequence_dim

from compass.residual_rl.mlp import build_mlp_network, get_activation

ACTION_MASK = (0, 1, 5)


@gin.configurable
class EmbodimentOneHotEncoder(nn.Module):
    '''Embodiment one-hot encoding.'''

    def __init__(self,
                 num_dims: int = 8,
                 supported_types: list[str] = ['carter', 'h1', 'spot', 'g1']):
        super().__init__()
        self.num_dims = num_dims
        self.supported_types = supported_types
        assert self.num_dims >= len(self.supported_types), \
            "Number of dimensions less than the number of supported types"

    def forward(self, embodiment_type: str):
        if embodiment_type not in self.supported_types:
            raise ValueError(f"Embodiment type '{embodiment_type}' not supported. "
                             f"Must be one of {self.supported_types}")

        # Create one-hot encoding
        index = self.supported_types.index(embodiment_type)
        encoding = torch.zeros(self.num_dims)
        encoding[index] = 1.0
        return encoding


@gin.configurable
class MLPActionPolicy(nn.Module):
    '''MLP action policy.'''

    def __init__(self, in_channels: int, command_n_channels: int):
        super().__init__()
        self.command_fc = build_mlp_network(input_dim=in_channels,
                                            hidden_dims=[1024, 512, 256, 128],
                                            output_dim=command_n_channels,
                                            activation_fn=get_activation('relu', False))

    def forward(self, x):
        return {'mean': self.command_fc(x)}


@gin.configurable
class MLPActionPolicyDistribution(MLPActionPolicy):

    def __init__(self, in_channels, command_n_channels, init_noise_std: float = 0.1):
        super().__init__(in_channels, command_n_channels)

        # Global std variance for each action dim (trainable)
        self.std = nn.Parameter(init_noise_std * torch.ones(command_n_channels))

    def forward(self, x):
        return {'mean': self.command_fc(x), 'std': self.std}


@gin.configurable
class ESDistillationKLLoss(nn.Module):
    """
    Compute KL(teacher || student) for each sample.
    Returns average KL across the batch.
    Ref: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians # pylint: disable=line-too-long
    """

    def forward(self, output, batch):
        losses = {}
        mu_teacher = pack_sequence_dim(batch['action'])[:, ACTION_MASK]
        std_teacher = pack_sequence_dim(batch['action_sigma'])[:, ACTION_MASK]
        mu_student = output['mean'][:, ACTION_MASK]
        std_student = output['std'].repeat(std_teacher.shape[0], 1)[:, ACTION_MASK]

        # Convert std to log_std
        log_std_teacher = torch.log(std_teacher)
        log_std_student = torch.log(std_student)

        # Term1: log(\sigma_S / \sigma_T)
        term1 = log_std_student - log_std_teacher

        # Term2: (sigma_T^2 + (mu_T - mu_S)^2) / (2 sigma_S^2)
        numerator = std_teacher.pow(2) + (mu_teacher - mu_student).pow(2)
        denominator = 2.0 * std_student.pow(2)
        term2 = numerator / denominator

        # KL = term1 + term2 - 0.5
        kl = term1 + term2 - 0.5
        # Average across actions, then across batch
        losses['action_kl'] = kl.mean()
        return losses


@gin.configurable
class ESDistillationMSELoss(nn.Module):
    '''Loss function for action policy distillation.'''

    def __init__(self):
        super().__init__()
        self.loss_fn = F.mse_loss

    def forward(self, output, batch):
        losses = {}
        mse = F.mse_loss(output['mean'][:, ACTION_MASK],
                         pack_sequence_dim(batch['action'])[:, ACTION_MASK],
                         reduction='none')
        losses['action'] = torch.sum(mse, dim=-1, keepdims=True).mean()
        return losses


@gin.configurable
class ESDistillationPolicy(nn.Module):
    '''Embodiment specialist distillation policy.'''

    def __init__(self, policy_model: nn.Module, embodiment_encoder: nn.Module,
                 policy_state_dim: int, command_n_channels: int):
        super().__init__()
        self.embodiment_encoder = embodiment_encoder()
        self.policy_model = policy_model(in_channels=policy_state_dim +
                                         self.embodiment_encoder.num_dims,
                                         command_n_channels=command_n_channels)

    def forward(self, batch):
        policy_state = batch['policy_state']
        embodiment_encodings = [
            self.embodiment_encoder(embodiment).repeat(policy_state.shape[1], 1)
            for embodiment in batch['embodiment']
        ]
        embodiment_encodings = torch.stack(embodiment_encodings, dim=0).to(policy_state.device)
        x = pack_sequence_dim(torch.cat([policy_state, embodiment_encodings], dim=-1))
        return self.policy_model(x)


class ESDistillationPolicyWrapper(nn.Module):
    '''Wrapper of the distilled policy to enable inference.'''

    def __init__(self, distillation_policy_ckpt_path: str, embodiment_type: str):
        super().__init__()
        # Load the checkpoint and remove the prefix if any.
        state_dict = torch.load(distillation_policy_ckpt_path, weights_only=False)['state_dict']
        state_dict = {k.removeprefix('model.'): v for k, v in state_dict.items()}
        # Load the state dict.
        self.model = ESDistillationPolicy()
        self.model.load_state_dict(state_dict, strict=True)

        # Embodiment type.
        self.embodiment_type = embodiment_type

    def forward(self, policy_state):
        batch = {
            'policy_state': policy_state.unsqueeze(1),
            'embodiment': [self.embodiment_type] * policy_state.shape[0]
        }
        return self.model(batch)['mean']
