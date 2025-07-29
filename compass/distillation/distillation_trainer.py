# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import gin
import pytorch_lightning as pl
import torch

# pylint: disable=unused-import
from compass.distillation.distillation import (MLPActionPolicy, MLPActionPolicyDistribution,
                                               ESDistillationKLLoss, ESDistillationMSELoss)


@gin.configurable
class ESDistillationTrainer(pl.LightningModule):
    '''Pytorch Lightnining module of ES distillation for training.
    '''

    def __init__(self, model, loss, lr):
        super().__init__()
        self.lr = lr
        # Save hyperparameters.
        self.save_hyperparameters()

        # Model
        self.model = model()

        # Losses
        self.loss = loss()

    def forward(self, batch):
        return self.model(batch)

    def shared_step(self, batch):
        output = self.forward(batch)
        losses = self.loss(output, batch)
        return losses, output

    def training_step(self, batch, batch_idx):
        losses, output = self.shared_step(batch)
        self.log_and_visualize(batch, output, losses, batch_idx, prefix='train')
        return self.loss_reducing(losses)

    def validation_step(self, batch, batch_idx):
        losses, output = self.shared_step(batch)
        self.log_and_visualize(batch, output, losses, batch_idx, prefix='validation')
        val_loss = self.loss_reducing(losses)
        return {'val_loss': val_loss}

    def test_step(self, batch, batch_idx):
        losses, output = self.shared_step(batch)
        self.log_and_visualize(batch, output, losses, batch_idx, prefix='test')

    def log_and_visualize(self, batch, output, losses, batch_idx, prefix='train'):    #pylint: disable=unused-argument
        # Log losses
        for key, value in losses.items():
            self.log(f'{prefix}/losses/{key}', value, sync_dist=True)

        # Log total loss and videos for validation.
        if prefix == 'validation':
            # Log total validation loss.
            self.log('val_loss', self.loss_reducing(losses), sync_dist=True, on_epoch=True)

    def loss_reducing(self, loss: torch.Tensor):
        total_loss = sum([x for x in loss.values()])
        return total_loss

    def configure_optimizers(self):
        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        # scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'epoch'}]
