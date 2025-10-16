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

import bisect
import os

import h5py
import gin
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


@gin.configurable
class RLSpecialistDataModule(pl.LightningDataModule):
    '''Datamodule with dataset collected from Isaac Sim.
    '''

    def __init__(self, dataset_path: str, batch_size: int, sequence_length: int, num_workers: int):
        super().__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_workers = num_workers
        self.dataset_path = dataset_path
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = RLSpecialistDataset(os.path.join(self.dataset_path, 'train'),
                                                     self.sequence_length)
            self.val_dataset = RLSpecialistDataset(os.path.join(self.dataset_path, 'val'),
                                                   self.sequence_length)
        if stage == 'test' or stage is None:
            self.test_dataset = RLSpecialistDataset(os.path.join(self.dataset_path, 'test'),
                                                    self.sequence_length)

    def train_dataloader(self):
        train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True,
                          sampler=train_sampler)

    def val_dataloader(self):
        val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True,
                          sampler=val_sampler)

    def test_dataloader(self):
        test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True,
                          sampler=test_sampler)

    def load_test_data(self):
        """ Load test data. This function is used in local Jupyter testing environment only.
        """
        test_dataset = RLSpecialistDataset(os.path.join(self.dataset_path, 'test'),
                                           self.sequence_length)
        return DataLoader(
            test_dataset,
            batch_size=16,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )


class RLSpecialistDataset(Dataset):
    '''Dataset from Isaac sim.
    '''

    def __init__(self, dataset_path: str, sequence_length: int):
        super().__init__()
        self.sequence_length = sequence_length
        self.rl_batch_size = 0
        self.rl_num_steps = 0
        self.hdfs = []
        self.accumulated_sample_sizes = []
        self.num_samples = 0

        # Iterate each embodiment in the dataset.
        for embodiment in os.listdir(dataset_path):
            embodiment_path = os.path.join(dataset_path, embodiment)
            # Iterate the sorted runs for the given scenario.
            dataset_files = [
                run_file for run_file in os.listdir(embodiment_path) if run_file.endswith('h5')
            ]
            dataset_files = sorted(dataset_files)
            with tqdm(total=len(dataset_files),
                      desc=f"Loading data from {embodiment_path}",
                      unit="file") as pbar:
                for dataset_file in dataset_files:
                    hdf = h5py.File(os.path.join(embodiment_path, dataset_file), 'r')
                    self.hdfs.append(hdf)
                    self.rl_batch_size = hdf['image'].shape[0]
                    self.rl_num_steps = hdf['image'].shape[1]
                    self.accumulated_sample_sizes.append(self.num_samples)
                    self.num_samples += (self.rl_num_steps //
                                         self.sequence_length) * self.rl_batch_size
                    pbar.update(1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Get the cooresponding hdf.
        hdf_idx = bisect.bisect_left(self.accumulated_sample_sizes, index + 1) - 1
        hdf_sample_idx = index - self.accumulated_sample_sizes[hdf_idx]
        return self._get_elements_batch(self.hdfs[hdf_idx], hdf_sample_idx)

    def _get_elements_batch(self, hdf, index):
        samples_per_batch = self.rl_num_steps // self.sequence_length
        batch_idx = index // samples_per_batch
        step_idx = (index % samples_per_batch) * self.sequence_length
        elements_batch = {}
        elements_batch['image'] = torch.tensor(
            hdf['image'][batch_idx, step_idx:step_idx + self.sequence_length, :]).permute(
                0, 3, 1, 2) / 255.0
        elements_batch['route'] = torch.tensor(hdf['route'][batch_idx, step_idx:step_idx +
                                                            self.sequence_length, :])
        elements_batch['speed'] = torch.tensor(hdf['speed'][batch_idx, step_idx:step_idx +
                                                            self.sequence_length, :])
        elements_batch['action'] = torch.tensor(hdf['action'][batch_idx, step_idx:step_idx +
                                                              self.sequence_length, :])
        elements_batch['policy_state'] = torch.tensor(
            hdf['policy_state'][batch_idx, step_idx:step_idx + self.sequence_length, :])

        # Get embodiment information.
        elements_batch['embodiment'] = hdf.attrs['embodiment']
        elements_batch['action_sigma'] = torch.tensor(hdf['action_sigma']).repeat(
            self.sequence_length, 1)
        return elements_batch
