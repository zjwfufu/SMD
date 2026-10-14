# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from smd.utils.distributed import get_rank, world_size


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: torch.utils.data.Dataset, shuffle: bool = True, seed: int = 0, start_idx: int = 0):
        """
        Sampler that generates an infinite stream of indices from a dataset.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to sample from
            shuffle (bool): Whether to shuffle the dataset
            seed (int): The seed for the random number generator
            start_idx (int): The start index for the sampler
        """
        assert len(dataset) > 0

        super().__init__()
        self.dataset_size = len(dataset)
        self.num_replicas = world_size()
        self.shuffle = shuffle
        self.seed = seed

        # Initialize start index specific to this rank
        self.idx = start_idx + get_rank()

    def __iter__(self):
        idx = self.idx
        epoch = None

        while True:
            # Determine which epoch this index belongs to
            current_epoch = idx // self.dataset_size

            if epoch != current_epoch:
                epoch = current_epoch
                order = np.arange(self.dataset_size)

                if self.shuffle:
                    local_seed = (self.seed + epoch) % (2**32 - 1)
                    np.random.RandomState(local_seed).shuffle(order)

            # Yield the item at the current shuffled position
            return_idx = int(order[idx % self.dataset_size])
            yield return_idx

            # Stride forward by the number of replicas
            idx += self.num_replicas
