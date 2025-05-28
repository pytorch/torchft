# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data
====

This module provides helper classes to implement fault tolerant data loaders.

We recommend using torchdata's StatefulDataLoader to checkpoint each replica's
dataloader frequently to avoid duplicate batches.
"""

from typing import Optional

import torch.distributed as dist
from torch.utils import data


# pyre-fixme[24]: expected generic parameter
class DistributedSampler(data.distributed.DistributedSampler):
    """
    DistributedSampler extends the standard PyTorch DistributedSampler with a
    `num_replica_groups` that is used to shard the data across the fault
    tolerance replica groups.

    torchft doesn't know how many replica groups ahead of time so we need to set
    this to be the max number.

    This sampler is inherently lossy when used with torchft. torchft
    occasionally drops batches on rejoining and if a replica group is down that
    group examples will never be used. This can lead to imbalances if using a
    small dataset.

    This will shard the input dataset into ``num_replicas*num_replica_group``
    number of shards.

    Each shard rank is calculated via: ``rank + num_replicas*replica_group_id``

    num_replicas and replica_group_id must be the same on all workers.
    """

    def __init__(
        self,
        dataset: data.Dataset,
        replica_group_id: int,
        num_replica_groups: int,
        group_rank: Optional[int] = None,
        num_replicas: Optional[int] = None,
        **kwargs: object,
    ) -> None:
        """
        Args:
            data: the dataset to use
            replica_group_id: the group ID (0-num_replica_groups) to use for this shard of data.
            num_replica_groups: the max number of global replica groups
            rank: the local group rank
            num_replicas: the local group world size
        """
        if group_rank is None:
            group_rank = dist.get_rank()
        if num_replicas is None:
            num_replicas = dist.get_world_size()

        self.global_rank: int = group_rank + num_replicas * replica_group_id
        self.global_world_size: int = num_replicas * num_replica_groups

        super().__init__(
            dataset,
            rank=self.global_rank,
            num_replicas=self.global_world_size,
            # pyre-fixme[6]: got object
            **kwargs,
        )
