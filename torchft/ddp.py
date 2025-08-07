# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Distributed Data Parallel
==========================

This module implements a DistributedDataParallel wrapper that works with the
Manager to provide fault tolerance.
"""

import os
import sys
from typing import TYPE_CHECKING, Optional, cast
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.algorithms.join import Joinable
from torch.nn import parallel

from torchft.process_group import ProcessGroup, ProcessGroupDummy, ProcessGroupGloo

if TYPE_CHECKING:
    from torchft.manager import Manager, _ManagedFuture


class DistributedDataParallel(parallel.DistributedDataParallel):
    """
    This is a patched DistributedDataParallel implementation that makes it
    compatible with torchft.

    Important notes:

    * This requires states to be synced on step 0 using an external mechanism
      rather than an internal broadcast (torchft.Manager will do this).
    * Using non-basic features of the DDP may cause your model to catch fire as
      they haven't been tested with torchft.
    * This doesn't any sanity checks such as verifying parameter sizes are the
      same across workers.
    """

    def __init__(self, manager: "Manager", module: nn.Module, **kwargs: object) -> None:
        # use a dummy PG to soak up the init all reduce, actual comms will go
        # through the comm_hook.
        pg = ProcessGroupDummy(0, 1)

        super().__init__(
            module,
            process_group=pg,
            # HACK: This forces the reducer to never rebuild buckets.
            # The reducer normally rebuilds the buckets after the first training
            # step which can improve performance but is incompatible with
            # torchft as it will cause the buckets to diverge for recovering
            # replicas.
            find_unused_parameters=True,
            # pyre-fixme[6]: got object
            **kwargs,
        )

        self.register_comm_hook(manager, self._comm_hook)

    @staticmethod
    def _comm_hook(
        state: "Manager", bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        work = state.allreduce(bucket.buffer())
        work.wait()
        fut = work.get_future()

        # We need to return the underlying future here otherwise
        # this can hang
        fut = cast("_ManagedFuture[torch.Tensor]", fut)
        assert fut._fut
        return fut._fut


class PureDistributedDataParallel(nn.Module):
    """
    A pure Python reimplementation of the DDP wrapper.

    We recommend using DistributedDataParallel instead of this class.

    This calls one allreduce per gradient tensor and doesn't use a reducer. This
    may be very slow for real models.
    """

    def __init__(self, manager: "Manager", module: nn.Module) -> None:
        super().__init__()

        self.module = module

        def post_grad_hook(p: torch.Tensor) -> None:
            if p.grad is not None:
                manager.allreduce(p.grad)

        for p in module.parameters():
            p.register_post_accumulate_grad_hook(post_grad_hook)

    def forward(self, *args: object) -> object:
        return self.module(*args)
