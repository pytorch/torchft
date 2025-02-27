# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
LocalSGD
=========
This module implements a fault tolerant version of LocalSGD and related methods.
"""
import logging
from types import TracebackType
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Type

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer
from torch.utils.hooks import RemovableHandle

from torchft.manager import Manager

logger: logging.Logger = logging.getLogger(__name__)


class LocalSGD:
    """
    LocalSGD is a context manager that
    implements the algorithm described in https://arxiv.org/pdf/1805.09767

    This will synchronize the model parameters periodically in a fault tolerant
    way using a torchft Manager. The allreduce on the parameters will happen
    every sync_every steps after the optimizer.step call.

    To implement safe and fault tolerant, this requires a backup copy of the
    weights. By default these are stored in CPU memory. If any error occurs
    during the LocalSGD step, the step will be discarded and the model
    parameters will reset back to the last time LocalSGD synchronized.

    The backup weights could be eliminated by relaxing the guarantee of exactly
    `sync_every` steps but that would diverge from the LocalSGD algorithm.
    DiLoCo also needs this backup copy to compute the delta.

    The torchft quorum is computed at the beginning of ``sync_every`` steps. If
    any error occurs, or a worker fails between syncs, ``sync_every`` steps will be
    discarded and a new quorum will be computed on the next step.

    If running in async mode, on a joining worker the first ``sync_every`` steps
    will discarded as the model will be recovering during that period. When
    using sync mode, the checkpoint will be restored prior to the first step.

    TODO: add a way via Manager to detect workers failing early for shrink only
    TODO: add DiLoCo support
    """

    def __init__(
        self,
        manager: Manager,
        model: nn.Module,
        optimizer: optim.Optimizer,
        sync_every: int,
        backup_device: Optional[torch.device] = None,
        pin_memory: bool = True,
    ) -> None:
        """
        Args:
            manager: The manager to use.
            model: The model to wrap.
            optimizer: The optimizer used by the model.
            sync_every: How often to sync the model weights.
            backup_device: The device to store the backup of the model parameters on. (default cpu)
            pin_memory: Whether to pin the memory used for the backup of the model parameters.
        """
        super().__init__()
        self._manager = manager
        self._model = model
        self._local_optimizer = optimizer
        self._local_step = 0
        self._sync_every = sync_every
        assert sync_every >= 1, "sync_every must be greater than or equal to 1"
        device = backup_device or torch.device("cpu")
        self._backup_parameters: Dict[str, torch.Tensor] = {}
        for name, p in self._model.named_parameters():
            t = torch.empty(*tuple(p.shape), dtype=p.dtype, device=device)
            if (
                pin_memory
                and t.device == torch.device("cpu")
                and torch.cuda.is_available()
            ):
                t = t.pin_memory()
            self._backup_parameters[name] = t

        self._hooks: List[RemovableHandle] = []
        # Need to copy the parameters to the host to be safe if we are on the first step.
        self._save_parameters()

    def __enter__(self) -> "LocalSGD":
        # Add optimizer hook which increments the local step counter and syncs if necessary
        self._hooks.append(
            self._local_optimizer.register_step_post_hook(self._step_post_hook)
        )
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        # Handle any cleanup or error handling here
        if exc_type is not None:
            # If an exception occurred, restore parameters
            self._restore_parameters()
        # Clean up hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        return False  # Propagate exceptions

    def _save_parameters(self) -> None:
        with torch.no_grad():
            # TODO: consider running copy on a separate stream
            for name, p in self._model.named_parameters():
                self._backup_parameters[name].copy_(p.data, non_blocking=True)

    def _restore_parameters(self) -> None:
        with torch.no_grad():
            # TODO: consider running copy on a separate stream
            for name, p in self._model.named_parameters():
                p.data.copy_(self._backup_parameters[name], non_blocking=False)

    def _step_post_hook(
        self, _optim: optim.Optimizer, _args: List[object], _kwargs: Dict[str, object]
    ) -> None:
        """
        This hook is registered on the optimizer and is called after the optimizer step.
        """
        self._local_step += 1
        if self._local_step >= self._sync_every:
            self.sync()

    def sync(self) -> None:
        """
        Synchronizes and averages the model weights across the manager.
        """
        self._manager.start_quorum()
        self._perform_sync()
        self._local_step = 0

    def _perform_sync(self) -> None:
        """
        Performs the synchronization of the model weights across the manager.
        This method is intended to be overridden by subclasses to implement custom
        synchronization logic.
        """
        self._average()
        if self._manager.should_commit():
            self._save_parameters()
        else:
            # commit failed, restore from the backup parameters
            self._restore_parameters()

    def _average(self) -> None:
        # TODO: do we need to broadcast buffers like DDP does?

        works = []

        for p in self._model.parameters():
            # TODO: bucketize parameters
            works.append(self._manager.allreduce(p.data.detach()))

        for work in works:
            work.wait()


class DiLoCo(LocalSGD):
    """
    DiLoCo is a subclass of LocalSGD that overrides the synchronization
    mechanism to average and synchronize the pseudogradients (delta of the previous global weight and current local weights).

    diloco: https://arxiv.org/pdf/2311.08105
    """

    bucket_cap_mb = 32 * 1024 * 1024
    use_bucketization = False

    def __init__(
        self,
        manager: Manager,
        model: nn.Module,
        inner_optimizer: optim.Optimizer,
        outer_optimizer: optim.Optimizer,
        sync_every: int,
        backup_device: Optional[torch.device] = None,
        pin_memory: bool = True,
        use_bucketization: bool = False,
        bucket_cap_mb: int = None,
    ) -> None:
        if manager._use_async_quorum:
            raise ValueError(
                "Using DiLoCo require synchronous quorum to be enabled. "
                "Ensure that the manager is initialized with use_async_quorum=False"
            )
        super().__init__(
            manager, model, inner_optimizer, sync_every, backup_device, pin_memory
        )
        self._outer_optimizer = outer_optimizer
        if bucket_cap_mb is not None:
            self.bucket_cap_mb = int(bucket_cap_mb * 1024 * 1024)

        self.use_bucketization = use_bucketization

    def _perform_sync(self) -> None:
        """
        Overrides the sync method to calculate the pseugradient, average them across the manager group, and
        step using the outer optimizer.
        """

        # Set the .grad field of each parameter to its pseudogradient
        for name, p in self._model.named_parameters():
            assert name in self._backup_parameters
            pseudogradient = p.data - self._backup_parameters[name]
            p.grad = pseudogradient

        self._average_grads()
        # Restore the parameters back to the previous state
        self._restore_parameters()

        if self._manager.should_commit():
            # Use the outer optimizer to update the model parameters
            self._outer_optimizer.step()
            self._save_parameters()
        self._outer_optimizer.zero_grad()

    def _average_grads(self) -> None:
        """
        Efficiently averages gradients across the group using either:
        - Per-parameter allreduce (old behavior)
        - Bucketized allreduce (new behavior)
        """
        if self.use_bucketization:
            self._allreduce_bucketized()
        else:
            self._allreduce_per_param()

    def _allreduce_per_param(self) -> None:
        """Performs allreduce on each gradient tensor separately (original method)."""
        works = []
        for p in self._model.parameters():
            if p.grad is None:
                continue
            work = self._manager.allreduce(p.grad)
            works.append(work)

        for work in works:
            work.wait()

    def _allreduce_bucketized(self) -> None:
        """
        Averages gradients using bucketized allreduce with a fixed 32MB buffer.
        """

        grads = [p.grad for p in self._model.parameters() if p.grad is not None]
        if not grads:
            return

        # Compute total size and allocate a flat buffer
        total_size = sum(g.numel() for g in grads)
        dtype, device = grads[0].dtype, grads[0].device

        # Process in fixed 32MB chunks
        offset = 0
        while offset < total_size:
            # Compute chunk size
            chunk_size = min(
                self.BUCKET_SIZE_BYTES // grads[0].element_size(), total_size - offset
            )

            flat_buffer = torch.zeros(chunk_size, dtype=dtype, device=device)

            # Pack gradients into buffer
            pack_offset, bucket_tensors = 0, []
            for g in grads:
                numel = g.numel()
                if pack_offset + numel > chunk_size:
                    break
                flat_buffer[pack_offset : pack_offset + numel].copy_(g.view(-1))
                bucket_tensors.append((g, pack_offset, numel))
                pack_offset += numel

            work = self._manager.allreduce(flat_buffer)
            work.wait()

            for g, pack_offset, numel in bucket_tensors:
                g.copy_(flat_buffer[pack_offset : pack_offset + numel].view_as(g))

            offset += chunk_size  # Move to next chunk
