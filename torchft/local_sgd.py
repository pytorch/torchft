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
import math
import threading
from contextlib import nullcontext
from types import TracebackType
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.distributed.tensor import DTensor
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer
from torch.utils.hooks import RemovableHandle

from torchft.manager import Manager

logger: logging.Logger = logging.getLogger(__name__)


def extract_local_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    Returns a cloned version of the input tensor. If the input tensor is a DTensor,
    it extracts and clones its local representation.
    """
    new_tensor = None
    if isinstance(t, DTensor):
        new_tensor = t.to_local().clone()
    else:
        new_tensor = t.clone()
    new_tensor.grad = None
    return new_tensor


class LocalSGD:
    """
    LocalSGD is a context manager that
    implements the algorithm described in https://arxiv.org/pdf/1805.09767

    This will synchronize the model parameters periodically in a fault tolerant
    way using a torchft Manager. The allreduce on the parameters will happen
    every sync_every steps after the optimizer.step call.

    The torchft quorum is computed at the beginning of ``sync_every`` steps. If
    any error occurs, or a worker fails between syncs, ``sync_every`` steps will be
    discarded and a new quorum will be computed on the next step.

    If running in async mode, on a joining worker the first ``sync_every`` steps
    will discarded as the model will be recovering during that period. When
    using sync mode, the checkpoint will be restored prior to the first step.
    """

    def __init__(
        self,
        manager: Manager,
        model: nn.Module,
        optimizer: optim.Optimizer,
        sync_every: int,
    ) -> None:
        """
        Args:
            manager: The manager to use.
            model: The model to wrap.
            optimizer: The optimizer used by the model.
            sync_every: How often to sync the model weights.
        """
        super().__init__()
        self._manager = manager
        self._model = model
        self._local_optimizer = optimizer
        self._local_step = 0
        self._sync_every = sync_every
        assert sync_every >= 1, "sync_every must be greater than or equal to 1"

        self._hooks: List[RemovableHandle] = []

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
        # Clean up hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        return False  # Propagate exceptions

    def _step_post_hook(
        self, _optim: optim.Optimizer, _args: Tuple[Any, ...], _kwargs: Dict[str, Any]
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
        """
        averaged_parameters = self._average()
        if self._manager.should_commit():
            # Update the model parameters with the averaged values
            for param, avg_param in zip(self._model.parameters(), averaged_parameters):
                if isinstance(param, DTensor):
                    # we averaged the local version of the tensor so need to copy it back as a DTensor
                    param.data.copy_(
                        DTensor.from_local(
                            avg_param,
                            param.device_mesh,
                            param.placements,
                            shape=param.shape,
                            stride=param.stride(),
                        )
                    )
                else:
                    param.data.copy_(avg_param)

    def _average(self) -> list[torch.Tensor]:
        """
        Averages the model parameters across the manager and returns the averaged parameters.
        """
        works = []
        averaged_parameters = []
        for p in self._model.parameters():
            # Create a new tensor to store the averaged parameter
            avg_param = extract_local_tensor(p)
            works.append(self._manager.allreduce(avg_param))
            averaged_parameters.append(avg_param)
        for work in works:
            work.wait()
        return averaged_parameters


class _StreamingDiLoCoFragment:
    bucket_cap_mb: int = 32 * 1024 * 1024
    use_bucketization: bool = False

    def __init__(
        self,
        manager: Manager,
        model_fragment: nn.Module,
        fragment_sync_offset: int,
        inner_optimizer: optim.Optimizer,
        outer_optimizer: optim.Optimizer,
        sync_every: int,
        backup_device: Optional[torch.device] = None,
        pin_memory: bool = True,
        use_bucketization: bool = False,
        bucket_cap_mb: Optional[int] = None,
        should_quantize: bool = False,
        fragment_sync_delay: int = 0,
        fragment_update_alpha: float = 0.0,
    ) -> None:
        if fragment_sync_offset > sync_every:
            raise ValueError("Fragment must be synced once before `sync_every` steps")

        self._manager = manager
        self._model_fragment = model_fragment
        self._fragment_sync_offset = fragment_sync_offset
        self._local_optimizer = inner_optimizer
        self._sync_every = sync_every
        assert sync_every >= 1, "sync_every must be greater than or equal to 1"
        self._backup_device = backup_device
        self._pin_memory = pin_memory
        self._fragment_sync_delay = fragment_sync_delay
        self._fragment_update_alpha = fragment_update_alpha

        self._outer_optimizer = outer_optimizer

        # Stores pending all reduce
        self._allreduce_futures: list[torch.futures.Future[torch.Tensor]] = []
        self._stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )

        # Recorded on `_stream` to wait for allreduce to finish
        self._stop_event: Optional[torch.cuda.Event] = None

        if bucket_cap_mb is not None:
            self.bucket_cap_mb = int(bucket_cap_mb * 1024 * 1024)

        self.use_bucketization = use_bucketization
        self.should_quantize = should_quantize

        self._grads: Dict[str, torch.Tensor] = {}

        # Used to save global parameters so that they can be restored in case
        # commit fails
        self.original_parameters: Dict[str, torch.Tensor] = {}

        # Used to mix the local and global parameters
        self._local_parameters: Dict[str, torch.Tensor] = {}

        for name, p in self._model_fragment.named_parameters():
            if isinstance(p, DTensor):
                p = extract_local_tensor(p.data)

            backup_device = self._backup_device or torch.device("cpu")
            t = torch.empty(*tuple(p.shape), dtype=p.dtype, device=backup_device)
            if (
                self._pin_memory
                and t.device == torch.device("cpu")
                and torch.cuda.is_available()
            ):
                t = t.pin_memory()
            self.original_parameters[name] = t

    @torch.profiler.record_function("torchft::local_sgd::save_parameters")
    def save_parameters(self) -> None:
        with torch.no_grad():
            # TODO: consider running copy on a separate stream
            for name, p in self._model_fragment.named_parameters():
                param_to_local = extract_local_tensor(p.data)
                self.original_parameters[name].copy_(param_to_local, non_blocking=True)

    def _save_local_parameters(self) -> None:
        """
        Saves a copy of the model's parameters.
        """
        with torch.no_grad():
            for name, p in self._model_fragment.named_parameters():
                self._local_parameters[name] = extract_local_tensor(p.data)

    @torch.profiler.record_function("torchft::local_sgd::restore_parameters")
    def restore_parameters(self) -> None:
        with torch.no_grad():
            # TODO: consider running copy on a separate stream
            for name, p in self._model_fragment.named_parameters():
                if isinstance(p, DTensor):
                    # we averaged the local version of the tensor so need to copy it back as a DTensor
                    p.data.copy_(
                        DTensor.from_local(
                            self.original_parameters[name],
                            p.device_mesh,
                            p.placements,
                            shape=p.shape,
                            stride=p.stride(),
                        ),
                        non_blocking=False,
                    )
                else:
                    p.data.copy_(self.original_parameters[name], non_blocking=False)

    def _save_grads(self) -> None:
        """
        Saves pseudo-gradients of the parameters
        """
        with torch.no_grad():
            for name, p in self._model_fragment.named_parameters():
                if isinstance(p, DTensor):
                    local_param = p.to_local()
                else:
                    local_param = p
                pseudogradient = local_param - self.original_parameters[name].to(
                    p.device
                )
                self._grads[name] = pseudogradient

    def _set_grads(self) -> None:
        """
        Sets the gradients of the model fragment from the allreduce result
        """
        with torch.no_grad():
            for name, p in self._model_fragment.named_parameters():
                # avoid copying the gradient, it should be on the same device
                if isinstance(p, DTensor):
                    p.grad = DTensor.from_local(
                        self._grads[name],
                        p.device_mesh,
                        p.placements,
                        shape=p.shape,
                        stride=p.stride(),
                    )
                else:
                    p.grad = self._grads[name]

                # No longer needed
                del self._grads[name]

    def _clear_local_parameters(self) -> None:
        """
        Clears the saved copy of the model's parameters
        """
        self._local_parameters = {}

    def _merge_parameters(self) -> None:
        """
        Merges the local and global parameters.
        """
        for name, p in self._model_fragment.named_parameters():
            torch.lerp(
                p.data, self._local_parameters[name], 1 - self._fragment_update_alpha
            )

    @torch.profiler.record_function("torchft::local_sgd::wait")
    def wait(self) -> None:
        """
        Waits for the previously scheduled allreduce to finish
        """
        if len(self._allreduce_futures) == 0:
            return

        if self._stream is not None:
            assert self._stop_event is not None
            self._stop_event.synchronize()
            self._stop_event = None

        self._allreduce_futures = []

    def should_prepare_fragment(self, step: int) -> bool:
        """
        Determines if the fragment should be asynchronously sent to other replicas
        """
        step_to_prepare = step - self._fragment_sync_offset
        return step_to_prepare % self._sync_every == 0

    def should_sync_fragment(self, step: int) -> bool:
        """
        Determines if the fragment should be synchronized with other replicas
        """
        step_to_sync = step - self._fragment_sync_offset - self._fragment_sync_delay
        return step_to_sync % self._sync_every == 0

    @torch.profiler.record_function("torchft::local_sgd::prepare_sync")
    def prepare_sync(self) -> None:
        """
        Calculate the pseugradient, average them across the manager group and starts
        allreduce on the pseudo-gradients but doesn't wait for it to finish.
        """
        self._save_grads()

        assert len(self._allreduce_futures) == 0

        # Make sure tensors are available to `_stream`
        if self._stream is not None:
            self._stream.wait_stream(torch.cuda.current_stream())

        with (
            torch.cuda.stream(self._stream)
            if self._stream is not None
            else nullcontext()
        ):
            self._average_grads()

            for work in self._allreduce_futures:
                work.wait()

            if self._stream is not None:
                self._stop_event = torch.cuda.Event()
                self._stop_event.record()

    @torch.profiler.record_function("torchft::local_sgd::perform_sync")
    def perform_sync(self) -> bool:
        """
        Overrides the sync method to wait for the scheduled allreduce to finish and
        steps using the outer optimizer.
        """
        # Waiting for an allreduce before it has been sent is currently not supported.
        # Please make sure to not do this to avoid running into inconsistencies.
        #
        # This can happen when using large values of `fragment_sync_delay`.
        # The node might not have participated in syncing of this fragment.
        #
        # The allreduce for other nodes who did might actually
        # succeed and in that case, we shouldn't allow recovery
        # from this node.
        #
        # We do need to increase the `max_step` here so we
        # don't end up in an infinite loop of needing to recover
        # but we can't let other nodes recover from this node
        # because it doesn't have the latest state.
        #
        # We can add a `is_catching_up` flag to the state_dict
        # to disallow recoveries from this node. Such nodes can
        # be excluded from `max_step` calculation unless all
        # nodes are catching up. This approach makes the replica state
        # of global parameters diverge though. So we could add recovery
        # for a particular fragment from a peer node as a part of the
        # `should_commit` or next `quorum` when a node is catching up.
        assert len(self._allreduce_futures) > 0

        self.wait()

        # save the parameters so they can be used for merging
        self._save_local_parameters()
        # Restore the parameters back to the previous state
        self.restore_parameters()

        # For large values of `fragment_sync_delay`, this call can be
        # a problem.
        #
        # This can return success even if the allreduce failed. Because
        # the process group could have been reconfigured while the
        # allreduce was inflight. The inflight allreduce may or may
        # not have been aborted.
        #
        # We can track errors per allreduce to
        # let the commit fail here. But this has the downside of
        # reconfiguring the pg too many times resulting in
        # more aborts and more commit failures.
        should_commit = self._manager.should_commit()

        if should_commit:
            # Use the outer optimizer to update the model parameters
            self._set_grads()
            self._outer_optimizer.step()
            self.save_parameters()
            self._merge_parameters()
        self._outer_optimizer.zero_grad()

        # free up memory
        self._clear_local_parameters()

        return should_commit

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
        for name, p in self._model_fragment.named_parameters():
            # Perform allreduce on the pseudogradients
            work = self._manager.allreduce(
                self._grads[name], should_quantize=self.should_quantize
            )
            self._allreduce_futures.append(work)

    def _bucketize_and_allreduce(
        self,
        tensors: List[torch.Tensor],
        bucket_size_bytes: int,
    ) -> None:
        """
        Applies allreduce on a list of tensors using bucketization.

        Args:
            tensors: List of torch tensors (e.g., gradients).
            bucket_size_bytes: Max size of each bucket in bytes.
        """
        if not tensors:
            return

        total_size = sum(t.numel() for t in tensors)
        dtype, device = tensors[0].dtype, tensors[0].device

        offset = 0
        flat_index = 0
        while offset < total_size:
            chunk_size = min(
                bucket_size_bytes // tensors[0].element_size(), total_size - offset
            )
            flat_buffer: torch.Tensor = torch.zeros(
                chunk_size, dtype=dtype, device=device
            )

            pack_offset: int = 0
            bucket_tensors: list[Tuple[torch.Tensor, int, int]] = []
            for t in tensors[flat_index:]:
                numel = t.numel()
                if pack_offset + numel > chunk_size:
                    break
                flat_buffer[pack_offset : pack_offset + numel].copy_(t.view(-1))
                bucket_tensors.append((t, pack_offset, numel))
                pack_offset += numel
                flat_index += 1

            work = self._manager.allreduce(
                flat_buffer, should_quantize=self.should_quantize
            )

            def callback(fut: torch.futures.Future[torch.Tensor]) -> None:
                nonlocal bucket_tensors, flat_buffer
                for t, pack_offset, numel in bucket_tensors:
                    t.copy_(flat_buffer[pack_offset : pack_offset + numel].view_as(t))

            work = work.then(callback)
            self._allreduce_futures.append(work)

            offset += chunk_size

    def _allreduce_bucketized(self) -> None:
        """
        Averages gradients using bucketized allreduce with a fixed buffer.
        """
        grads = list(self._grads.values())
        assert len(grads) > 0, "No gradients to allreduce"
        self._bucketize_and_allreduce(
            grads,
            bucket_size_bytes=self.bucket_cap_mb,
        )


class DiLoCo:
    """
    DiLoCo is a subclass of LocalSGD that overrides the synchronization
    mechanism to average and synchronize the pseudogradients (delta of the previous global weight and current local weights).

    The class implements a more general version of DiLoco, Streaming DiLoCo,
    which synchronizes fragments of pseudogradients at different steps.

    This algorithm requires a backup copy of the
    weights. By default these are stored in CPU memory. If any error occurs
    during the DiLoCo step, the step will be discarded and the model
    parameters will reset back to the last time DiLoCo synchronized.

    DiLoCo paper: https://arxiv.org/pdf/2311.08105
    Streaming DiLoCo paper: https://arxiv.org/pdf/2501.18512
    """

    def __init__(
        self,
        manager: Manager,
        model_fragments: List[nn.Module],
        inner_optimizer: optim.Optimizer,
        outer_optimizer: optim.Optimizer,
        sync_every: int,
        backup_device: Optional[torch.device] = None,
        pin_memory: bool = True,
        use_bucketization: bool = False,
        bucket_cap_mb: Optional[int] = None,
        should_quantize: bool = False,
        fragment_sync_delay: int = 0,
        fragment_update_alpha: float = 0.0,
    ) -> None:
        """
        Args:
            manager: The manager to use.
            model_fragments: The fragments of the model to wrap.
            inner_optimizer: The optimizer used for the local parameters every step.
            outer_optimizer: The optimizer used for the global parameters updated every "sync_every" steps.
            sync_every: How often to update the model weights.
            backup_device: The device to store the backup weights on. If None, the backup weights will be on CPU.
            pin_memory: Whether to pin the memory for the backup weights (only for CPU device).
            should_quantize: Whether to quantize the gradients before allreduce.
            fragment_sync_delay: Controls the number of inner steps to wait before blocking on a fragment's
                                 synchronization. This is the "tao" parameter in the Streaming DiLoCo paper.
            fragment_update_alpha: Determines how to mix the local and global optimized parameters
        """

        if manager._use_async_quorum:
            raise ValueError(
                "Using DiLoCo require synchronous quorum to be enabled. "
                "Ensure that the manager is initialized with use_async_quorum=False"
            )

        if sync_every < len(model_fragments):
            raise ValueError("Only 1 fragment can be syncrhonized at a time")

        if fragment_sync_delay >= sync_every:
            raise ValueError(
                "Fragment must be synced before it is reduced another time"
            )

        if fragment_update_alpha < 0 or fragment_update_alpha > 1:
            raise ValueError("fragment_update_alpha must be between 0 and 1")

        super().__init__()
        self._manager = manager

        # Protects `_local_step`
        self._lock = threading.Lock()

        # The number of training iterations performed.
        # Used to synchronize which fragment to send across all
        # replicas
        self._local_step = 0

        # Sync `_local_step` with other replicas
        self._manager.register_state_dict_fn(
            "local_step",
            self._load_step,
            lambda: self._local_step,
        )

        # Used to perform quorum before any training happens
        self._should_recover = True

        self._hooks: List[RemovableHandle] = []

        self._local_optimizer = inner_optimizer

        self._fragments: List[_StreamingDiLoCoFragment] = [
            _StreamingDiLoCoFragment(
                manager,
                model_fragment,
                math.floor((sync_every / len(model_fragments)) * (i + 1)),
                inner_optimizer,
                # TODO: Support different outer optimizers for each fragment
                outer_optimizer,
                sync_every,
                backup_device,
                pin_memory,
                use_bucketization,
                bucket_cap_mb,
                should_quantize,
                fragment_sync_delay,
                fragment_update_alpha,
            )
            for i, model_fragment in enumerate(model_fragments)
        ]

        # This is to make sure we adhere to the assumptions made by the
        # `_StreamingDiLoCoFragment` about the fragment sync schedule.
        assert fragment_sync_delay < sync_every // len(model_fragments)

        # Used to ensure that we try to sync a fragment after we've sent a prepare for it
        self._first_prepare_sent: set[int] = set()

        # Need to copy the parameters to the host to be safe if we are on the first step.
        self._save_parameters()

    def _load_step(self, step: int) -> None:
        with self._lock:
            self._local_step = step

    def _save_parameters(self) -> None:
        for fragment in self._fragments:
            fragment.save_parameters()

    def _restore_parameters(self) -> None:
        for fragment in self._fragments:
            fragment.restore_parameters()

    def __enter__(self) -> "DiLoCo":
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
        # Clean up hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        return False  # Propagate exceptions

    def _wait(self) -> None:
        """
        Waits for allreduce to finish on all fragments
        """
        for fragment in self._fragments:
            fragment.wait()

        self._first_prepare_sent.clear()

    def _quorum_loop(self) -> None:
        """
        Performs infinite retries until quorum is successfull
        """
        while True:
            self._manager.start_quorum()

            if self._manager.errored() is None:
                return

    def _step_post_hook(
        self, _optim: optim.Optimizer, _args: Tuple[Any, ...], _kwargs: Dict[str, Any]
    ) -> None:
        """
        This hook is registered on the optimizer and is called after the optimizer step.
        """
        if self._should_recover:
            # Get the correct step when. This will continue after other committed.
            self._quorum_loop()
            self._should_recover = False
            # This is to be consistent with the nodes that are not recovering. They
            # proceed with the code below on the step after quorum completes.
            return

        # We need to make sure all nodes send the same fragments in order.
        # This is to avoid deadlocking e.g.
        #
        # 1. Step 1 - Node A sends fragment 1
        # 2. Step 1 - Node B sends fragment 2
        # 3. Step 2 - Node A waits for fragment 1
        # 4. Step 2 - Node B waits for fragment 2
        #
        # Both of them will fail because Node A didn't send fragment 2
        # and Node B didn't send fragment 1.
        with self._lock:
            self._local_step += 1
            step = self._local_step

        # Start sending fragments
        for i, fragment in enumerate(self._fragments):
            if not fragment.should_prepare_fragment(step):
                continue

            logger.debug(f"preparing fragment {i} at step {step}")

            self._first_prepare_sent.add(i)
            fragment.prepare_sync()

        for i, fragment in enumerate(self._fragments):
            if not fragment.should_sync_fragment(step):
                continue

            # We need to have sent an allreduce before we can syncing
            # a fragment
            if i not in self._first_prepare_sent:
                continue

            logger.debug(f"syncing fragment {i} at step {step}")

            if not fragment.perform_sync():
                # Cancel all the previously scheduled allreduce by simply
                # waiting for them. They should have failed but lets be
                # paranoid anyway.
                #
                # We could choose to resend the failed fragments but that is
                # more complicated since it involves coordinating all nodes to
                # rewind and resend the fragments.
                self._wait()

                # Reset the local step. This is needed in case manager `should_commit` fails.
                #
                # This is because there can be a node that has the same `max_step` as the
                # nodes that reached the commit point. However, this node failed before
                # it could reach the commit point. So the local steps for these two nodes
                # are not the same. But either can be used for recovery.
                #
                # To make sure both return the same step, we just reset the step to 0
                # and start from scratch.
                #
                # In the happy path, we don't need to reset the step because --
                # Nodes participating in the commit bumped their `max_step`.
                # Any new nodes will take `local_step` from one of these nodes, which must
                # be the same across all nodes because they took the same number of steps
                # since the last commit to get to the most recent commit.
                with self._lock:
                    self._local_step = 0

            # Avoid doing allreduce after quorum failed.
            #
            # Maybe a different quorum formed without this node, so this node
            # will incorrectly try to allreduce potentially on an incorrect
            # fragment because the local_step is also out of sync.
            # The replica will need recovery later anyway.
            #
            # So in case it didn't crash (e.g. network errors), we can save some
            # training data by looping here. Otherwise that training data goes to
            # waste after recovery
            self._quorum_loop()

            # TODO: Since we do quorum after commit, there might be a big gap until
            # the next allreduce. This increases the chances of nodes failing
            # and so the allreduce to fail.
            # - We could maybe do a quorum again right before preparing for a fragment
            #   using `shrink_only`. This might make it tricky for new nodes to join
            #   though.
            # - Maintain a sequence number in the state dict that gets bumped at every
            #   quorum call. Then we can do a quorum right before allreduce and avoid
            #   doing quorums after commit.

            # We need to set make sure `_local_step` is still
            # the same across all replicas if `quorum_id` changed.
            #
            # We can't garuntee a majority of replicas in this new quorum
            # has the latest `max_step`.
            #
            # TODO: This is garuntee is currently lacking
            # in torchft unless `shrink_only` is set.
            #
            # After the quorum though, everyone will have the same
            # `local_step` because replicas with the chosen
            # `max_step` will have the same `local_step`. That is
            # because we don't take additional steps after commit.
