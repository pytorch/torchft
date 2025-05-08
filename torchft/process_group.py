# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Process Groups
=========================

This module implements fault tolerant process groups that can be reconfigured
and resized at runtime.

These extend the standard PyTorch ProcessGroup API and can be used in most
places that would accept a standard process group. As these can change size at
runtime users need to take care to not assume a static rank or world size.
"""

import logging
import os
import threading
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing.connection import Connection
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# pyre-fixme[21]: no attribute ProcessGroupGloo
from torch.distributed import (
    DeviceMesh,
    PrefixStore,
    ProcessGroup as BaseProcessGroup,
    ProcessGroupGloo as BaseProcessGroupGloo,
    Store,
    TCPStore,
)
from torch.distributed.distributed_c10d import (
    AllgatherOptions,
    AllreduceCoalescedOptions,
    AllreduceOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    ReduceOp,
    ReduceScatterOptions,
    Work,
)
from torch.futures import Future
from torch.utils._pytree import tree_any

# We import these for backwards compatibility
from torchft.device_mesh import *  # noqa: F401
from torchft.futures import context_timeout, stream_timeout
from torchft.multiprocessing import _MonitoredPipe

if TYPE_CHECKING:
    from torchft.manager import Manager

logger: logging.Logger = logging.getLogger(__name__)

# TODO: use non strings which are cheaper
_QUEUE_CLOSE = "queue_close"
_FUTURE_RESULT = "fut_result"
_FUTURE_EXCEPTION = "fut_exception"


T = TypeVar("T")


def create_store_client(store_addr: str, timeout: timedelta) -> Store:
    """
    Creates a PrefixStore(TCPStore(...)) client from an address in the format:

    host:port/prefix

    Ex: localhost:1234/my/prefix
    """
    host, _, rest = store_addr.partition(":")
    port, _, prefix = rest.partition("/")

    store = TCPStore(
        host_name=host,
        port=int(port),
        is_master=False,
        wait_for_workers=False,
        timeout=timeout,
    )
    store = PrefixStore(prefix, store)
    return store


class ProcessGroup(BaseProcessGroup):
    def __init__(self, *args: object, **kwargs: object) -> None:
        # pyre-fixme[6]: got object
        super().__init__(*args, **kwargs)

        self._group_name: Optional[str] = None

    # pyre-fixme[14]: inconsistent override
    def allgather(
        self,
        output_tensors: List[List[torch.Tensor]],
        input_tensor: List[torch.Tensor],
        opts: AllgatherOptions,
    ) -> Work:
        """
        Gathers tensors from the whole group in a list.

        See torch.distributed.all_gather for more details.
        """
        raise NotImplementedError("not implemented")

    # pyre-fixme[14]: inconsistent override
    def allgather_into_tensor_coalesced(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[torch.Tensor],
        opts: AllgatherOptions,
    ) -> Work:
        """
        Performs an allgather operation on coalesced tensors.

        See torch.distributed.allgather_coalesced for more details.
        """
        raise NotImplementedError("not implemented")

    # pyre-fixme[14]: inconsistent override
    def allreduce(
        self,
        tensors: List[torch.Tensor],
        opts: Union[AllreduceOptions, ReduceOp],
    ) -> Work:
        """
        Reduces the tensor data across all machines in such a way that all get the final result.

        See torch.distributed.all_reduce for more details.
        """
        raise NotImplementedError("not implemented")

    def allreduce_coalesced(
        self,
        tensors: List[torch.Tensor],
        opts: AllreduceCoalescedOptions,
    ) -> Work:
        """
        Performs an all_reduce operation in a coalesced manner.

        See torch.distributed.all_reduce_coalesced for more details.
        """
        raise NotImplementedError("not implemented")

    # pyre-fixme[14]: inconsistent override
    def alltoall_base(
        self,
        output_buffer: torch.Tensor,
        input_buffer: torch.Tensor,
        output_split_sizes: List[int],
        input_split_sizes: List[int],
        opts: AllToAllOptions,
    ) -> Work:
        """
        Performs an all_to_all operation.

        See torch.distributed.all_to_all_single for more details.
        """
        raise NotImplementedError("not implemented")

    def barrier(self, opts: BarrierOptions) -> Work:
        """
        Synchronizes all processes.

        See torch.distributed.barrier for more details.
        """
        raise NotImplementedError("not implemented")

    # pyre-fixme[14]: inconsistent override
    def broadcast(
        self, tensor_list: List[torch.Tensor], opts: BroadcastOptions
    ) -> Work:
        """
        Broadcasts the tensor to the whole group.

        See torch.distributed.broadcast for more details.
        """
        raise NotImplementedError("not implemented")

    def broadcast_one(self, tensor: torch.Tensor, root: int) -> Work:
        opts = BroadcastOptions()
        opts.rootRank = root
        return self.broadcast([tensor], opts)

    # pyre-fixme[14]: inconsistent override
    def recv(self, tensors: List[torch.Tensor], src_rank: int, tag: int) -> Work:
        """
        Receives a list of tensors from the process with rank `rank`.

        See torch.distributed.recv for more details.
        """
        raise NotImplementedError("not implemented")

    # pyre-fixme[14]: inconsistent override
    def reduce_scatter(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[List[torch.Tensor]],
        opts: ReduceScatterOptions,
    ) -> Work:
        """
        Reduces, then scatters a list of tensors to all processes in a group.

        See torch.distributed.reduce_scatter for more details.
        """
        raise NotImplementedError("not implemented")

    # pyre-fixme[14]: inconsistent override
    def reduce_scatter_tensor_coalesced(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[torch.Tensor],
        opts: ReduceScatterOptions,
    ) -> Work:
        """
        Performs a reduce-scatter operation on coalesced tensors.

        See torch.distributed.reduce_scatter_tensor for more details.
        """
        raise NotImplementedError("not implemented")

    # pyre-fixme[14]: inconsistent override
    def send(self, tensors: List[torch.Tensor], dst_rank: int, tag: int) -> Work:
        """
        Sends a list of tensors to the process with rank `dst_rank`.

        See torch.distributed.send for more details.
        """
        raise NotImplementedError("not implemented")

    def configure(self, store_addr: str, rank: int, world_size: int) -> None:
        """
        This reconfigures the ProcessGroup to use a new store, rank and world size.

        Every time this is called it must be provided with a unique prefixed
        store address. I.e. localhost:1234/my/prefix/1

        This function will block until the underlying ProcessGroup is created.
        If an error occurs this will throw.

        Args:
            store_addr: address of the store to use
            rank: rank of this process
            world_size: world size of this process group
        """
        raise NotImplementedError("not implemented")

    def size(self) -> int:
        raise NotImplementedError("not implemented")

    def getBackendName(self) -> str:
        raise NotImplementedError("not implemented")

    def _register(self, name: str) -> str:
        group_name = f"{self.getBackendName()}:{name}"

        # This is needed for DeviceMesh and functional collectives to work.
        # Resizable worlds don't fit well into DeviceMesh so we register a world
        # size 1 PG.

        def create_pg(
            prefix_store: PrefixStore, rank: int, world_size: int, timeout: float
        ) -> ProcessGroup:
            return self

        if torch.cuda.is_available():
            devices = ["cuda", "cpu"]
        else:
            devices = ["cpu"]
        dist.Backend.register_backend(group_name, create_pg, devices=devices)

        return group_name

    def register(self, name: str) -> "ProcessGroup":
        """
        Registers the process group with the global registry. This enables usage
        with things like functional_collectives which are compilable.

        This should only be called once.

        Args:
            name: name must be a unique name for this process group
        """

        group_name = self._register(name)

        return dist.new_group(
            ranks=[dist.get_rank()],
            backend=group_name,
            group_desc=group_name,
            timeout=timedelta(seconds=60.0),  # this timeout isn't used
        )

    @property
    def group_name(self) -> str:
        if self._group_name is None:
            raise ValueError("ProcessGroup name not set")
        return self._group_name

    def _set_group_name(self, name: str) -> None:
        self._group_name = name

    def unregister(self) -> None:
        """
        Unregisters the process group with the global registry.

        Must be registered first.
        """
        dist.destroy_process_group(self)

    def abort(self) -> None:
        """
        Aborts the process group.
        """
        pass

    def shutdown(self) -> None:
        """
        Shuts down the process group.
        """
        pass

    def errored(self) -> Optional[Exception]:
        """
        Whether an async error occured that requires reconfiguration.
        """
        return None

    def set_timeout(self, timeout: timedelta) -> None:
        """
        Sets the default timeout for the process group.
        """
        raise NotImplementedError("set_timeout not implemented")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ProcessGroupWrapper(ProcessGroup):
    """
    This is a wrapper around any ProcessGroup with a reconfiguration method.

    Args:
        timeout: timeout for reconfiguration for TCPStore
        pg: optional ProcessGroup to use, if None a new one will be created
    """

    def __init__(
        self,
        timeout: timedelta = timedelta(seconds=60),
        pg: Optional[ProcessGroup] = None,
    ) -> None:
        super().__init__(0, 1)
        self._pg: Optional[BaseProcessGroup] = pg
        self._timeout = timeout

    def configure(self, store_addr: str, rank: int, world_size: int) -> None:
        pg = self._pg
        if isinstance(pg, ProcessGroup):
            pg.configure(store_addr, rank, world_size)
            return

        # abort if already initialized
        self.abort()

        store = create_store_client(store_addr, timeout=self._timeout)

        self._pg = self._create_pg(store, rank, world_size)

    def abort(self) -> None:
        pg = self._pg
        if pg is not None:
            if hasattr(pg, "abort"):
                pg.abort()
            else:
                try:
                    backend = pg._get_backend(torch.device("cuda"))
                except RuntimeError:
                    backend = None
                if backend is not None and hasattr(backend, "abort"):
                    backend.abort()

            self._pg = None

    def shutdown(self) -> None:
        # TODO: abort PG if possible
        self._pg = None

    def _create_pg(self, store: Store, rank: int, world_size: int) -> BaseProcessGroup:
        raise NotImplementedError("not implemented")

    def _wrap_work(self, work: Work, opts: object) -> Work:
        return work

    def _opts_hook(self, opts: T) -> T:
        return opts

    @contextmanager
    def _run_context(self) -> Generator[None, None, None]:
        yield

    def set_timeout(self, timeout: timedelta) -> None:
        self._timeout = timeout

    def allgather(
        self,
        output_tensors: List[List[torch.Tensor]],
        input_tensor: List[torch.Tensor],
        opts: AllgatherOptions,
    ) -> Work:
        with self._run_context():
            return self._wrap_work(
                self.parent.allgather(
                    output_tensors, input_tensor, self._opts_hook(opts)
                ),
                opts,
            )

    def allgather_into_tensor_coalesced(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[torch.Tensor],
        opts: AllgatherOptions,
    ) -> Work:
        with self._run_context():
            return self._wrap_work(
                self.parent.allgather_into_tensor_coalesced(
                    output_tensors, input_tensors, self._opts_hook(opts)
                ),
                opts,
            )

    def allreduce(self, tensors: List[torch.Tensor], opts: object) -> Work:
        with self._run_context():
            return self._wrap_work(
                self.parent.allreduce(tensors, self._opts_hook(opts)), opts
            )

    def allreduce_coalesced(
        self, tensors: List[torch.Tensor], opts: Union[AllreduceOptions, ReduceOp]
    ) -> Work:
        with self._run_context():
            return self._wrap_work(
                self.parent.allreduce_coalesced(tensors, self._opts_hook(opts)), opts
            )

    def alltoall_base(
        self,
        output_buffer: torch.Tensor,
        input_buffer: torch.Tensor,
        output_split_sizes: List[int],
        input_split_sizes: List[int],
        opts: AllToAllOptions,
    ) -> Work:
        with self._run_context():
            return self._wrap_work(
                self.parent.alltoall_base(
                    output_buffer,
                    input_buffer,
                    output_split_sizes,
                    input_split_sizes,
                    self._opts_hook(opts),
                ),
                opts,
            )

    def barrier(self, opts: BarrierOptions) -> Work:
        with self._run_context():
            return self._wrap_work(self.parent.barrier(self._opts_hook(opts)), opts)

    def broadcast(self, tensor_list: List[torch.Tensor], opts: object) -> Work:
        with self._run_context():
            return self._wrap_work(
                self.parent.broadcast(tensor_list, self._opts_hook(opts)), opts
            )

    def recv(self, tensors: List[torch.Tensor], src_rank: int, tag: int) -> Work:
        with self._run_context():
            return self._wrap_work(self.parent.recv(tensors, src_rank, tag), None)

    def reduce_scatter(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[List[torch.Tensor]],
        opts: object,
    ) -> Work:
        with self._run_context():
            return self._wrap_work(
                self.parent.reduce_scatter(
                    output_tensors, input_tensors, self._opts_hook(opts)
                ),
                opts,
            )

    def reduce_scatter_tensor_coalesced(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[torch.Tensor],
        opts: ReduceScatterOptions,
    ) -> Work:
        with self._run_context():
            return self._wrap_work(
                self.parent.reduce_scatter_tensor_coalesced(
                    output_tensors, input_tensors, self._opts_hook(opts)
                ),
                opts,
            )

    def send(self, tensors: List[torch.Tensor], dst_rank: int, tag: int) -> Work:
        with self._run_context():
            return self._wrap_work(self.parent.send(tensors, dst_rank, tag), None)

    def size(self) -> int:
        return self.parent.size()

    @property
    def parent(self) -> BaseProcessGroup:
        assert self._pg is not None, "process group not initialized"
        return self._pg

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pg={self._pg})"


class ProcessGroupGloo(ProcessGroupWrapper):
    """
    This is a reconfigurable version of ProcessGroupGloo.
    """

    def _create_pg(self, store: Store, rank: int, world_size: int) -> BaseProcessGroup:
        pg = BaseProcessGroup(store, rank, world_size)
        pg._set_default_backend(ProcessGroup.BackendType.GLOO)
        # pyre-fixme[16]: no attribute ProcessGroupGloo
        backend_class = BaseProcessGroupGloo(store, rank, world_size, self._timeout)
        backend_class._set_sequence_number_for_group()
        pg._register_backend(
            torch.device("cpu"), ProcessGroup.BackendType.GLOO, backend_class
        )
        if torch.cuda.is_available():
            pg._register_backend(
                torch.device("cuda"), ProcessGroup.BackendType.GLOO, backend_class
            )
        return pg

    def getBackendName(self) -> str:
        return "torchft-gloo"

    # pyre-fixme[14,15]: inconsistent override
    def reduce_scatter(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[List[torch.Tensor]],
        opts: ReduceScatterOptions,
    ) -> None:
        """
        This function is a placeholder for the reduce_scatter operation in the
        ProcessGroupGloo class. However, this operation is not supported by the
        Gloo backend, and thus, calling this function will raise a
        RuntimeError.

        Raises:
            RuntimeError: Always raised since reduce_scatter is not
            supported by ProcessGroupGloo.
        """
        raise RuntimeError("ProcessGroupGloo does not support reduce_scatter.")

    # pyre-fixme[15]: inconsistent override
    def reduce_scatter_tensor_coalesced(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[torch.Tensor],
        opts: ReduceScatterOptions,
    ) -> None:
        """
        This function is a placeholder for the reduce_scatter_tensor_coalesced
        operation in the ProcessGroupGloo class.
        However, this operation is not supported by the
        Gloo backend, and thus, calling this function will raise a
        RuntimeError.

        Raises:
            RuntimeError: Always raised since reduce_scatter is not
            supported by ProcessGroupGloo.
        """
        raise RuntimeError(
            "ProcessGroupGloo does not support reduce_scatter_tensor_coalesced."
        )


class _WorkCUDATimeout(Work):
    def __init__(self, pg: ProcessGroup, work: Work, timeout: timedelta) -> None:
        super().__init__()
        self._pg = pg
        self._work = work
        self._timeout = timeout

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        async_timeout = timeout or self._timeout
        with self._stream_timeout(self._pg, async_timeout):
            # In newer versions of PyTorch work may not exist if the call was
            # not async. In these cases we can just schedule the stream timeout
            # and return.
            if self._work is not None:
                if not self._work.wait():
                    return False

            # Always use cuda stream for timeout to avoid ProcessGroupNCCL
            # watchdog firing and crashing the process.
            if timeout is not None:
                torch.cuda.synchronize()

            return True

    @classmethod
    @contextmanager
    def _stream_timeout(
        cls, pg: ProcessGroup, timeout: timedelta
    ) -> Generator[None, None, None]:
        """
        Set a timeout on the CUDA stream for the given process group.

        This does not hold a reference to self to avoid holding the work
        object/tensors longer than necessary.

        Args:
            pg: The process group to call abort on.
            timeout: The timeout to set on the CUDA stream.
        """

        def callback() -> None:
            logger.error(f"aborting after {timeout}!")
            pg.abort()

        # make sure .wait() can be cancelled if it blocks i.e. in barrier
        with context_timeout(callback, timeout):
            yield

        # Cancel work if the cuda stream doesn't complete
        stream_timeout(callback, timeout)

    def get_future(self) -> torch.futures.Future[object]:
        fut = self._work.get_future()

        def done_callback(fut: torch.futures.Future[object]) -> None:
            try:
                with self._stream_timeout(self._pg, self._timeout):
                    fut.wait()

            except Exception as e:
                logger.error(f"done callback failed: {e}")

        fut.add_done_callback(done_callback)
        return fut


class ProcessGroupNCCL(ProcessGroupWrapper):
    """
    This is a reconfigurable version of ProcessGroupNCCL.

    If you are using a supported version of NCCL (NCCL >= 2.26, torch >= 2.7)
    this will attempt to use ncclCommAbort to recover from any timeouts.

    This uses a Python user space event loop to asynchronously wait for the NCCL
    operations to complete. This should not be used with very long timeouts as
    the timeout entries are not cleaned up until the elapsed duration completes
    which may result in slowness or excess memory usage.

    WARNING: this may result in deadlocks due to NCCL error handling and on old
    versions of torch/NCCL will result in deadlocks.

    Args:
        timeout: the timeout to use for NCCL operations.
    """

    def __init__(self, timeout: timedelta = timedelta(seconds=60.0)) -> None:
        super().__init__(timeout)
        self._use_abort: bool = torch.cuda.nccl.version() >= (2, 25)

        self._errored: Optional[Exception] = None

        NONBLOCKING_TIMEOUT_ENV = "TORCH_NCCL_NONBLOCKING_TIMEOUT"
        if NONBLOCKING_TIMEOUT_ENV not in os.environ:
            warnings.warn(
                f"{NONBLOCKING_TIMEOUT_ENV} is not set, defaulting to {timeout}. "
                "If any nonblocking NCCL operations have already run this may "
                "result in the default timeout of 30 minutes and hangs on error."
            )
            os.environ[NONBLOCKING_TIMEOUT_ENV] = str(timeout.total_seconds())

    def _opts_hook(self, opts: T) -> T:
        if not self._use_abort:
            return opts

        # We need to clear the timeout to apply our own timeout that doesn't
        # crash the whole program.
        if hasattr(opts, "timeout"):
            # apply default timeout to disable
            opts.timeout = AllgatherOptions().timeout
        return opts

    def _wrap_work(self, work: Work, opts: object) -> Work:
        if not self._use_abort:
            return work

        timeout = self._timeout
        # pyre-fixme[16]: no attribute timeout
        if hasattr(opts, "timeout") and opts.timeout.total_seconds() > 0:
            timeout = opts.timeout
        return _WorkCUDATimeout(self, work, timeout)

    @contextmanager
    def _run_context(self) -> Generator[None, None, None]:
        timeout: timedelta = self._timeout

        def callback() -> None:
            logger.error(f"aborting after {timeout}!")
            self.abort()

        # when running in blocking mode we need to make sure collectives can
        # timeout
        with context_timeout(callback, timeout):
            yield

    def _create_pg(self, store: Store, rank: int, world_size: int) -> BaseProcessGroup:
        # pyre-fixme[21]: no attribute ProcessGroupNCCL
        from torch.distributed import ProcessGroupNCCL as BaseProcessGroupNCCL

        self._errored = None

        # pyre-fixme[16]: no attribute ProcessGroupNCCL
        opts = BaseProcessGroupNCCL.Options()
        opts.config.blocking = False

        pg = BaseProcessGroup(store, rank, world_size)
        pg._set_default_backend(ProcessGroup.BackendType.NCCL)
        # pyre-fixme[16]: no attribute ProcessGroupNCCL
        backend_class = BaseProcessGroupNCCL(store, rank, world_size, opts)
        backend_class._set_sequence_number_for_group()
        pg._register_backend(
            torch.device("cuda"), ProcessGroup.BackendType.NCCL, backend_class
        )
        return pg

    def abort(self) -> None:
        # We need to set the error before aborting to ensure that errored()
        # returns the error correctly when NCCL abort fires and unblocks the
        # stream.
        self._errored = RuntimeError("aborted")

        super().abort()

    def errored(self) -> Optional[Exception]:
        # force a synchronization to ensure all work is complete
        torch.cuda.synchronize()

        return self._errored

    def getBackendName(self) -> str:
        return "torchft-nccl"


class _DummyWork(dist._Work):
    def __init__(self, result: object) -> None:
        super().__init__()
        self.result_ = result
        # pyre-fixme[29]: Future is not a function
        self.future_: torch.futures.Future[object] = torch.futures.Future()
        self.future_.set_result(result)

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        return True

    def get_future(self) -> torch.futures.Future[object]:
        return self.future_


class ProcessGroupDummy(ProcessGroup):
    """
    This process group discards all data passed to it and returns success. This
    is intended for rare cases where we want to discard certain operations
    without modifying the underlying library.

    This PG only supports world_size of 1.
    """

    def __init__(self, rank: int, world: int) -> None:
        super().__init__(rank, world)
        assert rank == 0
        assert world == 1

        self._rank = rank
        self._world = world
        self.wait_count = 0
        self.get_future_count = 0
        self._work: List[Work] = []
        self.configure_count = 0

    def configure(self, store_addr: str, rank: int, world_size: int) -> None:
        self.configure_count += 1

    def allgather(
        self,
        output_tensors: List[List[torch.Tensor]],
        input_tensor: List[torch.Tensor],
        opts: object,
    ) -> Work:
        for o, i in zip(output_tensors[0], input_tensor):
            o.copy_(i)

        res = _DummyWork(output_tensors)
        self._work.append(res)
        return res

    def allgather_into_tensor_coalesced(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[torch.Tensor],
        opts: AllgatherOptions,
    ) -> Work:
        for o, i in zip(output_tensors, input_tensors):
            o.copy_(i)

        res = _DummyWork(output_tensors)
        self._work.append(res)
        return res

    def allreduce(self, tensors: List[torch.Tensor], opts: object) -> Work:
        res = _DummyWork(tensors)
        self._work.append(res)
        return res

    def allreduce_coalesced(
        self, tensors: List[torch.Tensor], opts: Union[AllreduceOptions, ReduceOp]
    ) -> Work:
        res = _DummyWork(tensors)
        self._work.append(res)
        return res

    def alltoall_base(
        self,
        output_buffer: torch.Tensor,
        input_buffer: torch.Tensor,
        output_split_sizes: List[int],
        input_split_sizes: List[int],
        opts: AllToAllOptions,
    ) -> Work:
        output_buffer.copy_(input_buffer)
        res = _DummyWork([output_buffer])
        self._work.append(res)
        return res

    def barrier(self, opts: BarrierOptions) -> Work:
        return _DummyWork(None)

    def broadcast(self, tensor_list: List[torch.Tensor], opts: object) -> Work:
        res = _DummyWork(tensor_list)
        self._work.append(res)
        return res

    def recv(self, tensors: List[torch.Tensor], src_rank: int, tag: int) -> Work:
        return _DummyWork(None)

    def reduce_scatter(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[List[torch.Tensor]],
        opts: object,
    ) -> Work:
        for o, i in zip(output_tensors, input_tensors[0]):
            o.copy_(i)

        res = _DummyWork(output_tensors)
        self._work.append(res)
        return res

    def reduce_scatter_tensor_coalesced(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[torch.Tensor],
        opts: ReduceScatterOptions,
    ) -> Work:
        for o, i in zip(output_tensors, input_tensors):
            o.copy_(i)

        res = _DummyWork(output_tensors)
        self._work.append(res)
        return res

    def send(self, tensors: List[torch.Tensor], dst_rank: int, tag: int) -> Work:
        return _DummyWork(None)

    def size(self) -> int:
        return self._world

    def getBackendName(self) -> str:
        return "torchft-dummy"


class _ErrorSwallowingWork(Work):
    def __init__(
        self,
        pg: "ErrorSwallowingProcessGroupWrapper",
        work: Work,
        default_result: object,
    ) -> None:
        super().__init__()

        self._pg = pg
        self._work = work
        self._default_result = default_result

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        try:
            self._work.wait()
        except Exception as e:
            self._pg.report_error(e)

        return True

    def get_future(self) -> Future[object]:
        fut = self._work.get_future()

        # schedule error handling as a continuation on the Future
        def callback(
            fut: torch.futures.Future[List[torch.Tensor]],
        ) -> object:
            try:
                return fut.value()
            except Exception as e:
                logger.exception(f"got exception in future -- skipping remaining: {e}")
                self._pg.report_error(e)
                return self._default_result

        fut = fut.then(callback)
        return fut


class ErrorSwallowingProcessGroupWrapper(ProcessGroupWrapper):
    """
    This is a wrapper around any ProcessGroup that will swallow errors and
    return dummy results on error.

    This is intended to allow handling errors outside of the training loop to
    avoid having to modify modeling code to support error handling.

    After an error occurs all future operations will be skipped until the
    process group is reconfigured via ``configure``.
    """

    def __init__(self, pg: ProcessGroup) -> None:
        super().__init__(pg=pg)

        self._error: Optional[Exception] = None

    def configure(self, store_addr: str, rank: int, world_size: int) -> None:
        self._error = None

        super().configure(store_addr, rank, world_size)

    def report_error(self, e: Exception) -> None:
        """
        Report an error to this process group. This will cause all future
        operations to be skipped until the process group is reconfigured via
        ``configure``.

        Args:
            e: exception to report
        """
        self._error = e

    def error(self) -> Optional[Exception]:
        """
        Returns the error that was reported to this process group.

        Returns:
            exception that was reported
        """
        return self._error

    def allreduce(self, tensors: List[torch.Tensor], opts: object) -> Work:
        if self._error is not None:
            return _DummyWork(tensors)

        try:
            return _ErrorSwallowingWork(
                self,
                super().allreduce(tensors, opts),
                tensors,
            )
        except Exception as e:
            self.report_error(e)
            return _DummyWork(tensors)


class _ManagedWork(Work):
    def __init__(self, manager: "Manager", work: Work, default_result: object) -> None:
        super().__init__()

        self._manager = manager
        self._work = work
        self._default_result = default_result

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        try:
            if self._work is not None:
                if timeout is not None:
                    self._work.wait(timeout)
                else:
                    self._work.wait()
        except Exception as e:
            self._manager.report_error(e)

        return True

    def get_future(self) -> Future[object]:
        return self._manager.wrap_future(self._work.get_future(), self._default_result)


class ManagedProcessGroup(ProcessGroupWrapper):
    """
    This is a wrapper around any ProcessGroup that is managed by a torchft
    Manager.

    This uses the ProcessGroup that is configured in the Manager. The world size
    is dynamic and will report the number of active particpants in the quorum to
    the model.

    Any errors will be asynchronously reported to the manager and only successes
    will be returned to the caller.
    """

    def __init__(self, manager: "Manager") -> None:
        super().__init__(pg=manager._pg)

        self._manager = manager

    def allreduce(self, tensors: List[torch.Tensor], opts: object) -> Work:
        # Ensure we have a valid quorum and are configured before trying to do
        # any work.
        self._manager.wait_quorum()

        if self._manager.errored() is not None:
            return _DummyWork(tensors)
        try:
            work = super().allreduce(tensors, opts)
        except Exception as e:
            self._manager.report_error(e)
            return _DummyWork(tensors)

        return _ManagedWork(
            self._manager,
            work,
            tensors,
        )

    def size(self) -> int:
        return self._manager.num_participants()

    def getBackendName(self) -> str:
        return self._manager._pg.getBackendName()


class _BabyWork(Work):
    def __init__(
        self,
        pg: "ProcessGroupBaby",
        op_id: int,
        stream: Optional[torch.cuda.Stream],
    ) -> None:
        super().__init__()

        self._pg = pg
        self._op_id = op_id
        self._stream = stream

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        return self._pg._wait(self._op_id, timeout)

    def synchronize(self) -> None:
        # TODO: No one seems to use this and NCCL wait already only waits the
        # stream and is non-blocking on the CPU side so no real need for a
        # separate call.
        raise NotImplementedError("not implemented")

    def get_future(self) -> Future[object]:
        return self._pg._get_future(self._op_id, self._stream)

    def __del__(self) -> None:
        self._pg._del(self._op_id)


def _is_any_cuda(obj: object) -> bool:
    """
    Returns true if any of the tensors in the object are CUDA tensors.

    Supports lists, tuples, dicts, and tensors.
    """
    return tree_any(lambda obj: isinstance(obj, torch.Tensor) and obj.is_cuda, obj)


@dataclass
class _OpMetadata:
    work: Work
    stream: Optional[torch.cuda.Stream]

    @contextmanager
    def set_stream(self) -> Generator[None, None, None]:
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                yield
        else:
            yield


@dataclass
class _FutureMetadata:
    future: Future[object]
    stream: Optional[torch.cuda.Stream]

    @contextmanager
    def set_stream(self) -> Generator[None, None, None]:
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                yield
        else:
            yield


def _maybe_share_tensors(
    tensor: Union[List[List[torch.Tensor]], List[torch.Tensor], torch.Tensor]
) -> None:
    """Move a tensor / list of tensors to shared memory if not already in shared memory."""
    if isinstance(tensor, list):
        for t in tensor:
            _maybe_share_tensors(t)
    elif isinstance(tensor, torch.Tensor):
        if not tensor.is_shared():
            tensor.share_memory_()
    else:
        raise TypeError(f"expected tensor or list but got {type(tensor)}")


def _assert_list(tensors: Union[List[torch.Tensor], List[List[torch.Tensor]]]) -> None:
    """Assert that the input is a list of tensors or a nested list of tensors."""
    if not isinstance(tensors, list):
        raise TypeError(f"expected list but got {type(tensors)}")


class ProcessGroupBaby(ProcessGroup):
    """
    This is a process group that runs the underlying process group in a
    subprocess. Since it's running in a subprocess all tensors need to be in
    shared memory or will be moved to shared memory. CUDA tensors are implicitly
    share able and don't need any changes.
    """

    def __init__(self, timeout: Union[float, timedelta] = 60.0) -> None:
        super().__init__(0, 1)

        self._world_size = -1

        self._p: Optional[mp.Process] = None
        self._pipe: Optional[_MonitoredPipe] = None
        self._future_pipe: Optional[_MonitoredPipe] = None
        self._future_thread: Optional[threading.Thread] = None
        self._futures: Dict[int, _FutureMetadata] = {}
        self._futures_lock = threading.Lock()

        self._next_op_id = 0

        if isinstance(timeout, timedelta):
            timeout = timeout.total_seconds()

        self._timeout: float = timeout

    def shutdown(self) -> None:
        """
        Shutdown the process group. This will kill the underlying process and
        close all queues.

        This is a no-op if the process group is already shutdown.

        ProcessGroup can be reconfigured after shutdown.
        """

        if self._pipe is not None:
            self._pipe.close()

        future_pipe = self._future_pipe
        if future_pipe is not None:
            # wait for the future thread to exit and then close the queue
            future_pipe.close()

            future_thread = self._future_thread
            assert future_thread is not None

            future_thread.join(timeout=10.0)
            if future_thread.is_alive():
                raise RuntimeError("future thread did not exit")

        # Kill after closing queues to avoid log spam.
        if self._p is not None:
            self._p.kill()

    def configure(self, store_addr: str, rank: int, world_size: int) -> None:
        self._world_size = world_size

        self.shutdown()

        ctx = mp.get_context("spawn")
        req_local, req_remote = ctx.Pipe()
        future_local, future_remote = ctx.Pipe()

        self._pipe = req_local = _MonitoredPipe(req_local)
        self._future_pipe = future_local = _MonitoredPipe(future_local)

        curr_device = torch.cuda.current_device() if torch.cuda.is_available() else -1

        self._p = p = ctx.Process(
            target=self._worker,
            args=(
                store_addr,
                rank,
                world_size,
                req_remote,
                future_remote,
                curr_device,
            ),
            daemon=True,
        )
        p.start()

        # futures need thread to fire callbacks
        # this lock needs to be held when manipulating _futures
        self._futures_lock = threading.Lock()
        self._futures = {}
        self._future_thread = threading.Thread(
            target=self._future_handler,
            args=(future_local,),
            daemon=True,
        )
        self._future_thread.start()

        # fetch the status of the PG init
        # if an exception was returned get will throw
        assert req_local.recv(self._timeout) is None

    @classmethod
    def _create_pg(cls, store: Store, rank: int, world_size: int) -> BaseProcessGroup:
        """
        This is a class method to avoid pickling the class.
        """
        raise NotImplementedError("not implemented")

    @classmethod
    def _worker(
        cls,
        store_addr: str,
        rank: int,
        world_size: int,
        req_pipe: "Connection[object, object]",
        future_pipe: "Connection[object, object]",
        curr_device: int,
    ) -> None:
        try:
            if curr_device >= 0 and torch.cuda.is_available():
                torch.cuda.set_device(curr_device)

            store = create_store_client(
                store_addr,
                # default TCPStore timeout is 5 minutes
                timeout=timedelta(minutes=5),
            )

            try:
                pg = cls._create_pg(store, rank, world_size)
            except Exception as e:
                logger.exception(f"got exception in worker: {e}")
                req_pipe.send(e)
                return
            req_pipe.send(None)

            streams: Dict[str, torch.cuda.Stream] = {}
            work: Dict[int, _OpMetadata] = {}

            while True:
                op = cast(list[object], req_pipe.recv())
                cmd = op[0]
                if cmd == "func":
                    op_id: int
                    op_id, func_name, args, kwargs, stream_device, stream_id, event = (
                        cast(
                            Tuple[
                                int,
                                str,
                                list[object],
                                dict[str, object],
                                int,
                                int,
                                Optional[torch.cuda.Event],
                            ],
                            op[1:],
                        )
                    )

                    # To avoid potential deadlocks we need to preserve the
                    # stream/synchronization behavior of the parent process.
                    # We allocate one Stream per stream_id to make sure that we
                    # don't accidentally introduce cross stream synchronization
                    # points.
                    if stream_id is not None:
                        stream_key = f"{stream_device}/{stream_id}"
                        if stream_key not in streams:
                            streams[stream_key] = torch.cuda.Stream(
                                device=stream_device
                            )
                        stream = streams[stream_key]
                    else:
                        stream = None

                    with (
                        torch.cuda.stream(stream)
                        if stream is not None
                        else nullcontext()
                    ):
                        # Make the stream wait on the cuda event to make sure we
                        # don't start the operation until the tensor is ready.
                        if event is not None:
                            event.wait()

                        args = _PickleSafeOptions.unsafe_args(args)
                        fn = getattr(pg, func_name)

                        work[op_id] = _OpMetadata(
                            work=fn(*args, **kwargs),
                            stream=stream,
                        )

                elif cmd == "wait":
                    op_id, timeout = cast(tuple[int, timedelta], op[1:])

                    metadata = work[op_id]

                    with metadata.set_stream():
                        # With WorkNCCL this makes the stream wait not the CPU when
                        # no timeout is passed.
                        if timeout is not None:
                            metadata.work.wait(timeout)
                        else:
                            metadata.work.wait()

                        # Register event on the stream that we can pass to the main
                        # process.
                        event = (
                            torch.cuda.current_stream().record_event(
                                torch.cuda.Event(interprocess=True)
                            )
                            if metadata.stream is not None
                            else None
                        )

                    req_pipe.send((op_id, event))
                elif cmd == "del":
                    op_id: int = cast(int, op[1])
                    del work[op_id]
                elif cmd == "future":
                    op_id: int = cast(int, op[1])
                    metadata: _OpMetadata = work[op_id]

                    def callback(fut: Future[object], metadata: _OpMetadata) -> None:
                        try:
                            # create an event after the collective has been issued
                            # to wait on this before we call "future"
                            with metadata.set_stream():
                                fut.wait()
                                event = (
                                    torch.cuda.current_stream().record_event(
                                        torch.cuda.Event(interprocess=True)
                                    )
                                    if metadata.stream is not None
                                    else None
                                )

                            future_pipe.send((op_id, _FUTURE_RESULT, None, event))
                        except Exception as e:
                            future_pipe.send((op_id, _FUTURE_EXCEPTION, e, None))

                    metadata.work.get_future().add_done_callback(
                        lambda fut: callback(fut, metadata)
                    )
                elif cmd == "num_active_work":
                    req_pipe.send(len(work))
                else:
                    raise ValueError(f"unknown cmd: {cmd}")

        except Exception as e:
            logger.exception(f"worker errored: {e}")
            req_pipe.send(e)
            raise

    def _future_handler(self, future_pipe: _MonitoredPipe) -> None:
        try:
            while True:
                try:
                    cmd = future_pipe.recv(timedelta(seconds=10))
                except TimeoutError:
                    continue
                except OSError:
                    # subprocess exited
                    break

                op_id, mode, data, event = cast(
                    Tuple[int, str, object, Optional[torch.cuda.Event]], cmd
                )
                with self._futures_lock:
                    meta = self._futures[op_id]
                    del self._futures[op_id]
                with meta.set_stream():
                    if mode == _FUTURE_RESULT:
                        if event is not None:
                            event.wait()
                        meta.future.set_result(data)
                    elif mode == _FUTURE_EXCEPTION:
                        meta.future.set_exception(data)
                    else:
                        raise ValueError(f"unknown mode {mode}")
        except Exception as e:
            logger.exception(f"got unexpected error in future handler: {e}")

    def _get_future(
        self, op_id: int, stream: Optional[torch.cuda.Stream]
    ) -> Future[object]:
        with self._futures_lock:
            fut = Future()  # pyre-fixme[29]: is not a function
            self._futures[op_id] = _FutureMetadata(future=fut, stream=stream)
            assert self._pipe is not None
            self._pipe.send(("future", op_id))

        # TODO: return correct tensor instead of None
        return fut

    def _wait(self, op_id: int, timeout: Optional[timedelta] = None) -> bool:
        assert self._pipe is not None
        self._pipe.send(("wait", op_id, timeout))

        assert self._pipe is not None
        op_id, event = cast(
            Tuple[int, Optional[torch.cuda.Event]],
            self._pipe.recv(timeout or self._timeout),
        )
        assert op_id == op_id
        if event is not None:
            event.wait()

        return True

    def _del(self, op_id: int) -> None:
        assert self._pipe is not None
        try:
            self._pipe.send(("del", op_id))
        except OSError:
            # if pipe is closed we can safely do nothing
            pass

    def _run_func(self, func: str, *args: object, **kwargs: object) -> Work:
        pipe = self._pipe
        assert pipe is not None

        is_cuda = _is_any_cuda(args)

        stream_device = torch.cuda.current_stream().device if is_cuda else None
        stream_id = torch.cuda.current_stream().stream_id if is_cuda else None
        event = (
            torch.cuda.current_stream().record_event(
                torch.cuda.Event(interprocess=True)
            )
            if is_cuda
            else None
        )

        op_id = self._next_op_id
        self._next_op_id += 1

        pipe.send(
            (
                "func",
                op_id,
                func,
                _PickleSafeOptions.safe_args(args),
                kwargs,
                stream_device,
                stream_id,
                event,
            ),
        )

        return _BabyWork(
            pg=self,
            op_id=op_id,
            stream=torch.cuda.current_stream() if is_cuda else None,
        )

    def allgather(
        self,
        output_tensors: List[List[torch.Tensor]],
        input_tensor: List[torch.Tensor],
        opts: AllgatherOptions,
    ) -> Work:
        _assert_list(output_tensors)
        _assert_list(input_tensor)
        _maybe_share_tensors(output_tensors)
        _maybe_share_tensors(input_tensor)
        return self._run_func("allgather", output_tensors, input_tensor, opts)

    def allgather_into_tensor_coalesced(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[torch.Tensor],
        opts: AllgatherOptions,
    ) -> Work:
        _assert_list(output_tensors)
        _assert_list(input_tensors)
        _maybe_share_tensors(output_tensors)
        _maybe_share_tensors(input_tensors)
        return self._run_func(
            "allgather_into_tensor_coalesced", output_tensors, input_tensors, opts
        )

    def allreduce(
        self,
        tensors: List[torch.Tensor],
        opts: Union[dist.AllreduceOptions, dist.ReduceOp],
    ) -> Work:
        _assert_list(tensors)
        _maybe_share_tensors(tensors)
        return self._run_func("allreduce", tensors, opts)

    def allreduce_coalesced(
        self,
        tensors: List[torch.Tensor],
        opts: Union[dist.AllreduceCoalescedOptions, dist.ReduceOp],
    ) -> Work:
        _assert_list(tensors)
        _maybe_share_tensors(tensors)
        return self._run_func("allreduce_coalesced", tensors, opts)

    def alltoall_base(
        self,
        output_buffer: torch.Tensor,
        input_buffer: torch.Tensor,
        output_split_sizes: List[int],
        input_split_sizes: List[int],
        opts: AllToAllOptions,
    ) -> Work:
        _maybe_share_tensors(output_buffer)
        _maybe_share_tensors(input_buffer)
        return self._run_func(
            "alltoall_base",
            output_buffer,
            input_buffer,
            output_split_sizes,
            input_split_sizes,
            opts,
        )

    def barrier(self, opts: BarrierOptions) -> Work:
        return self._run_func("barrier", opts)

    def broadcast(
        self,
        tensor_list: List[torch.Tensor],
        opts: BroadcastOptions,
    ) -> Work:
        _assert_list(tensor_list)
        _maybe_share_tensors(tensor_list)
        return self._run_func("broadcast", tensor_list, opts)

    def recv(self, tensors: List[torch.Tensor], src_rank: int, tag: int) -> Work:
        _assert_list(tensors)
        _maybe_share_tensors(tensors)
        return self._run_func("recv", tensors, src_rank, tag)

    def reduce_scatter(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[List[torch.Tensor]],
        opts: ReduceScatterOptions,
    ) -> Work:
        _assert_list(output_tensors)
        _assert_list(input_tensors)
        _maybe_share_tensors(output_tensors)
        _maybe_share_tensors(input_tensors)
        return self._run_func("reduce_scatter", output_tensors, input_tensors, opts)

    def reduce_scatter_tensor_coalesced(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[torch.Tensor],
        opts: ReduceScatterOptions,
    ) -> Work:
        _assert_list(output_tensors)
        _assert_list(input_tensors)
        _maybe_share_tensors(output_tensors)
        _maybe_share_tensors(input_tensors)
        return self._run_func(
            "reduce_scatter_tensor_coalesced", output_tensors, input_tensors, opts
        )

    def send(self, tensors: List[torch.Tensor], dst_rank: int, tag: int) -> Work:
        _assert_list(tensors)
        _maybe_share_tensors(tensors)
        return self._run_func("send", tensors, dst_rank, tag)

    def size(self) -> int:
        return self._world_size

    def num_active_work(self) -> int:
        assert self._pipe is not None
        self._pipe.send(("num_active_work",))

        assert self._pipe is not None
        return cast(int, self._pipe.recv(self._timeout))

    def set_timeout(self, timeout: timedelta) -> None:
        self._timeout = timeout.total_seconds()


@dataclass
class _PickleSafeOptions:
    func: Callable[[], object]
    fields: Dict[str, object]

    @classmethod
    def safe_args(cls, args: T) -> T:
        if isinstance(args, tuple):
            return tuple(cls.safe_args(arg) for arg in args)
        elif isinstance(args, list):
            return [cls.safe_args(arg) for arg in args]
        elif isinstance(
            args,
            (
                AllgatherOptions,
                AllreduceOptions,
                AllreduceCoalescedOptions,
                AllToAllOptions,
                BarrierOptions,
                BroadcastOptions,
                ReduceScatterOptions,
            ),
        ):
            return cls.from_torch(args)
        else:
            return args

    @classmethod
    def unsafe_args(cls, args: T) -> T:
        if isinstance(args, tuple):
            return tuple(cls.unsafe_args(arg) for arg in args)
        elif isinstance(args, list):
            return [cls.unsafe_args(arg) for arg in args]
        elif isinstance(args, cls):
            return args.to_torch()
        else:
            return args

    @classmethod
    def from_torch(cls, opts: object) -> "_PickleSafeOptions":
        return cls(
            func=opts.__class__,
            fields={k: getattr(opts, k) for k in dir(opts) if not k.startswith("_")},
        )

    def to_torch(self) -> object:
        opts = self.func()
        for k, v in self.fields.items():
            setattr(opts, k, v)
        return opts


class ProcessGroupBabyGloo(ProcessGroupBaby):
    """
    This is a ProcessGroup that runs Gloo in a subprocess.

    For most use cases you should prefer ProcessGroupGloo or
    ProcessGroupBabyNCCL.
    """

    @classmethod
    def _create_pg(cls, store: Store, rank: int, world_size: int) -> BaseProcessGroup:
        pg = BaseProcessGroup(store, rank, world_size)
        pg._set_default_backend(ProcessGroup.BackendType.GLOO)
        # pyre-fixme[16]: no attribute ProcessGroupGloo
        backend_class = BaseProcessGroupGloo(store, rank, world_size)
        pg._register_backend(
            torch.device("cpu"), ProcessGroup.BackendType.GLOO, backend_class
        )
        return pg

    def getBackendName(self) -> str:
        return "torchft-baby-gloo"

    # pyre-fixme[15]: inconsistent override
    def reduce_scatter(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[List[torch.Tensor]],
        opts: ReduceScatterOptions,
    ) -> None:
        """
        This function is a placeholder for the reduce_scatter operation in the
        ProcessGroupGloo class. However, this operation is not supported by the
        Gloo backend, and thus, calling this function will raise a
        RuntimeError.

        Raises:
            RuntimeError: Always raised since reduce_scatter is not
            supported by ProcessGroupGloo.
        """
        raise RuntimeError("ProcessGroupBabyGloo does not support reduce_scatter.")

    # pyre-fixme[15]: inconsistent override
    def reduce_scatter_tensor_coalesced(
        self,
        output_tensors: List[torch.Tensor],
        input_tensors: List[torch.Tensor],
        opts: ReduceScatterOptions,
    ) -> None:
        """
        This function is a placeholder for the reduce_scatter_tensor_coalesced
        operation in the ProcessGroupBabyGloo class.
        However, this operation is not supported by the
        Gloo backend, and thus, calling this function will raise a
        RuntimeError.

        Raises:
            RuntimeError: Always raised since reduce_scatter is not
            supported by ProcessGroupBabyGloo.
        """
        raise RuntimeError(
            "ProcessGroupBabyGloo does not support reduce_scatter_tensor_coalesced."
        )


class ProcessGroupBabyNCCL(ProcessGroupBaby):
    """
    This is a ProcessGroup that runs NCCL in a subprocess.

    For the NCCL backend, extra memory will be used by the subprocesses CUDA
    context compared to running NCCL in the main process. This is typically
    around ~1GB.

    The returned Work objects only synchronize on the cuda stream and not on the
    CPU side. This works by passing CUDA Events between the processes. To do a
    CPU synchronize, call torch.cuda.synchronize() after wait().

    WARNING: If the child process is killed while an operation is running, CUDA
    tensors may leak in the current PyTorch implementation. TODO fix

    WARNING: As this uses a separate CUDA context for the subprocess, performance
    may be slower than using NCCL directly. Separate CUDA contexts can not run
    at the same time so network and compute kernels will not overlap execution
    and instead do time sharing which may reduce GPU utilization.
    """

    @classmethod
    def _create_pg(cls, store: Store, rank: int, world_size: int) -> BaseProcessGroup:
        from torch.distributed import ProcessGroupNCCL as BaseProcessGroupNCCL

        pg = BaseProcessGroup(store, rank, world_size)
        pg._set_default_backend(ProcessGroup.BackendType.NCCL)
        # pyre-fixme[16]: no attribute ProcessGroupNCCL
        backend_class = BaseProcessGroupNCCL(store, rank, world_size)
        backend_class._set_sequence_number_for_group()
        pg._register_backend(
            torch.device("cuda"), ProcessGroup.BackendType.NCCL, backend_class
        )
        return pg

    def getBackendName(self) -> str:
        return "torchft-baby-nccl"
