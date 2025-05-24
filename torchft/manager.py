# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Manager
=========

This module implements the Manager that manages the full fault tolerant training
loop.

The Manager is responsible for managing the
full training loop, communicating with the Lighthouse server to figure out
quorum, reconfiguring the ProcessGroups and restoring checkpoint state when
recovering.

This uses wrapper classes to wrap the standard PyTorch Optimizer and Module
classes to provide fault tolerance. These wrappers indented to add fault
tolerance with minimal changes to the users modeling code and training loop.

This is designed to work with the standard PyTorch DistributedDataParallel module
and Hybrid FSDP.

"""
import concurrent.futures
import logging
import multiprocessing
import os
import socket
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from enum import Enum
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, TypeVar, cast

import torch
from torch.distributed import ReduceOp, TCPStore

from torchft._torchft import LighthouseClient, ManagerClient, ManagerServer
from torchft.checkpointing import CheckpointTransport, HTTPTransport
from torchft.futures import future_timeout
from torchft.multiprocessing import _MonitoredPipe

if TYPE_CHECKING:
    from torchft.process_group import ProcessGroup

MANAGER_ADDR_KEY: str = "manager_addr"
MANAGER_PORT_ENV: str = "TORCHFT_MANAGER_PORT"
REPLICA_ID_KEY: str = "replica_id"

T = TypeVar("T")


class WorldSizeMode(Enum):
    """
    This controls the numerics for the job when doing allreduces across replicas
    when the world size is larger than ``min_replica_size``. The world size will
    never be smaller than ``min_replica_size``.

    DYNAMIC:
        The world size will dynamical increase to use all available
        replicas and normalize the gradient by the world size.
    FIXED_WITH_SPARES:
        The number of active replicas is ``min_replica_size`` and any spares
        will contribute zero gradients.
    """

    DYNAMIC = 0
    FIXED_WITH_SPARES = 1


class ExceptionWithTraceback(Exception):
    def __init__(self, e: Exception) -> None:
        self.original_exception = e
        self.stack_trace: str = traceback.format_exc()
        super().__init__(f"{e}\n{self.stack_trace}")


class Manager:
    """
    Manager manages the full fault tolerant training loop.

    This requires the that the TCPStore specified by the store_addr and
    store_port or MASTER_ADDR and MASTER_PORT environment variables to be
    started prior to creating this manager. If using a modern version of
    torchelastic this will already be the case. Otherwise, it should be started
    via torch.distributed.init_process_group prior to creating this manager.

    NOTE: when saving periodic checkpoints you must save and restore the
    Manager's state_dict as well to avoid synchronization issues.
    """

    def __init__(
        self,
        pg: "ProcessGroup",
        load_state_dict: Optional[Callable[[T], None]],
        state_dict: Optional[Callable[[], T]],
        min_replica_size: int,
        use_async_quorum: bool = True,
        timeout: timedelta = timedelta(seconds=60),
        quorum_timeout: timedelta = timedelta(seconds=60),
        connect_timeout: timedelta = timedelta(seconds=60),
        proactive_recovery_subscribe_timeout: timedelta = timedelta(milliseconds=100),
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        world_size_mode: WorldSizeMode = WorldSizeMode.DYNAMIC,
        store_addr: Optional[str] = None,
        store_port: Optional[int] = None,
        lighthouse_addr: Optional[str] = None,
        replica_id: Optional[str] = None,
        port: Optional[int] = None,
        hostname: str = socket.gethostname(),
        heartbeat_interval: timedelta = timedelta(milliseconds=100),
        checkpoint_transport: Optional[CheckpointTransport[Dict[str, T]]] = None,
        init_sync: bool = True,
        max_retries: Optional[int] = None,
        proactive_recovery: bool = False,
    ) -> None:
        """
        Args:
            load_state_dict: function to load the state dict when recovering
            state_dict: function to save the state dict with recovering
            min_replica_size: minimum number of replicas on each step
            port: if rank==0, the port to run the manager server on.
                Port assignment priority:
                1. this argument
                2. TORCHFT_MANAGER_PORT env var
                3. arbitrary port assigned via 0
            use_async_quorum: whether to run the quorum asynchronously during the forward pass
            timeout: the default timeout for all operations
                Included:
                    * collectives such as allreduce
                    * should_commit rpc
                    * checkpoint_address rpc
                    * checkpoint HTTP operations
                    * wrap_future
            quorum_timeout: the default timeout to wait for the quorum to complete.
                This generally should be longer than the training step time /
                the interval between quorum checks to avoid any split brain
                issues.

                For LocalSGD/DiLoCo this may need to be set to ~1h or longer
                depending on how frequently the syncs occur.
            connect_timeout: the timeout used for establishing rpc connections
                to ManagerServer and Lighthouse
            rank: the replica group local rank, referred to as group_rank in manager.py for clarity
            world_size: the replica group local world size, referred to as group_world_size in manager.py for clarity
            store_addr: TCPStore address for this replica group
            store_port: TCPStore port for this replica group
            lighthouse_addr: if rank==0, the address of the lighthouse server
            replica_id: if rank==0, the replica_id for this group
            hostname: if rank==0, the hostname to advertise to the lighthouse server
            checkpoint_transport: the checkpoint transport to use for
                transfering checkpoints to recovering replicas, defaults to HTTPTransport
            init_sync: whether to synchronize the model weights on step 0. If
                all of the model weights are initialized identically via
                ``torch.set_seed`` you should set this to False.
            max_retries: the maximum number of consecutive should_commit failures to allow
                before raising an exception. If None, will retry indefinitely.
        """
        self._load_state_dict = load_state_dict
        self._user_state_dict = state_dict
        self._pending_state_dict: Optional[Dict[str, object]] = None
        self._use_async_quorum = use_async_quorum
        self._timeout = timeout
        self._quorum_timeout = quorum_timeout
        self._connect_timeout = connect_timeout
        self._proactive_recovery_subscribe_timeout = (
            proactive_recovery_subscribe_timeout
        )
        self._replica_world_size_mode = world_size_mode
        self._init_sync = init_sync
        self._max_retries = max_retries
        self._commit_failures = 0

        store_addr = store_addr or os.environ["MASTER_ADDR"]
        store_port = store_port or int(os.environ["MASTER_PORT"])
        self._group_rank: int = rank if rank is not None else int(os.environ["RANK"])
        group_rank = self._group_rank
        group_world_size = world_size or int(os.environ["WORLD_SIZE"])
        self._min_replica_size = min_replica_size

        if checkpoint_transport is None:
            checkpoint_transport = HTTPTransport[Dict[str, T]](
                timeout=timeout,
                num_chunks=0,
            )

        self._checkpoint_transport: CheckpointTransport[Dict[str, T]] = (
            checkpoint_transport
        )
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="")
        self._quorum_future: Optional[concurrent.futures.Future] = None

        self._store = TCPStore(
            host_name=store_addr,
            port=store_port,
            is_master=False,
            wait_for_workers=False,
        )
        self._pg = pg
        self._manager: Optional[ManagerServer] = None

        self._recovery_stream: Optional["torch.cuda.Stream"] = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )

        lighthouse_addr: Optional[str] = lighthouse_addr
        if os.environ.get("TORCHFT_LIGHTHOUSE") is not None:
            lighthouse_addr = (
                lighthouse_addr or os.environ["TORCHFT_LIGHTHOUSE"]
            )  # Else error in tests, since TORCHFT_LIGHTHOUSE may not be set

        self._proactive_recovery = proactive_recovery or int(
            os.environ.get("TORCHFT_PROACTIVE_RECOVERY", 0)
        )

        if lighthouse_addr is not None and self._proactive_recovery:
            ctx = multiprocessing.get_context("spawn")
            error_local, error_remote = ctx.Pipe()
            self._error_pipe = _MonitoredPipe(error_local)
            self._error_remote = _MonitoredPipe(error_remote)
            self._failure_listener_stop_event = ctx.Event()

            self._failure_listener_process = ctx.Process(
                target=_failure_listener_process_main,
                args=(
                    lighthouse_addr,
                    self._connect_timeout,
                    self._failure_listener_stop_event,
                    error_remote,
                    self._proactive_recovery_subscribe_timeout,
                ),
                daemon=True,
            )
            self._failure_listener_process.start()
        else:
            self._failure_listener_process = None
            self._error_pipe = None
            self._failure_listener_stop_event = None

        # Initialize and start the error processing thread if the listener process is active
        self._error_processor_thread: Optional[threading.Thread] = None
        self._error_processor_stop_event: Optional[threading.Event] = None
        if self._failure_listener_process is not None:
            self._error_processor_stop_event = threading.Event()
            self._error_processor_thread = threading.Thread(
                target=self._error_processor_loop,
                name="TorchFTErrorProcessor",
                daemon=True,
            )
            self._error_processor_thread.start()

        if self._group_rank == 0:
            if port is None:
                port = int(os.environ.get(MANAGER_PORT_ENV, 0))

            bind = f"[::]:{port}"

            # We need a unique identifier in the case that a worker restarts quickly and
            # replaces the previous worker with the same ID.
            new_uuid = str(uuid.uuid4())
            if replica_id is None or replica_id == "":
                replica_id = new_uuid
            else:
                replica_id = f"{replica_id}:{new_uuid}"

            self._manager = ManagerServer(
                replica_id=replica_id,
                lighthouse_addr=lighthouse_addr,
                hostname=hostname,
                bind=bind,
                store_addr=f"{store_addr}:{store_port}",
                world_size=group_world_size,
                heartbeat_interval=heartbeat_interval,
                connect_timeout=connect_timeout,
            )
            self._store.set(MANAGER_ADDR_KEY, self._manager.address())
            self._store.set(REPLICA_ID_KEY, replica_id)

        addr = self._store.get(MANAGER_ADDR_KEY).decode("utf-8")
        self._client = ManagerClient(addr, connect_timeout=connect_timeout)
        replica_id = self._store.get(REPLICA_ID_KEY).decode("utf-8")
        self._logger = _ManagerLogger(
            manager=self, replica_id=replica_id or "", group_rank=group_rank
        )

        self._step = 0
        self._quorum_id = -1
        self._errored: Optional[ExceptionWithTraceback] = None
        self._healing = False
        self._pending_work: List[torch.futures.Future[object]] = []
        self._batches_committed = 0

        # first step is 1
        self._participating_replica_rank: Optional[int] = None
        self._participating_replica_world_size: int = 0

    def set_state_dict_fns(
        self, load_state_dict: Callable[[T], None], state_dict: Callable[[], T]
    ) -> None:
        self._load_state_dict = load_state_dict
        self._user_state_dict = state_dict

    def _error_handler(self, err):
        self._logger.info(f"Received error: {err}")
        self.report_error(err)
        self._pg.abort()

    def _error_processor_loop(self) -> None:
        """Continuously checks the error pipe from the listener process and reports errors."""
        assert (
            self._error_pipe is not None
        ), "Error pipe must be initialized for error processor loop."
        assert (
            self._error_processor_stop_event is not None
        ), "Stop event must be initialized for error processor loop."

        try:
            while not self._error_processor_stop_event.is_set():
                try:
                    item = self._error_pipe.recv(0.1)
                except TimeoutError:
                    continue
                except OSError:
                    break
                except Exception as e:
                    self._error_handler(e)
        finally:
            pass

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the manager and checkpoint server.
        """
        if self._manager is not None:
            self._manager.shutdown()

        # Stop the error processor thread first
        if (
            self._error_processor_thread is not None
            and self._error_processor_stop_event is not None
        ):
            self._logger.info("Setting error processor thread stop event")
            self._error_processor_stop_event.set()
            if wait:
                self._logger.info("Waiting for error processor thread to complete")
                try:
                    self._error_processor_thread.join(timeout=5)  # Short timeout
                    if self._error_processor_thread.is_alive():
                        self._logger.warn(
                            "Error processor thread did not terminate in time."
                        )
                    else:
                        self._logger.info("Error processor thread shutdown completed.")
                except Exception as e:
                    self._logger.warn(f"Error waiting for error processor thread: {e}")

        # Stop the failure listener process if it exists
        if (
            hasattr(self, "_failure_listener_process")
            and self._failure_listener_process is not None
        ):
            self._logger.info("Setting failure listener stop event for process")
            if (
                hasattr(self, "_failure_listener_stop_event")
                and self._failure_listener_stop_event is not None
            ):
                self._failure_listener_stop_event.set()

            if wait:
                self._logger.info("Waiting for failure listener process to complete")
                try:
                    self._failure_listener_process.join(timeout=10)  # Process join
                    if self._failure_listener_process.is_alive():
                        self._logger.warn(
                            "Failure listener process did not terminate, attempting to terminate."
                        )
                        self._failure_listener_process.terminate()  # Force terminate if join times out
                        self._failure_listener_process.join(
                            timeout=1
                        )  # Wait for terminate
                    else:
                        self._logger.info("Failure listener process shutdown completed")
                except Exception as e:
                    self._logger.warn(
                        f"Error waiting for/terminating failure listener process: {e}"
                    )

            # Clean up pipe
            if hasattr(self, "_error_pipe") and self._error_pipe is not None:
                self._error_pipe.close()

        self._checkpoint_transport.shutdown(wait=wait)
        self._executor.shutdown(wait=wait)

    def allreduce(self, tensor: torch.Tensor) -> torch.futures.Future[torch.Tensor]:
        """
        Fault tolerant allreduce the tensor and return a Future that will be completed when
        the tensor is ready.

        This will automatically scale the tensor by 1 / world_size.

        If an error occurs during the allreduce:

        * The Future will be completed with no error and instead tracked asynchronously.
        * After the first error, all subsequent calls will be noops and immediately return.
        * The tensor must be zeroed before being used as it may be corrupted.

        Args:
            tensor: the tensor to allreduce
        Returns:
            a Future that will be completed with the allreduced tensor
        """
        if self.errored():
            fut = torch.futures.Future()  # pyre-fixme[29]: not a function
            fut.set_result(tensor)
            return fut

        self.wait_quorum()

        if not self.is_participating():
            tensor.zero_()

        # TODO: increase timeout when waiting when healing
        try:
            # Run the allreduce async and save the work object so we can wait on
            # it later.
            work = self._pg.allreduce([tensor], ReduceOp.SUM)
            fut = work.get_future()

            # schedule grad normalization as a continuation
            # on the Future
            def callback(
                fut: torch.futures.Future[List[torch.Tensor]],
            ) -> torch.Tensor:
                nonlocal tensor

                # check for exceptions
                fut.value()

                tensor /= self.num_participants()

                return tensor

            fut = fut.then(callback)
            fut = self.wrap_future(fut, tensor)
            return fut

        except Exception as e:
            self._logger.exception(
                f"got exception in all reduce -- skipping remaining: {e}"
            )
            self.report_error(e)

            fut = torch.futures.Future()  # pyre-fixme[29]: not a function
            fut.set_result(tensor)
            return fut

    def report_error(self, e: Exception) -> None:
        """
        Report an error to the manager.

        This will cause the manager to skip the current step and will be
        reconfigured on the next step.

        This should be called when an error occurs that leads to a corrupted
        gradient that needs to be discarded.
        """
        self._errored = ExceptionWithTraceback(e)

    def errored(self) -> Optional[ExceptionWithTraceback]:
        """
        Get whether an error has occurred.

        Returns:
            The error or None if no error has occurred.
        """
        return self._errored

    def wrap_future(
        self,
        fut: torch.futures.Future[T],
        default: T,
        timeout: Optional[timedelta] = None,
    ) -> torch.futures.Future[T]:
        """
        Wrap a Future and swallow any errors that occur and report them to the manager.

        If an error occurs, the Future will be completed with the default value.

        Args:
            fut: the Future to wrap
            default: the default value to complete the Future with if an error occurs
            timeout: the timeout for the Future, if None, the manager's timeout will be used
        """

        # add a timeout to the future
        fut = future_timeout(fut, timeout or self._timeout)

        # schedule error handling as a continuation on the Future
        def callback(
            fut: torch.futures.Future[T],
        ) -> T:
            nonlocal default

            try:
                return fut.value()
            except Exception as e:
                self._logger.exception(
                    f"got exception in future -- skipping remaining: {e}"
                )
                self.report_error(e)
                return default

        fut = fut.then(callback)
        self._pending_work.append(cast(torch.futures.Future[object], fut))
        return fut

    def start_quorum(
        self,
        allow_heal: bool = True,
        shrink_only: bool = False,
        timeout: Optional[timedelta] = None,
    ) -> None:
        """
        .. note::
            We recommend using the :py:class:`torchft.optim.OptimizerWrapper` instead of calling this directly.

        Computes a new quorum (potentially asynchronously) and readies the
        manager for a new step.

        It's best practice to call this before the forwards pass of each step for
        performance as computing quorum may take some time.

        Args:
            allow_heal: (experimental) whether to allow healing at the beginning of the step
                If allow_heal is set, the manager will attempt to heal either
                synchronously before returning or asynchronously prior to any network
                calls. All replicas must pass the same value to allow_heal.
            timeout: the timeout for quorum to be ready, if None, the manager's timeout will be used
                recovery operations will use the manager timeout
        """

        # wait for previous quorum to complete
        if self._quorum_future is not None:
            self._quorum_future.result()

        self._errored = None
        self._healing = False

        # TODO: we should really be wrapping this whole section in a try-except
        # block to allow gracefully recovering from issues in PG setup and quorum.

        self._quorum_future = self._executor.submit(
            self._async_quorum,
            allow_heal=allow_heal,
            shrink_only=shrink_only,
            quorum_timeout=timeout or self._quorum_timeout,
            curr_device=(
                torch.cuda.current_device() if torch.cuda.is_available() else -1
            ),
        )
        if not self._use_async_quorum:
            self.wait_quorum()

            if self._healing:
                # eagerly apply pending state_dict so we can run the forwards pass
                self._apply_pending_state_dict()

                # we are forcing healing at the beginning so we're in a good state
                # and don't need to zero_grad
                self._healing = False

    def wait_quorum(self) -> None:
        """
        Wait for the quorum to complete.

        ProcessGroup will be in a healthy state after this returns.
        """
        assert (
            self._quorum_future is not None
        ), "must call start_quorum before wait_quorum"
        self._quorum_future.result()

    @torch.profiler.record_function("torchft::manager::_async_quorum")
    def _async_quorum(
        self,
        allow_heal: bool,
        shrink_only: bool,
        quorum_timeout: timedelta,
        curr_device: int,
    ) -> None:
        torch.multiprocessing._set_thread_name("torchft_quorum")

        if curr_device >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(curr_device)

        quorum = None
        with torch.profiler.record_function("torchft::manager::_client::_quorum"):
            quorum = self._client._quorum(
                group_rank=self._group_rank,
                step=self._step,
                checkpoint_metadata=self._checkpoint_transport.metadata(),
                shrink_only=shrink_only,
                timeout=quorum_timeout,
                init_sync=self._init_sync,
                commit_failures=self._commit_failures,
            )

        quorum_id = quorum.quorum_id
        replica_rank = quorum.replica_rank
        replica_world_size = quorum.replica_world_size
        recover_src_manager_address = quorum.recover_src_manager_address
        store_address = quorum.store_address
        max_step = quorum.max_step
        max_replica_rank = quorum.max_replica_rank
        max_replica_world_size = quorum.max_world_size
        heal = quorum.heal

        # When using async quorum we need to take the recovered workers.
        # When not using async quorum we need to take the max world size as all
        # workers will be healthy.
        self._participating_replica_rank, self._participating_replica_world_size = (
            (max_replica_rank, max_replica_world_size)
            if self._use_async_quorum or not allow_heal
            else (replica_rank, replica_world_size)
        )

        # For fixed with spares we need to ensure that we don't have more
        # participating replicas than the min replica size.
        if self._replica_world_size_mode == WorldSizeMode.FIXED_WITH_SPARES:
            self._participating_replica_world_size = min(
                self._participating_replica_world_size, self._min_replica_size
            )
            if (
                self._participating_replica_rank is not None
                and self._participating_replica_rank >= self._min_replica_size
            ):
                self._participating_replica_rank = None

        if quorum_id != self._quorum_id:
            store_prefixed_addr = (
                f"{store_address}/torchft/{quorum_id}/{self._group_rank}"
            )

            self._logger.info(f"reconfiguring for {quorum_id=} {store_prefixed_addr=}")
            # We use the replica rank and world as we want all replicas in the PG.
            try:
                with torch.profiler.record_function("torchft::manager::_pg.configure"):
                    self._pg.configure(
                        store_prefixed_addr, replica_rank, replica_world_size
                    )
                self._quorum_id = quorum_id
            except Exception as e:
                self._logger.exception(f"got exception in pg configure: {e}")
                self.report_error(e)
                return

        if allow_heal:
            # run recovery on the recovery stream if available
            recovery_stream = self._recovery_stream
            with (
                torch.cuda.stream(recovery_stream)
                if recovery_stream is not None
                else nullcontext()
            ):
                try:
                    if quorum.recover_dst_replica_ranks:
                        self._logger.info(
                            f"peers need recovery from us {quorum.recover_dst_replica_ranks}"
                        )
                        with torch.profiler.record_function(
                            "torchft::manager::_checkpoint_transport::send_checkpoint"
                        ):
                            self._checkpoint_transport.send_checkpoint(
                                dst_ranks=quorum.recover_dst_replica_ranks,
                                step=max_step,
                                state_dict=self._manager_state_dict(),
                                timeout=self._timeout,
                            )

                    # See manager.rs for healing conditions
                    if heal:
                        self._healing = True
                        self._logger.info(
                            f"healing required, fetching checkpoint metadata from {recover_src_manager_address=} {max_step=}"
                        )
                        primary_client = ManagerClient(
                            recover_src_manager_address,
                            connect_timeout=self._connect_timeout,
                        )
                        checkpoint_metadata = primary_client._checkpoint_metadata(
                            self._group_rank, timeout=self._timeout
                        )
                        recover_src_replica_rank = quorum.recover_src_replica_rank
                        assert (
                            recover_src_replica_rank is not None
                        ), "must have a recover rank when healing"

                        self._logger.info(
                            f"fetching checkpoint from {recover_src_replica_rank=} with {checkpoint_metadata=}"
                        )

                        # we apply the user state dict only when safe from the main thread
                        # save it for now
                        with torch.profiler.record_function(
                            "torchft::manager::_checkpoint_transport::recv_checkpoint"
                        ):
                            self._pending_state_dict = self._checkpoint_transport.recv_checkpoint(
                                src_rank=recover_src_replica_rank,
                                metadata=checkpoint_metadata,  # Depending on group rank
                                step=max_step,
                                timeout=self._timeout,
                            )

                        # pyre-fixme[6]: got object
                        self.load_state_dict(self._pending_state_dict["torchft"])

                        # This isn't strictly needed as loading the state_dict above should
                        # restore the correct step but it makes writing tests simpler.
                        self._step = max_step
                except Exception as e:
                    self._logger.exception(f"got exception in recovery: {e}")
                    self.report_error(e)
                    return

    def _apply_pending_state_dict(self) -> None:
        assert self._healing, "must be in healing state"

        # synchronize on future
        assert self._quorum_future is not None, "must call step before should_commit"
        self._quorum_future.result()

        pending_state_dict = self._pending_state_dict

        if pending_state_dict is None:
            assert self.errored(), "checkpoint was not staged and no error occured"
        else:
            self._logger.info("applying pending state dict")

            assert (
                self._load_state_dict is not None
            ), "user load_state_dict is not initialized."
            self._load_state_dict(pending_state_dict["user"])
            self._pending_state_dict = None
            self._logger.info("Loaded state dict.")

    @torch.profiler.record_function("torchft::manager::should_commit")
    def should_commit(self, timeout: Optional[timedelta] = None) -> bool:
        """
        .. note::
            We recommend using the :py:class:`torchft.optim.OptimizerWrapper` instead of calling this directly.

        Must be called after the backwards pass completes but before stepping the optimizer.

        The optimizer must only be stepped if this returns True.

        This must be called on all workers within a replica group. This uses a
        collective to ensure all workers within a replica return the same value.
        If an error occurs on any worker, all workers will return False.
        Different replica groups may return different values.

        This should only be called once per step.

        If max_retries is set and should_commit fails that many times consecutively,
        this method will raise a RuntimeError to prevent indefinite failure loops.

        Returns:
            True if the optimizer should be stepped, False otherwise
        Raises:
            RuntimeError: if should_commit fails max_retries times in a row and max_retries is set
        """
        for work in self._pending_work:
            # check at the beginning of since .wait() may trigger errors
            if self._errored is not None:
                break

            # We swallow the error at in a future then callback so this will
            # never return an error.
            work.wait()

        # make sure recovery is complete before committing
        if self._recovery_stream is not None:
            self._recovery_stream.synchronize()

        self._pending_work = []

        if err := self._pg.errored():
            self.report_error(err)

        # apply state_dict if healing
        if self._healing:
            self._apply_pending_state_dict()

        enough_replicas = self.num_participants() >= self._min_replica_size
        local_should_commit = enough_replicas and self._errored is None
        should_commit = self._client.should_commit(
            self._group_rank,
            self._step,
            local_should_commit,
            timeout=timeout or self._timeout,
        )
        self._logger.info(
            f"should_commit={should_commit} enough_replicas={enough_replicas}, errored={self._errored}"
        )

        self._checkpoint_transport.disallow_checkpoint()

        # decide whether we're in a healthy state to increase the step count
        if should_commit:
            self._step += 1
            self._batches_committed += self.num_participants()
            self._commit_failures = 0  # Reset failure counter on success
        else:
            self._commit_failures += 1
            # Check if we've hit max retries
            if (
                self._max_retries is not None
                and self._commit_failures > self._max_retries
            ):
                msg = f"should_commit failed {self._commit_failures} times consecutively, exceeding max_retries={self._max_retries}"
                self._logger.exception(msg)
                raise RuntimeError(msg)

        return should_commit

    def load_state_dict(self, state_dict: Dict[str, int]) -> None:
        """
        Load the state dict from a previous checkpoint.

        This will restore the step count and internal metadata.

        Args:
            state_dict: the state dict to load
        """
        self._step = state_dict["step"]
        self._batches_committed = state_dict["batches_committed"]

    def _manager_state_dict(self) -> Dict[str, object]:
        assert self._user_state_dict is not None, "user state_dict is not initialized."
        return {
            "user": self._user_state_dict(),
            "torchft": self.state_dict(),
        }

    def state_dict(self) -> Dict[str, int]:
        """
        Get the state dict for this manager.

        This can be used to checkpoint the state of the manager to restore
        from a previous checkpoint.

        Returns:
            the state dict for this manager
        """
        return {"step": self._step, "batches_committed": self._batches_committed}

    def current_step(self) -> int:
        """
        Get the current step count.

        This number is incremented on .step()

        Returns:
            the current step count
        """
        return self._step

    def batches_committed(self) -> int:
        """
        Get the total number of batches committed across all steps and replicas.
        5 replicas participating in 2 steps is 10 batches but may be more than
        10 examples depending on batch size.

        This number is incremented on .step()

        Returns:
            the total number of batches committed
        """
        return self._batches_committed

    def participating_rank(self) -> Optional[int]:
        """
        Get the replica group rank of the current quorum. This will be the same on all
        ranks within the replica group.

        If this replica group is not participating in the current quorum, this will be None.

        This will block on the async quorum if it is not yet ready.

        Returns:
            the rank of the current quorum
        """
        if self._quorum_future is None:
            return None

        self.wait_quorum()

        return self._participating_replica_rank

    def num_participants(self) -> int:
        """
        Get the number of participants in the current quorum.

        This is the number of replicas participating in the current step.

        This will block on the async quorum if it is not yet ready.

        Returns:
            the number of participants in the current quorum
        """
        if self._quorum_future is None:
            return 0

        self.wait_quorum()

        assert self._participating_replica_world_size >= 0, "internal error"
        return self._participating_replica_world_size

    def is_participating(self) -> bool:
        """
        Get whether this replica is participating in the current quorum.

        Returns:
            whether this replica is participating in the current quorum
        """
        if self._participating_replica_rank is None:
            return False
        if self._healing:
            assert self._use_async_quorum
            return False
        return True


class _ManagerLogger:
    def __init__(self, manager: Manager, replica_id: str, group_rank: int) -> None:
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._replica_id = replica_id
        self._group_rank = group_rank
        self._manager = manager

    def prefix(self) -> str:
        return f"[{self._replica_id}/{self._group_rank} - step {self._manager.current_step()}]"

    def info(self, msg: str) -> None:
        self._logger.info(f"{self.prefix()} {msg}")

    def warn(self, msg: str) -> None:
        self._logger.warn(f"{self.prefix()} {msg}")

    def exception(self, msg: str) -> None:
        self._logger.exception(f"{self.prefix()} {msg}")


def _failure_listener_process_main(
    lighthouse_addr_str: Optional[str],
    connect_timeout: timedelta,
    stop_event: multiprocessing.Event,
    error_pipe: Connection,
    subscribe_timeout: timedelta = timedelta(milliseconds=100),
):
    """
    Background process that monitors lighthouse for failures through gRPC stream (with an iterator interface) and reports them via error_pipe.
    """
    if not lighthouse_addr_str:
        return

    while not stop_event.is_set():
        try:
            lighthouse_client = LighthouseClient(
                lighthouse_addr_str, connect_timeout=connect_timeout
            )
            stream = lighthouse_client.subscribe_failures(timeout=subscribe_timeout)
            while not stop_event.is_set():
                try:
                    note = next(
                        stream
                    )  # This will block until a new item or timeout if stream supports it
                    if note:
                        if stop_event.is_set():
                            break
                        error = Exception(
                            f"Peer failure detected in listener process: replica {note.replica_id} has failed"
                        )
                        error_pipe.send(ExceptionWithTraceback(error))
                except StopIteration:
                    # Stream has ended, break out to outer loop to reconnect
                    if not stop_event.is_set():
                        logging.warning(
                            "Failure Listener: Stream ended unexpectedly, attempting to reconnect..."
                        )
                        break  # Break the inner loop to reconnect
                    else:
                        break
                except Exception as e_stream:
                    if not stop_event.is_set():
                        continue  # Break due to subscribe_timeout. Allows the process to check stop_event again.
                    else:
                        break
                if stop_event.is_set():
                    break
                time.sleep(0.01)  # Prevent CPU thrashing
        except Exception as e_outer:
            if not stop_event.is_set():
                logging.warning(
                    f"Failure Listener: Connection error: {e_outer}, retrying in 1 second..."
                )
                time.sleep(1)
            pass
