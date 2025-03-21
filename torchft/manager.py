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
import os
import socket
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, TypeVar, cast

import torch
from torch.distributed import ReduceOp, TCPStore

from torchft._torchft import ManagerClient, ManagerServer
from torchft.checkpointing import CheckpointTransport, HTTPTransport
from torchft.futures import future_timeout

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
            rank: the replica group local rank
            world_size: the replica group local world size
            store_addr: TCPStore address for this replica group
            store_port: TCPStore port for this replica group
            lighthouse_addr: if rank==0, the address of the lighthouse server
            replica_id: if rank==0, the replica_id for this group
            hostname: if rank==0, the hostname to advertise to the lighthouse server
            checkpoint_transport: the checkpoint transport to use for
                transfering checkpoints to recovering replicas, defaults to HTTPTransport
        """
        self._load_state_dict = load_state_dict
        self._user_state_dict = state_dict
        self._pending_state_dict: Optional[Dict[str, object]] = None
        self._use_async_quorum = use_async_quorum
        self._timeout = timeout
        self._quorum_timeout = quorum_timeout
        self._connect_timeout = connect_timeout
        self._world_size_mode = world_size_mode

        store_addr = store_addr or os.environ["MASTER_ADDR"]
        store_port = store_port or int(os.environ["MASTER_PORT"])
        self._rank: int = rank if rank is not None else int(os.environ["RANK"])
        rank = self._rank
        world_size = world_size or int(os.environ["WORLD_SIZE"])
        self._min_replica_size = min_replica_size

        if checkpoint_transport is None:
            checkpoint_transport = HTTPTransport[Dict[str, T]](
                timeout=timeout,
                num_chunks=0,
            )

        self._checkpoint_transport: CheckpointTransport[Dict[str, T]] = (
            checkpoint_transport
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="async_quorum"
        )
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

        if rank == 0:
            if port is None:
                port = int(os.environ.get(MANAGER_PORT_ENV, 0))

            bind = f"[::]:{port}"
            lighthouse_addr = lighthouse_addr or os.environ["TORCHFT_LIGHTHOUSE"]

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
                world_size=world_size,
                heartbeat_interval=heartbeat_interval,
                connect_timeout=connect_timeout,
            )

            self._store.set(MANAGER_ADDR_KEY, self._manager.address())
            self._store.set(REPLICA_ID_KEY, replica_id)

        addr = self._store.get(MANAGER_ADDR_KEY).decode("utf-8")
        self._client = ManagerClient(addr, connect_timeout=connect_timeout)

        replica_id = self._store.get(REPLICA_ID_KEY).decode("utf-8")
        self._logger = _ManagerLogger(
            manager=self, replica_id=replica_id or "", rank=rank
        )

        self._step = 0
        self._quorum_id = -1
        self._errored: Optional[Exception] = None
        self._healing = False
        self._pending_work: List[torch.futures.Future[object]] = []
        self._batches_committed = 0

        # first step is 1
        self._participating_rank: Optional[int] = None
        self._participating_world_size: int = 0

    def set_state_dict_fns(
        self, load_state_dict: Callable[[T], None], state_dict: Callable[[], T]
    ) -> None:
        self._load_state_dict = load_state_dict
        self._user_state_dict = state_dict

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the manager and checkpoint server.
        """
        self._checkpoint_transport.shutdown(wait=wait)
        if self._manager is not None:
            self._manager.shutdown()
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
        self._errored = e

    def errored(self) -> Optional[Exception]:
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
        quorum = self._client._quorum(
            rank=self._rank,
            step=self._step,
            checkpoint_metadata=self._checkpoint_transport.metadata(),
            shrink_only=shrink_only,
            timeout=quorum_timeout,
        )

        quorum_id = quorum.quorum_id
        replica_rank = quorum.replica_rank
        replica_world_size = quorum.replica_world_size
        recover_src_manager_address = quorum.recover_src_manager_address
        store_address = quorum.store_address
        max_step = quorum.max_step
        max_rank = quorum.max_rank
        max_world_size = quorum.max_world_size
        heal = quorum.heal

        # When using async quorum we need to take the recovered workers.
        # When not using async quorum we need to take the max world size as all
        # workers will be healthy.
        self._participating_rank, self._participating_world_size = (
            (max_rank, max_world_size)
            if self._use_async_quorum or not allow_heal
            else (replica_rank, replica_world_size)
        )

        # For fixed with spares we need to ensure that we don't have more
        # participating replicas than the min replica size.
        if self._world_size_mode == WorldSizeMode.FIXED_WITH_SPARES:
            self._participating_world_size = min(
                self._participating_world_size, self._min_replica_size
            )
            if (
                self._participating_rank is not None
                and self._participating_rank >= self._min_replica_size
            ):
                self._participating_rank = None

        if quorum_id != self._quorum_id:
            store_prefixed_addr = f"{store_address}/torchft/{quorum_id}/{self._rank}"

            self._logger.info(f"reconfiguring for {quorum_id=} {store_prefixed_addr=}")
            # We use the replica rank and world as we want all replicas in the PG.
            # TODO: handle configure errors
            self._pg.configure(store_prefixed_addr, replica_rank, replica_world_size)
            self._quorum_id = quorum_id

        if allow_heal:
            # run recovery on the recovery stream if available
            recovery_stream = self._recovery_stream
            with (
                torch.cuda.stream(recovery_stream)
                if recovery_stream is not None
                else nullcontext()
            ):
                if quorum.recover_dst_ranks:
                    self._logger.info(
                        f"peers need recovery from us {quorum.recover_dst_ranks}"
                    )
                    self._checkpoint_transport.send_checkpoint(
                        dst_ranks=quorum.recover_dst_ranks,
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
                        self._rank, timeout=self._timeout
                    )
                    recover_src_rank = quorum.recover_src_rank
                    assert (
                        recover_src_rank is not None
                    ), "must have a recover rank when healing"

                    self._logger.info(
                        f"fetching checkpoint from {recover_src_rank=} with {checkpoint_metadata=}"
                    )

                    # we apply the user state dict only when safe from the main thread
                    # save it for now
                    self._pending_state_dict = (
                        self._checkpoint_transport.recv_checkpoint(
                            src_rank=recover_src_rank,
                            metadata=checkpoint_metadata,
                            step=max_step,
                            timeout=self._timeout,
                        )
                    )

                    # pyre-fixme[6]: got object
                    self.load_state_dict(self._pending_state_dict["torchft"])

                    # This isn't strictly needed as loading the state_dict above should
                    # restore the correct step but it makes writing tests simpler.
                    self._step = max_step

    def _apply_pending_state_dict(self) -> None:
        assert self._healing, "must be in healing state"

        # synchronize on future
        assert self._quorum_future is not None, "must call step before should_commit"
        self._quorum_future.result()

        self._logger.info("applying pending state dict")

        assert self._pending_state_dict is not None, "checkpoint was not staged"
        assert (
            self._load_state_dict is not None
        ), "user load_state_dict is not initialized."
        self._load_state_dict(self._pending_state_dict["user"])
        self._pending_state_dict = None
        self._logger.info("Loaded state dict.")

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

        Returns:
            True if the optimizer should be stepped, False otherwise
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
            self._rank,
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

        return self._participating_rank

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

        assert self._participating_world_size >= 0, "internal error"
        return self._participating_world_size

    def is_participating(self) -> bool:
        """
        Get whether this replica is participating in the current quorum.

        Returns:
            whether this replica is participating in the current quorum
        """
        if self._participating_rank is None:
            return False
        if self._healing:
            assert self._use_async_quorum
            return False
        return True


class _ManagerLogger:
    def __init__(self, manager: Manager, replica_id: str, rank: int) -> None:
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._replica_id = replica_id
        self._rank = rank
        self._manager = manager

    def prefix(self) -> str:
        return (
            f"[{self._replica_id}/{self._rank} - step {self._manager.current_step()}]"
        )

    def info(self, msg: str) -> None:
        self._logger.info(f"{self.prefix()} {msg}")

    def warn(self, msg: str) -> None:
        self._logger.warn(f"{self.prefix()} {msg}")

    def exception(self, msg: str) -> None:
        self._logger.exception(f"{self.prefix()} {msg}")
