# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
from enum import Enum
import os
from typing import Dict, TYPE_CHECKING
import random
import asyncio
import logging

from monarch.actor import Actor, current_rank, endpoint
import torch

logger = logging.getLogger()

if TYPE_CHECKING:
    from ..train_distributed import Replica, MonarchSlurm


class Failure(Enum):
    """
    A set of supported failures injected by FailureController
    """
    SEGFAULT = 0
    KILL_PROC = 1
    COMMS = 2
    KILL_SLURM = 3
    DEADLOCK = 4


class FailureActor(Actor):
    @endpoint
    def fail(self, failure: Failure) -> None:
        rank = current_rank().rank
        logger.info(f"[FailureController] {rank} chosen to fail with {failure}")
        match failure:
            case Failure.SEGFAULT:
                self.segfault()
            case Failure.KILL_PROC:
                self.kill_proc()
            case Failure.COMMS:
                self.kill_comms()
            case Failure.DEADLOCK:
                self.deadlock()

    def segfault(self):
        """
        Triggers a SIGSEGV on the process
        """
        crash_func = ctypes.CFUNCTYPE(None)()
        crash_func()

    def kill_proc(self):
        """
        Immediately kills the process with non-zero exit code
        """
        os._exit(1)

    def kill_comms(self):
        """
        Forcefully aborts the ProcessGroup and NCCL communicator
        """
        torch.distributed.distributed_c10d._abort_process_group()

    def deadlock(self):
        """
        Deadlocks the GIL, resulting in ProcessGroupNCCL timeout and terminal failure
        """
        libc = ctypes.PyDLL(None)
        libc.sleep.argtypes = tuple([ctypes.c_uint])
        libc.sleep.restype = ctypes.c_uint
        libc.sleep(70)


class FailureController:
    @staticmethod
    def kill_slurm(scheduler: "MonarchSlurm") -> None:
        """
        Kills a random replica SLURM job
        """
        candidates = [
            mesh_name for mesh_name in scheduler.job_handles.keys()
            if "replica_" in mesh_name
        ]
        selected = random.choice(candidates)
        logger.info(f"[FailureController] Killing SLURM job for {selected}")
        scheduler.kill_job(selected)

    @staticmethod
    async def execute_failures(replicas: Dict[int, "Replica"], scheduler: "MonarchSlurm"):
        startup_wait = 30
        rest_time = 60
        logger.info(f"[FailureController] Starting failure injection in {startup_wait} seconds")
        await asyncio.sleep(startup_wait)  # allow startups.

        last_replica = last_failure = None
        while replicas:
            try:
                running_replicas = list(replicas.values())
                # allow deadlocked replicas more time to recover
                if last_failure == Failure.DEADLOCK and last_replica in running_replicas:
                    running_replicas.remove(last_replica)
                
                last_replica = random.choice(running_replicas)
                last_failure = random.choice(list(Failure))
                try:
                    if last_failure == Failure.KILL_SLURM:
                        FailureController.kill_slurm(scheduler)
                    else:
                        last_replica.actor.inject_failure.broadcast(last_failure)
                    logger.info(
                        f"[FailureController] Failure injection ({last_failure}) sent to replica {last_replica.rid}"
                    )
                except Exception as e:
                    logger.info(
                        f"[FailureController] Failed to inject failure in replica {last_replica.rid}: {e}"
                    )
                await asyncio.sleep(rest_time)
            except Exception as e:
                logger.info(f"[FailureController] Something went wrong while injecting failure: {e}")