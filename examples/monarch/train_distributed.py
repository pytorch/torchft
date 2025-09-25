# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import asyncio
import atexit
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

import torch
from monarch._rust_bindings.monarch_hyperactor.alloc import AllocConstraints, AllocSpec
from monarch._src.actor.allocator import RemoteAllocator, TorchXRemoteAllocInitializer
from monarch.actor import Actor, current_rank, endpoint, ProcMesh, this_host
from monarch.tools import commands
from monarch.tools.components import hyperactor
from monarch.tools.config import Config
from monarch.utils import setup_env_for_distributed
from torchtitan.config import ConfigManager, JobConfig
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer
from utils.failure import Failure, FailureActor, FailureController


# ==== Allocation boilerplate - much of this will be upstreamed into Monarch ====
class MonarchSlurm:
    # Cluster Configuration - update these values for your specific cluster
    machine: str = "gpu.xlarge"
    machine_memory: int = 2062607
    job_name_prefix: str = "monarch-torchft"

    def __init__(self):
        self.job_handles: Dict[str, str] = {}
        atexit.register(self.kill_jobs)

    def get_config(self, mesh_name: str, nodes_per_mesh: int) -> Config:
        mesh = [f"{mesh_name}:{nodes_per_mesh}:{MonarchSlurm.machine}"]
        # to enable relative import of utils on actors
        current_dir = os.path.dirname(os.path.abspath(__file__))
        env = {"PYTHONPATH": current_dir}

        appdef = hyperactor.host_mesh(meshes=mesh, env=env)

        for role in appdef.roles:
            role.resource.memMB = MonarchSlurm.machine_memory

        return Config(scheduler="slurm", appdef=appdef)

    async def get_or_create_job(self, mesh_name: str, nodes_per_mesh: int = 1) -> None:
        config = self.get_config(mesh_name, nodes_per_mesh)
        job_name = f"{MonarchSlurm.job_name_prefix}-{mesh_name}"
        server_spec = await commands.get_or_create(job_name, config, force_restart=True)
        self.job_handles[mesh_name] = server_spec.name

    def kill_jobs(self):
        for mesh_name in self.job_handles.keys():
            self.kill_job(mesh_name)

    def kill_job(self, mesh_name: str):
        try:
            job_handle = self.job_handles[mesh_name]
            logger.info(f"Destroying job for mesh {mesh_name}")
            commands.kill(f"slurm:///{job_handle}")
        except Exception as e:
            logger.warning(f"Failed to destroy job for {mesh_name}: {e}")

    def proc_mesh(
        self,
        mesh_name: str,
        num_hosts: int = 1,
        num_gpus: int = 8,
    ) -> ProcMesh:
        allocator = RemoteAllocator(
            world_id=MonarchSlurm.job_name_prefix,
            initializer=TorchXRemoteAllocInitializer(
                f"slurm:///{self.job_handles[mesh_name]}"
            ),
        )
        alloc = allocator.allocate(
            AllocSpec(AllocConstraints(), hosts=num_hosts, gpus=num_gpus)
        )

        return ProcMesh.from_alloc(alloc)


# ==== allocation boilerplate ====


class LighthouseActor(Actor):
    def __init__(self) -> None:
        self.lighthouse = None

    @endpoint
    def start_lighthouse(self) -> str:
        # inline import because of https://github.com/meta-pytorch/monarch/issues/804
        from torchft.coordination import LighthouseServer

        self.lighthouse = LighthouseServer(
            bind="[::]:0", min_replicas=1, join_timeout_ms=10000
        )
        return self.lighthouse.address()

    @endpoint
    def stop_lighthouse(self) -> None:
        if not self.lighthouse:
            raise RuntimeError("Lighthouse not started!")
        self.lighthouse.shutdown()


class TrainingActor(Actor):
    def __init__(self, job_config: JobConfig, replica_id: int) -> None:
        self.job_config = job_config
        rank = current_rank().rank
        self.uid = f"[replica_{replica_id}_trainer_{rank}]"

    @endpoint
    async def start_training(self, lighthouse_address: str) -> None:
        init_logger()

        os.environ["TORCHFT_LIGHTHOUSE"] = lighthouse_address
        trainer = Trainer(self.job_config)
        logger.info(f"{self.uid} initialized successfully on {os.getpid()}")

        try:
            logger.info(f"{self.uid} starting training")
            trainer.train()
        except Exception:
            if trainer:
                trainer.close()
            raise
        else:
            trainer.close()
        finally:
            torch.distributed.destroy_process_group()
            logger.info(f"{self.uid} trainer cleaned up")


@dataclass
class JobSpec:
    job_config: JobConfig
    remote_lighthouse: bool
    replica_count: int
    hosts_per_replica: int
    gpus_per_node: int
    with_failures: bool
    lighthouse_address: str = ""


@dataclass
class Replica:
    rid: int
    proc_mesh: ProcMesh
    actor: "ReplicaActor"
    attempt_number: int = 0


# This does not currently benefit from being an actor, but will once
# Monarch supervision APIs are fleshed out.
class ReplicaActor(Actor):
    def __init__(self, spec: JobSpec, replica_id: int, scheduler: MonarchSlurm) -> None:
        self.spec = deepcopy(spec)
        self.replica_id = replica_id

        self.uid = f"[replica_{replica_id}]"
        self.spec.job_config.fault_tolerance.replica_id = self.replica_id
        self.scheduler = scheduler

        self.failure_actors: FailureActor | None = None

    @endpoint
    async def start_replica(self) -> None:
        init_logger()
        logger.info(f"{self.uid} Spawning trainers")

        trainers_proc_mesh: ProcMesh | None = None
        try:
            trainers_proc_mesh = self.scheduler.proc_mesh(
                f"replica_{self.replica_id}",
                self.spec.hosts_per_replica,
                self.spec.gpus_per_node,
            )
            await trainers_proc_mesh.logging_option(
                stream_to_client=True, aggregate_window_sec=None
            )
            await setup_env_for_distributed(trainers_proc_mesh)

            training_actors = trainers_proc_mesh.spawn(
                "training_actors",
                TrainingActor,
                self.spec.job_config,
                self.replica_id,
            )

            self.failure_actors = trainers_proc_mesh.spawn(
                "failure_actors", FailureActor
            )

            logger.info(f"{self.uid} Starting trainers")
            await training_actors.start_training.call(self.spec.lighthouse_address)
            await trainers_proc_mesh.stop()
        except Exception as e:
            if trainers_proc_mesh:
                await trainers_proc_mesh.stop()
            raise e

    @endpoint
    async def inject_failure(self, failure_type: Failure):
        if self.failure_actors:
            try:
                logger.info(
                    f"{self.uid} Injecting failure ({failure_type}) into random trainer"
                )

                await self.failure_actors.fail.choose(failure_type)
            except Exception as e:
                error_msg = f"{self.uid} Injected failure: {e}"
                logger.error(error_msg)
        else:
            error_msg = f"{self.uid} No failure actors available"
            logger.error(error_msg)


# delay before re-creating proc mesh on existing job. change as needed.
PROC_ATTEMPT_DELAY = 0
# proc attempts before getting a new scheduler allocation. change as needed.
PROC_ATTEMPTS = 3
# attempts before failing training on replica. change as needed.
MAX_ATTEMPT = PROC_ATTEMPTS * 3


class OrchestrationManager:
    def __init__(self, spec: JobSpec) -> None:
        self.spec = spec
        self.replicas: Dict[int, Replica] = {}
        self.lighthouse_actor: LighthouseActor | None = None
        self.lighthouse_mesh: ProcMesh | None = None

        self.scheduler = MonarchSlurm()

    async def start_training(self) -> None:
        logger.info(
            f"[Controller] Creating training system with {self.spec.replica_count} replicas"
        )

        for replica_id in range(self.spec.replica_count):
            await self.scheduler.get_or_create_job(
                f"replica_{replica_id}", self.spec.hosts_per_replica
            )

        mesh_futures = {}
        for i in range(self.spec.replica_count):
            mesh_futures[i] = asyncio.create_task(self._run_replica(i, 0))

        failure_future = None
        if self.spec.with_failures:
            failure_future = asyncio.create_task(
                FailureController.execute_failures(self.replicas, self.scheduler)
            )

        await asyncio.gather(*mesh_futures.values(), return_exceptions=True)

        if failure_future:
            failure_future.cancel()

    async def start_lighthouse(self) -> None:
        if self.spec.remote_lighthouse:
            await self.scheduler.get_or_create_job("lighthouse")
            self.lighthouse_mesh = self.scheduler.proc_mesh("lighthouse", num_gpus=1)
        else:
            self.lighthouse_mesh = this_host().spawn_procs({"gpus": 1})

        await self.lighthouse_mesh.logging_option(
            stream_to_client=True, aggregate_window_sec=None
        )
        self.lighthouse_actor = self.lighthouse_mesh.spawn(
            "lighthouse_actor", LighthouseActor
        )
        self.spec.lighthouse_address = (
            await self.lighthouse_actor.start_lighthouse.call_one()
        )

    async def stop_lighthouse(self) -> None:
        try:
            if self.lighthouse_mesh:
                await self.lighthouse_actor.stop_lighthouse.call_one()
                await self.lighthouse_mesh.stop()
            logger.info("[Controller] Lighthouse stopped")
        except Exception as e:
            logger.warning(f"[Controller] Failed to stop lighthouse: {e}")

    async def _run_replica(self, replica_id: int, attempt_number: int) -> None:
        if attempt_number >= MAX_ATTEMPT:
            logger.info(f"[Controller] Replica {replica_id} has failed too many times.")
            return

        try:
            await self._spin_up_replica(replica_id, attempt_number)
            logger.info(f"[Controller] replica {replica_id} done")
            await self._teardown(replica_id)
        except Exception as e:
            await self._teardown(replica_id)
            logger.info(f"[Controller] replica {replica_id} failed: {e}")
            await self._run_replica(replica_id, attempt_number + 1)

    async def _spin_up_replica(self, replica_id: int, attempt_number: int = 0) -> None:
        if attempt_number != 0 and attempt_number % PROC_ATTEMPTS == 0:
            logger.info(
                f"[Controller] Replica {replica_id} has failed {attempt_number} times. Getting new allocation."
            )
            self.scheduler.kill_job(f"replica_{replica_id}")
            await self.scheduler.get_or_create_job(
                f"replica_{replica_id}", self.spec.hosts_per_replica
            )
        delay = 0 if not attempt_number else PROC_ATTEMPT_DELAY
        logger.info(
            f"[Controller] Spinning up replica with ID {replica_id} in {delay} seconds"
        )
        await asyncio.sleep(delay)

        replica_proc_mesh = this_host().spawn_procs({"gpus": 1})
        await replica_proc_mesh.logging_option(aggregate_window_sec=None)

        replica_actor = replica_proc_mesh.spawn(
            "replica_actor", ReplicaActor, self.spec, replica_id, self.scheduler
        )

        replica = Replica(replica_id, replica_proc_mesh, replica_actor, attempt_number)
        self.replicas[replica_id] = replica
        await replica.actor.start_replica.call_one()

    async def _teardown(self, replica_id: int) -> None:
        try:
            replica = self.replicas[replica_id]
            await replica.proc_mesh.stop()
            del replica.proc_mesh
            del self.replicas[replica_id]
        except Exception as e:
            logger.error(f"[Controller] Failed to _teardown replica {replica_id}: {e}")


# === CLI / CONFIG === #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monarch-TorchFT Distributed Training Example"
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument(
        "--replica-count", type=int, default=2, help="Number of replicas (default: 2)"
    )
    parser.add_argument(
        "--gpu-per-node", type=int, default=8, help="GPUs per replica (default: 8)"
    )
    parser.add_argument(
        "--host-per-replica", type=int, default=1, help="Hosts per replica (default: 1)"
    )
    parser.add_argument(
        "--remote-lighthouse",
        action="store_true",
        help="Run the LighthouseServer on a worker node (default: False)",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=50,
        help="Number of training steps (default: 50)",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="debug_model.toml",
        help=f"Relative path to model configuration file (default: {os.path.join(script_dir, 'debug_model.toml')})",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="c4_test",
        help=f"Relative path to training dataset (default: {os.path.join(script_dir, 'c4_test')})",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="debug_tokenizer",
        help=f"Relative path to tokenizer (default: {os.path.join(script_dir, 'debug_tokenizer')})",
    )
    parser.add_argument(
        "--with-failures",
        action="store_true",
        help="Enable the failure injector utility (default: False)",
    )

    return parser.parse_args()


def make_job_spec(args: argparse.Namespace) -> JobSpec:
    data_parallel_shard_degree = args.gpu_per_node * args.host_per_replica

    output_path = "./outputs"
    training_dataset = args.dataset_path.split("/")[-1]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_args = [
        "--job.config_file",
        os.path.join(script_dir, args.model_config),
        "--model.tokenizer_path",
        os.path.join(script_dir, args.tokenizer_path),
        "--comm.trace_buf_size",
        "0",
        "--metrics.log_freq",
        "1",
        "--fault_tolerance.enable",
        "--fault_tolerance.group_size",
        str(args.replica_count),
        "--parallelism.data_parallel_shard_degree",
        str(data_parallel_shard_degree),
        "--activation_checkpoint.mode",
        "full",
        "--comm.train_timeout_seconds",
        "60",
        "--training.steps",
        str(args.training_steps),
        "--training.dataset",
        training_dataset,
        "--training.dataset_path",
        os.path.join(script_dir, args.dataset_path),
        "--job.dump_folder",
        output_path,
        "--metrics.enable_tensorboard",
    ]

    config_manager = ConfigManager()
    job_config = config_manager.parse_args(default_args)

    return JobSpec(
        job_config=job_config,
        remote_lighthouse=args.remote_lighthouse,
        replica_count=args.replica_count,
        hosts_per_replica=args.host_per_replica,
        gpus_per_node=args.gpu_per_node,
        with_failures=args.with_failures,
    )


# === CLI / CONFIG === #


async def main() -> None:
    init_logger()

    args = parse_args()
    job_spec = make_job_spec(args)

    orchestrator = OrchestrationManager(job_spec)
    try:
        await orchestrator.start_lighthouse()
        await orchestrator.start_training()
    finally:
        await orchestrator.stop_lighthouse()


if __name__ == "__main__":
    asyncio.run(main())
