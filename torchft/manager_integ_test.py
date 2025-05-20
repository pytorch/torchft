import copy
import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, Generator, List, Optional, Protocol, Set, Tuple, TypeVar
from unittest import TestCase

import torch
import torch.distributed as dist
from parameterized import parameterized
from torch import nn, optim
from torch._dynamo.utils import timed

from torchft._torchft import LighthouseServer
from torchft.ddp import DistributedDataParallel
from torchft.local_sgd import DiLoCo, LocalSGD
from torchft.manager import Manager
from torchft.optim import OptimizerWrapper
from torchft.process_group import ProcessGroupBabyNCCL, ProcessGroupGloo

logger: logging.Logger = logging.getLogger(__name__)

INIT_LOCK: threading.Lock = threading.Lock()


class MyModel(nn.Module):
    def __init__(self, in_dim: int = 3, out_dim: int = 4) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layers = nn.ModuleList(
            [
                nn.Linear(in_dim, 8),
                nn.ReLU(),
                nn.Linear(8, out_dim),
                nn.ReLU(),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def get_rand_inputs(
        self, batch_size: int, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        return torch.rand(batch_size, self.in_dim, device=device)

    def get_rand_labels(
        self, batch_size: int, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        return torch.randint(3, (batch_size,), device=device)


class InjectedFailure(Exception):
    pass


class FailureInjector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._failures: Set[Tuple[int, int]] = set()
        self.count = 0

    def fail_at(self, rank: int, step: int) -> "FailureInjector":
        with self._lock:
            self._failures.add((rank, step))
            return self

    def check(self, rank: int, step: int) -> None:
        with self._lock:
            key = (rank, step)
            if key in self._failures:
                self.count += 1
                self._failures.remove(key)
                print(f"injecting failure {rank=} {step=}")
                raise InjectedFailure(f"injected failure {rank=} {step=}")


# R for an arbitrary return type
R = TypeVar("R", covariant=True)


class TrainLoop(Protocol[R]):
    def __call__(
        self, rank: int, store_port: int, device: torch.device, runner: "Runner"
    ) -> R: ...


@dataclass
class Runner:
    replica_id: int
    num_replicas: int
    lighthouse_address: str
    failure_injector: FailureInjector
    train_loop: TrainLoop[object]

    use_cuda: bool = False
    world_size: int = 1
    attempts: int = 3
    manager_args: Dict[str, object] = field(default_factory=dict)
    train_loop_args: Dict[str, Any] = field(default_factory=dict)

    def _replica_main(self) -> List[object]:
        store = dist.TCPStore(
            host_name="localhost",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )

        with ThreadPoolExecutor(
            max_workers=self.world_size, thread_name_prefix=f"replica{self.replica_id}"
        ) as executor:
            futures = []
            for rank in range(self.world_size):
                if self.use_cuda:
                    num_cuda_devices = torch.cuda.device_count()
                    assert num_cuda_devices >= self.num_replicas
                    device_index = (
                        num_cuda_devices // self.num_replicas
                    ) * self.replica_id + rank
                    device = torch.device(f"cuda:{device_index}")
                else:
                    device = torch.device("cpu")

                futures.append(
                    executor.submit(
                        self.train_loop,
                        rank=rank,
                        store_port=store.port,
                        device=device,
                        runner=self,
                    )
                )

            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    logger.exception(f"worker {self.replica_id=} threw exception: {e}")
                    raise

            return [fut.result() for fut in futures]

    def run_replica(self) -> List[object]:
        for i in range(self.attempts):
            try:
                print(
                    f"starting replica group {self.replica_id=} {self.world_size=} attempt {i}"
                )
                return self._replica_main()
            except InjectedFailure as e:
                print("got injected failure", i, e)
                if i == self.attempts - 1:
                    raise
                continue

        raise RuntimeError("ran out of attempts")


def ddp_train_loop(
    rank: int,
    store_port: int,
    device: torch.device,
    runner: Runner,
) -> Dict[str, Dict[str, object]]:
    with ExitStack() as stack:

        def load_state_dict(state_dict: Dict[str, Dict[str, object]]) -> None:
            m.load_state_dict(state_dict["model"])
            optimizer.load_state_dict(state_dict["optim"])

        def state_dict() -> Dict[str, Dict[str, object]]:
            return {
                "model": m.state_dict(),
                "optim": optimizer.state_dict(),
            }

        print(f"worker {runner.replica_id=} {rank=} {runner.world_size=} starting")

        pg = ProcessGroupGloo()
        manager = Manager(
            pg=pg,
            min_replica_size=2,
            load_state_dict=load_state_dict,
            state_dict=state_dict,
            replica_id=str(runner.replica_id),
            store_addr="localhost",
            store_port=store_port,
            rank=rank,
            world_size=runner.world_size,
            lighthouse_addr=runner.lighthouse_address,
            port=19530 + runner.replica_id,
            # pyre-fixme[6]: Incompatible parameter type
            **runner.manager_args,
        )
        stack.callback(lambda: manager.shutdown(wait=False))

        with INIT_LOCK:
            # We need to lock during init for testing init_sync=False as all
            # threads share the same RNG
            torch.manual_seed(42)
            m: nn.Module = MyModel()

        m: nn.Module = DistributedDataParallel(manager, m)
        optimizer: optim.Optimizer = OptimizerWrapper(
            manager, optim.Adam(m.parameters())
        )
        criterion = nn.CrossEntropyLoss()

        while True:
            inputs = torch.rand(2, 3)
            labels = torch.randint(4, (2,))

            optimizer.zero_grad()
            out = m(inputs)
            loss = criterion(out, labels)

            loss.backward()

            optimizer.step()

            if manager.current_step() >= 4:
                break

            runner.failure_injector.check(rank, manager.current_step())

        # return state_dict so we can check consistency
        return state_dict()


class ManagerIntegTest(TestCase):
    @contextmanager
    def assertElapsedLessThan(
        self, timeout: float, msg: str = ""
    ) -> Generator[None, None, None]:
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, timeout, msg)

    def test_ddp_healthy(self) -> None:
        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 2
        futures = []

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id in range(num_replicas):
                failure_injector = FailureInjector()
                runner = Runner(
                    replica_id=replica_id,
                    num_replicas=num_replicas,
                    lighthouse_address=lighthouse.address(),
                    failure_injector=failure_injector,
                    train_loop=ddp_train_loop,
                )
                futures.append(executor.submit(runner.run_replica))

        state_dicts = []

        for fut in as_completed(futures):
            state_dicts.append(fut.result())

        lighthouse.shutdown()

        for state_dict in state_dicts:
            torch.testing.assert_close(state_dict, state_dicts[0])

    @parameterized.expand(
        [
            (
                "async_quorum",
                True,
            ),
            (
                "sync_quorum",
                False,
            ),
        ]
    )
    def test_ddp_recovery(
        self,
        name: str,
        use_async_quorum: bool,
    ) -> None:
        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 2
        futures = []

        failure_injectors = [
            FailureInjector(),
            FailureInjector().fail_at(0, 2),
        ]

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id, failure_injector in zip(
                range(num_replicas), failure_injectors
            ):
                runner = Runner(
                    replica_id=replica_id,
                    num_replicas=num_replicas,
                    lighthouse_address=lighthouse.address(),
                    failure_injector=failure_injector,
                    manager_args={
                        "use_async_quorum": use_async_quorum,
                    },
                    train_loop=ddp_train_loop,
                )
                futures.append(executor.submit(runner.run_replica))

            state_dicts = []

            for fut in as_completed(futures):
                try:
                    state_dicts.append(fut.result())
                except Exception as e:
                    print(e)
                    raise

        lighthouse.shutdown()

        for state_dict in state_dicts:
            torch.testing.assert_close(state_dict, state_dicts[0])

        self.assertEqual(failure_injectors[1].count, 1)

    def test_ddp_skip_init_sync(
        self,
    ) -> None:
        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 2
        futures = []

        # no failures
        failure_injectors = [
            FailureInjector(),
            FailureInjector(),
        ]

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id, failure_injector in zip(
                range(num_replicas), failure_injectors
            ):
                runner = Runner(
                    replica_id=replica_id,
                    num_replicas=num_replicas,
                    lighthouse_address=lighthouse.address(),
                    failure_injector=failure_injector,
                    manager_args={
                        "use_async_quorum": False,
                        "init_sync": False,
                    },
                    train_loop=ddp_train_loop,
                )
                futures.append(executor.submit(runner.run_replica))

            state_dicts = []

            for fut in as_completed(futures):
                try:
                    state_dicts.append(fut.result())
                except Exception as e:
                    print(e)
                    raise

        lighthouse.shutdown()

        for state_dict in state_dicts:
            torch.testing.assert_close(state_dict, state_dicts[0])

    def test_ddp_recovery_multi_rank(self) -> None:
        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 2
        world_size = 2
        futures = []

        failure_injectors = [
            FailureInjector(),
            FailureInjector().fail_at(0, 2).fail_at(1, 2),
        ]

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id, failure_injector in zip(
                range(num_replicas), failure_injectors
            ):
                runner = Runner(
                    replica_id=replica_id,
                    num_replicas=num_replicas,
                    lighthouse_address=lighthouse.address(),
                    failure_injector=failure_injector,
                    world_size=world_size,
                    train_loop=ddp_train_loop,
                )
                futures.append(executor.submit(runner.run_replica))

            state_dicts = []

            for fut in as_completed(futures):
                try:
                    state_dicts.append(fut.result())
                except Exception as e:
                    print(e)
                    raise

        lighthouse.shutdown()

        for state_dict in state_dicts:
            torch.testing.assert_close(state_dict, state_dicts[0])

    def test_quorum_timeout(self) -> None:
        with ExitStack() as stack:
            lighthouse = LighthouseServer(
                bind="[::]:0",
                min_replicas=2,
            )
            stack.callback(lighthouse.shutdown)

            store = dist.TCPStore(
                host_name="localhost",
                port=0,
                is_master=True,
                wait_for_workers=False,
            )

            pg = ProcessGroupGloo()
            manager = Manager(
                pg=pg,
                min_replica_size=2,
                load_state_dict=lambda x: None,
                state_dict=lambda: None,
                store_addr="localhost",
                store_port=store.port,
                rank=0,
                world_size=2,
                lighthouse_addr=lighthouse.address(),
                port=19530,
                use_async_quorum=False,
            )
            stack.callback(lambda: manager.shutdown(wait=False))

            with self.assertElapsedLessThan(1.0):
                with self.assertRaisesRegex(
                    TimeoutError,
                    "status: Cancelled, message.*Timeout expired",
                ):
                    manager.start_quorum(timeout=timedelta(seconds=0.01))

            with self.assertElapsedLessThan(1.0):
                with self.assertRaisesRegex(
                    TimeoutError,
                    "status: Cancelled, message.*Timeout expired",
                ):
                    manager.should_commit(timeout=timedelta(seconds=0.01))

    @parameterized.expand(
        [
            (True,),  # Test with CUDA
            (False,),  # Test without CUDA (CPU)
        ]
    )
    def test_manager_allreduce(self, use_cuda: bool) -> None:
        # Skip the test if use_cuda is True and there are not enough GPUs
        if use_cuda and torch.cuda.device_count() < 2:
            self.skipTest("Not enough GPUs for CUDA test")

        # manager supports allreduce but we found an issue where the future callback is getting called
        # before the allreduce is complete. This test is to ensure that the callback has stream synchronization
        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 2
        futures = []

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id in range(num_replicas):
                failure_injector = FailureInjector()
                runner = Runner(
                    replica_id=replica_id,
                    num_replicas=num_replicas,
                    lighthouse_address=lighthouse.address(),
                    failure_injector=failure_injector,
                    train_loop=all_reduce_callback,
                    use_cuda=use_cuda,
                )
                futures.append(executor.submit(runner.run_replica))

        results = []
        for fut in as_completed(futures):
            try:
                results.append(fut.result()[0])
            except Exception as e:
                print(e, flush=True)
                traceback.print_exc()
                raise

        lighthouse.shutdown()

        print(results)
        r0, r1 = results
        torch.testing.assert_close(r0, r1, check_device=False)


def all_reduce_callback(
    rank: int,
    store_port: int,
    device: torch.device,
    runner: Runner,
) -> Optional[torch.Tensor]:
    with ExitStack() as stack:
        print(f"worker {runner.replica_id=} {rank=} {runner.world_size=} starting")

        if device.type == "cuda":
            pg = ProcessGroupBabyNCCL()
        else:
            pg = ProcessGroupGloo()
        manager = Manager(
            pg=pg,
            min_replica_size=2,
            use_async_quorum=False,
            load_state_dict=lambda x: None,
            state_dict=lambda: None,
            replica_id=str(runner.replica_id),
            store_addr="localhost",
            store_port=store_port,
            rank=rank,
            world_size=runner.world_size,
            lighthouse_addr=runner.lighthouse_address,
            port=19530 + runner.replica_id,
            timeout=timedelta(seconds=10),
            quorum_timeout=timedelta(seconds=10),
            # pyre-fixme[6]: Incompatible parameter type
            **runner.manager_args,
        )
        stack.callback(lambda: manager.shutdown(wait=False))

        manager.start_quorum()
        t1 = torch.ones((1, 3), device=device)
        fut = manager.allreduce(t1)
        fut.wait()
        return t1
    return None
