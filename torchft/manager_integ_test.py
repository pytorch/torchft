import copy
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, Generator, List, Protocol, Set, Tuple
from unittest import TestCase

import torch
import torch.distributed as dist
from parameterized import parameterized
from torch import nn, optim

from torchft._torchft import LighthouseServer
from torchft.ddp import DistributedDataParallel
from torchft.local_sgd import DiLoCo, LocalSGD
from torchft.manager import Manager
from torchft.optim import OptimizerWrapper
from torchft.process_group import ProcessGroupGloo

logger: logging.Logger = logging.getLogger(__name__)


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


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


class TrainLoop(Protocol):
    def __call__(
        self, rank: int, store_port: int, runner: "Runner"
    ) -> Dict[str, Dict[str, object]]: ...


@dataclass
class Runner:
    replica_id: int
    lighthouse_address: str
    failure_injector: FailureInjector
    train_loop: TrainLoop

    world_size: int = 1
    attempts: int = 3
    manager_args: Dict[str, object] = field(default_factory=dict)
    train_loop_args: Dict[str, Any] = field(default_factory=dict)

    def _replica_main(self) -> List[Dict[str, Dict[str, object]]]:
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
                futures.append(
                    executor.submit(
                        self.train_loop,
                        rank=rank,
                        store_port=store.port,
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

    def run_replica(self) -> List[Dict[str, Dict[str, object]]]:
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

        m: nn.Module = DistributedDataParallel(manager, MyModel())
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


def local_sgd_train_loop(
    rank: int,
    store_port: int,
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

        m: nn.Module = MyModel()
        optimizer: optim.Optimizer = optim.Adam(m.parameters())
        criterion = nn.CrossEntropyLoss()

        with LocalSGD(manager, m, optimizer, sync_every=2):
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


def diloco_train_loop(
    rank: int,
    store_port: int,
    runner: Runner,
) -> Dict[str, Dict[str, object]]:
    with ExitStack() as stack:
        # Declare the model and optimizers
        m: nn.Module = MyModel()
        model_state_dict: Dict[str, Any] = runner.train_loop_args["model_state_dict"]
        m.load_state_dict(model_state_dict)

        # Setup optimizers
        inner_optimizer: optim.Optimizer = torch.optim.AdamW(
            m.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
        )
        outer_optimizer: optim.Optimizer = torch.optim.SGD(
            m.parameters(), lr=0.7, momentum=0.9, nesterov=True
        )

        # pyre-ignore[53]
        def load_state_dict(state_dict: Dict[str, Dict[str, object]]) -> None:
            m.load_state_dict(state_dict["model"])
            # TODO: make this cleaner so we don't have to save this
            diloco._backup_parameters = state_dict["backup_params"]
            inner_optimizer.load_state_dict(state_dict["inner_optim"])
            outer_optimizer.load_state_dict(state_dict["outer_optim"])

        def state_dict() -> Dict[str, Dict[str, object]]:  # pyre-ignore[53]
            return {
                "model": m.state_dict(),
                "backup_params": copy.deepcopy(diloco._backup_parameters),
                "inner_optim": inner_optimizer.state_dict(),
                "outer_optim": outer_optimizer.state_dict(),
            }

        print(f"worker {runner.replica_id=} {rank=} {runner.world_size=} starting")

        pg = ProcessGroupGloo()
        manager = Manager(
            pg=pg,
            min_replica_size=2,
            use_async_quorum=False,
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
        stack.callback(manager.shutdown)

        criterion = nn.CrossEntropyLoss()
        all_state_dicts = {}
        with DiLoCo(
            manager, m, inner_optimizer, outer_optimizer, sync_every=2
        ) as diloco:
            while True:
                inputs = torch.rand(2, 3)
                labels = torch.randint(4, (2,))

                out = m(inputs)
                loss = criterion(out, labels)

                inner_optimizer.zero_grad()
                loss.backward()
                inner_optimizer.step()
                manager_step_str = str(manager.current_step())
                all_state_dicts[manager_step_str] = state_dict()

                # after 4 model updates then break
                if manager.current_step() >= 4:
                    break

                runner.failure_injector.check(rank, manager.current_step())

        # return state_dict so we can check consistency
        return all_state_dicts


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
    def test_ddp_recovery(self, name: str, use_async_quorum: bool) -> None:
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
                state_dicts.append(fut.result())

        lighthouse.shutdown()

        for state_dict in state_dicts:
            torch.testing.assert_close(state_dict, state_dicts[0])

        self.assertEqual(failure_injectors[1].count, 1)

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

    def test_local_sgd_recovery(self) -> None:
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
                    lighthouse_address=lighthouse.address(),
                    failure_injector=failure_injector,
                    train_loop=local_sgd_train_loop,
                    manager_args={
                        "use_async_quorum": False,
                    },
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
            # LocalSGD only guarantees that the model is consistent across
            # replicas but uses separate optimizer states.
            torch.testing.assert_close(
                state_dict[0]["model"], state_dicts[0][0]["model"]
            )

        self.assertEqual(failure_injectors[1].count, 1)

    def test_diloco_healthy(self) -> None:
        lighthouse = Lighthouse(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 2
        futures = []

        torch.manual_seed(42)
        # Initialize the model so we can pass in the state_dict
        m: nn.Module = MyModel()

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id in range(num_replicas):
                failure_injector = FailureInjector()
                runner = Runner(
                    replica_id=replica_id,
                    lighthouse_address=lighthouse.address(),
                    failure_injector=failure_injector,
                    train_loop=diloco_train_loop,
                    train_loop_args={
                        "model_state_dict": m.state_dict(),
                    },
                )
                futures.append(executor.submit(runner.run_replica))

        state_dicts = []

        for fut in as_completed(futures):
            state_dicts.append(fut.result()[0])

        lighthouse.shutdown()

        for replica_group in state_dicts:
            for step, state_dict in replica_group.items():
                # inner optimizer will be different, outer optimizer and model should be the same
                torch.testing.assert_close(
                    state_dict["backup_params"],
                    state_dicts[0][str(step)]["backup_params"],
                )
                torch.testing.assert_close(
                    state_dict["outer_optim"], state_dicts[0][str(step)]["outer_optim"]
                )

    def test_diloco_recovery(self) -> None:
        lighthouse = Lighthouse(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 2
        futures = []

        failure_injectors = [
            FailureInjector(),
            FailureInjector().fail_at(0, 2),
        ]

        torch.manual_seed(42)
        # Initialize the model so we can pass in the state_dict
        m: nn.Module = MyModel()

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id, failure_injector in zip(
                range(num_replicas), failure_injectors
            ):
                runner = Runner(
                    replica_id=replica_id,
                    lighthouse_address=lighthouse.address(),
                    failure_injector=failure_injector,
                    train_loop=diloco_train_loop,
                    train_loop_args={
                        "model_state_dict": m.state_dict(),
                    },
                )
                futures.append(executor.submit(runner.run_replica))

            state_dicts = []

            for fut in as_completed(futures):
                try:
                    state_dicts.append(fut.result()[0])
                except Exception as e:
                    print(e)
                    raise

        lighthouse.shutdown()
        for replica_group in state_dicts:
            for step, state_dict in replica_group.items():
                str_step = str(step)
                if str_step in state_dicts[0]:
                    # inner optimizer will be different, outer optimizer and model should be the same
                    torch.testing.assert_close(
                        state_dict["backup_params"],
                        state_dicts[0][str_step]["backup_params"],
                    )
                    torch.testing.assert_close(
                        state_dict["outer_optim"],
                        state_dicts[0][str_step]["outer_optim"],
                    )

        self.assertEqual(failure_injectors[1].count, 1)

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
