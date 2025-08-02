import copy
import logging
import os
import re
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import field
from datetime import timedelta
from typing import Any, Dict, cast
from unittest import TestCase, skipIf

import torch
from parameterized import parameterized
from torch import nn, optim
from torch.distributed.pipelining import SplitPoint, pipeline
from torch.distributed.tensor import DTensor, Replicate

from torchft._test.diloco_trainer import DiLoCoTrainer, MultiMyModel
from torchft._torchft import LighthouseServer
from torchft.device_mesh import ft_init_device_mesh
from torchft.local_sgd import DiLoCo, LocalSGD
from torchft.manager import Manager
from torchft.manager_integ_test import (
    EventInjector,
    EventInjectorEvent,
    MyModel,
    Runner,
)
from torchft.process_group import (
    FakeProcessGroupWrapper,
    ProcessGroupBabyNCCL,
    ProcessGroupGloo,
)

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def local_sgd_train_loop(
    rank: int,
    store_port: int,
    device: torch.device,
    runner: Runner,
    train_loop_args: dict[str, Any] = {},
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

        if device.type == "cuda":
            pg = ProcessGroupBabyNCCL()
        else:
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
            timeout=timedelta(seconds=10),
            # pyre-fixme[6]: Incompatible parameter type
            **runner.manager_args,
        )
        stack.callback(lambda: manager.shutdown(wait=False))

        m: nn.Module = MyModel().to(device)

        optimizer: optim.Optimizer = optim.Adam(m.parameters())
        criterion = nn.CrossEntropyLoss()

        with LocalSGD(manager, m, optimizer, sync_every=2) as local_sgd:
            while True:
                inputs = torch.rand(2, 3).to(device)
                labels = torch.randint(4, (2,)).to(device)

                optimizer.zero_grad()
                out = m(inputs)
                loss = criterion(out, labels)
                loss.backward()

                optimizer.step()

                if manager.current_step() >= 4:
                    break

                runner.event_injector.check(rank, manager.current_step())

        # return state_dict so we can check consistency
        return state_dict()
    return {}


def diloco_train_loop(
    rank: int,
    store_port: int,
    device: torch.device,
    runner: Runner,
    train_loop_args: dict[str, Any] = {},
) -> Dict[str, Dict[str, object]]:

    model_state_dict = train_loop_args.get("model_state_dict", {})
    n_fragments = train_loop_args.get("n_fragments", 1)
    diloco_args = train_loop_args.get("diloco_args", {})

    with ExitStack() as stack:
        trainer = DiLoCoTrainer(
            rank, store_port, device, runner, model_state_dict, n_fragments, diloco_args
        )
        stack.callback(trainer.manager.shutdown)
        return trainer.train_loop()
    return {}


def assert_equal_global_state(
    n_fragments: int,
    rep0: dict[str, dict[str, dict[str, dict[str, object]]]],
    rep1: dict[str, dict[str, dict[str, dict[str, object]]]],
) -> None:
    """
    Asserts that the global state of the two replicas are equal
    """
    for step in rep0.keys():
        for i in range(n_fragments):
            torch.testing.assert_close(
                rep1[step]["user"][f"StreamingDiLoCoFragment_{i}"],
                rep0[step]["user"][f"StreamingDiLoCoFragment_{i}"],
                check_device=False,
                msg=f"{step=} {i=}",
            )
        # Check all outer optimizers
        for i in range(
            len(
                cast(
                    dict[str, dict[str, torch.Tensor]],
                    rep0[step]["user"]["default"]["outer_optim"],
                ).keys()
            )
        ):
            torch.testing.assert_close(
                cast(
                    dict[str, dict[str, torch.Tensor]],
                    rep1[step]["user"]["default"]["outer_optim"],
                )[f"{i}"],
                cast(
                    dict[str, dict[str, torch.Tensor]],
                    rep0[step]["user"]["default"]["outer_optim"],
                )[f"{i}"],
                check_device=False,
            )


class LocalSGDIntegTest(TestCase):
    # TODO: race condition due to using NCCL in threads causes manager allreduce to sometimes not be correct
    # Because of that the test is disabled for cuda
    @parameterized.expand(
        [
            # (True,),
            (False,),
        ]
    )
    def test_local_sgd_recovery(self, use_cuda: bool) -> None:
        # Skip the test if use_cuda is True and there are not enough GPUs
        if use_cuda and torch.cuda.device_count() < 2:
            self.skipTest("Not enough GPUs for CUDA test")

        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 2
        futures = []

        event_injectors = [
            EventInjector(),
            EventInjector().fail_at(0, 2),
        ]

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id, event_injector in zip(range(num_replicas), event_injectors):
                runner = Runner(
                    replica_id=replica_id,
                    num_replicas=num_replicas,
                    lighthouse_address=lighthouse.address(),
                    event_injector=event_injector,
                    train_loop=local_sgd_train_loop,
                    use_cuda=use_cuda,
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
                state_dict[0]["model"], state_dicts[0][0]["model"], check_device=False
            )

        self.assertEqual(event_injectors[1].count[EventInjectorEvent.Failure], 1)

    @parameterized.expand(
        [
            # (True,),
            (False,),
        ]
    )
    def test_diloco_healthy(self, use_cuda: bool) -> None:
        # Skip the test if use_cuda is True and there are not enough GPUs
        if use_cuda and torch.cuda.device_count() < 2:
            self.skipTest("Not enough GPUs for CUDA test")

        lighthouse = LighthouseServer(bind="[::]:0", min_replicas=2)
        num_replicas = 2
        futures = []

        torch.manual_seed(42)
        # Initialize the model so we can pass in the state_dict
        m: nn.Module = MultiMyModel(2, 3, 1)

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id in range(num_replicas):
                event_injector = EventInjector()
                runner = Runner(
                    replica_id=replica_id,
                    num_replicas=num_replicas,
                    lighthouse_address=lighthouse.address(),
                    event_injector=event_injector,
                    train_loop=diloco_train_loop,
                    use_cuda=use_cuda,
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
                    print(e, flush=True)
                    traceback.print_exc()
                    raise

        lighthouse.shutdown()

        rep0, rep1 = state_dicts
        assert_equal_global_state(1, rep1, rep0)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @skipIf(sys.platform == "darwin", "not reliable on mac")
    @parameterized.expand(
        [
            # (True,),
            (False,),
        ]
    )
    def test_diloco_recovery(self, use_cuda: bool) -> None:
        # Skip the test if use_cuda is True and there are not enough GPUs
        if use_cuda and torch.cuda.device_count() < 2:
            self.skipTest("Not enough GPUs for CUDA test")

        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 2
        futures = []

        event_injectors = [
            EventInjector(),
            EventInjector().fail_at(0, 2),
        ]

        torch.manual_seed(42)
        # Initialize the model so we can pass in the state_dict
        m: nn.Module = MultiMyModel(2, 3, 1)

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id, event_injector in zip(range(num_replicas), event_injectors):
                runner = Runner(
                    replica_id=replica_id,
                    num_replicas=num_replicas,
                    lighthouse_address=lighthouse.address(),
                    event_injector=event_injector,
                    train_loop=diloco_train_loop,
                    train_loop_args={
                        "model_state_dict": m.state_dict(),
                    },
                )
                futures.append(executor.submit(runner.run_replica))

            state_dicts = []

            for fut in as_completed(futures):
                continue

            for fut in futures:
                try:
                    state_dicts.append(fut.result()[0])
                except Exception as e:
                    print(e)
                    raise

        lighthouse.shutdown()

        rep0, rep1 = state_dicts

        # Inner optimizer and local model parameters will be different e.g.
        # with 2 replicas r1 and r2, we sync every 2 steps
        #
        # - Manager Step 1
        #   - Step 1: r1 and r2 step
        #   - Step 2: r1 and r2 step, sync the model, quorum succeeds
        # - Manager Step 2
        #   - Step 1: r1 steps but r2 fails
        #   - Step 2:
        #     - r1 steps, sync fails because r2 is down
        #     - r1 recovers r2 from the model state at this step
        #       that is different from the model for r1 at the beginning
        #       of step Manager Step 2
        #
        # Outer optimizer and global model should be the same
        assert_equal_global_state(1, rep1, rep0)

        self.assertEqual(event_injectors[1].count[EventInjectorEvent.Failure], 1)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @skipIf(sys.platform == "darwin", "not reliable on mac")
    @parameterized.expand(
        [
            # (True,),
            (False,),
        ]
    )
    def test_streaming_diloco_recovery(self, use_cuda: bool) -> None:
        # Skip the test if use_cuda is True and there are not enough GPUs
        if use_cuda and torch.cuda.device_count() < 2:
            self.skipTest("Not enough GPUs for CUDA test")

        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 2
        futures = []

        event_injectors = [
            EventInjector(),
            EventInjector().fail_at(0, 2),
        ]

        torch.manual_seed(42)
        # Initialize the model so we can pass in the state_dict
        m: nn.Module = MultiMyModel(2, 3, 2)

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id, event_injector in zip(range(num_replicas), event_injectors):
                runner = Runner(
                    replica_id=replica_id,
                    num_replicas=num_replicas,
                    lighthouse_address=lighthouse.address(),
                    event_injector=event_injector,
                    train_loop=diloco_train_loop,
                    train_loop_args={
                        "model_state_dict": m.state_dict(),
                        "n_fragments": 2,
                        "diloco_args": {
                            "fragment_sync_delay": 1,
                            "sync_every": 4,
                        },
                    },
                )
                futures.append(executor.submit(runner.run_replica))

            state_dicts = []

            for fut in as_completed(futures):
                continue

            for fut in futures:
                try:
                    state_dicts.append(fut.result()[0])
                except Exception as e:
                    print(e)
                    raise

        lighthouse.shutdown()

        rep0, rep1 = state_dicts

        assert_equal_global_state(2, rep1, rep0)

        self.assertEqual(event_injectors[1].count[EventInjectorEvent.Failure], 1)

    CONFIG: list[tuple[bool, int, int, float]] = [
        (use_cuda, n_fragments, fragment_sync_delay, alpha)
        for use_cuda in [False]
        for n_fragments in [1, 2]
        for fragment_sync_delay in [0, 1]
        for alpha in [0.0, 0.5, 1.0]
    ]

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @skipIf(sys.platform == "darwin", "not reliable on mac")
    @parameterized.expand(CONFIG)
    def test_streaming_diloco_upscale(
        self, use_cuda: bool, n_fragments: int, fragment_sync_delay: int, alpha: float
    ) -> None:
        # Skip the test if use_cuda is True and there are not enough GPUs
        if use_cuda and torch.cuda.device_count() < 2:
            self.skipTest("Not enough GPUs for CUDA test")

        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 3
        futures = []
        executors = []

        barrier = threading.Barrier(num_replicas)

        event_injectors = [
            # Make this replica join after other replicas have made 2 steps
            EventInjector().barrier_at(0, 0, barrier),
            EventInjector().barrier_at(0, 2, barrier),
            EventInjector().barrier_at(0, 2, barrier),
        ]

        torch.manual_seed(42)
        # Initialize the model so we can pass in the state_dict
        m: nn.Module = MultiMyModel(2, 3, n_fragments)

        for replica_id, event_injector in zip(range(num_replicas), event_injectors):
            executor = ThreadPoolExecutor(max_workers=1)
            executors.append(executor)
            runner = Runner(
                replica_id=replica_id,
                num_replicas=num_replicas,
                lighthouse_address=lighthouse.address(),
                event_injector=event_injector,
                train_loop=diloco_train_loop,
                train_loop_args={
                    "model_state_dict": m.state_dict(),
                    "n_fragments": n_fragments,
                    "diloco_args": {
                        "fragment_sync_delay": fragment_sync_delay,
                        "sync_every": 4,
                        "fragment_update_alpha": alpha,
                    },
                },
            )
            futures.append(executor.submit(runner.run_replica))

        state_dicts = []

        for fut in as_completed(futures):
            continue

        for fut in futures:
            try:
                state_dicts.append(fut.result()[0])
            except Exception as e:
                print(e)
                raise

        lighthouse.shutdown()

        rep0, rep1, rep2 = state_dicts

        assert_equal_global_state(n_fragments, rep0, rep1)
        assert_equal_global_state(n_fragments, rep0, rep2)

        for event_injector in event_injectors:
            self.assertEqual(event_injectors[1].count[EventInjectorEvent.Barrier], 1)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @skipIf(sys.platform == "darwin", "not reliable on mac")
    @parameterized.expand(CONFIG)
    def test_streaming_diloco_commit_failure(
        self, use_cuda: bool, n_fragments: int, fragment_sync_delay: int, alpha: float
    ) -> None:
        # Skip the test if use_cuda is True and there are not enough GPUs
        if use_cuda and torch.cuda.device_count() < 2:
            self.skipTest("Not enough GPUs for CUDA test")

        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=2,
        )
        num_replicas = 2
        futures = []
        executors = []

        event_injectors = [
            EventInjector().fail_allreduce_at(0, 1),
            EventInjector().fail_allreduce_at(0, 1),
        ]

        torch.manual_seed(42)
        # Initialize the model so we can pass in the state_dict
        m: nn.Module = MultiMyModel(2, 3, n_fragments)

        for replica_id, event_injector in zip(range(num_replicas), event_injectors):
            executor = ThreadPoolExecutor(max_workers=1)
            executors.append(executor)
            runner = Runner(
                replica_id=replica_id,
                num_replicas=num_replicas,
                lighthouse_address=lighthouse.address(),
                event_injector=event_injector,
                train_loop=diloco_train_loop,
                train_loop_args={
                    "model_state_dict": m.state_dict(),
                    "n_fragments": n_fragments,
                    "diloco_args": {
                        "fragment_sync_delay": fragment_sync_delay,
                        "sync_every": 4,
                        "fragment_update_alpha": alpha,
                    },
                },
            )
            futures.append(executor.submit(runner.run_replica))

        state_dicts = []

        for fut in as_completed(futures):
            continue

        for fut in futures:
            try:
                state_dicts.append(fut.result()[0])
            except Exception as e:
                print(e)
                raise

        lighthouse.shutdown()

        rep0, rep1 = state_dicts

        assert_equal_global_state(n_fragments, rep0, rep1)

        for event_injector in event_injectors:
            self.assertEqual(
                event_injector.count[EventInjectorEvent.AllreduceFailure], 1
            )


if __name__ == "__main__":
    import unittest

    unittest.main()
