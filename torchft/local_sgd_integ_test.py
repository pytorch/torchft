import copy
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from typing import Any, Dict
from unittest import TestCase

import torch
from parameterized import parameterized
from torch import nn, optim

from torchft.local_sgd import DiLoCo, LocalSGD
from torchft.manager import Manager
from torchft.manager_integ_test import FailureInjector, MyModel, Runner
from torchft.process_group import ProcessGroupGloo, ProcessGroupNCCL
from torchft.torchft import Lighthouse

logger: logging.Logger = logging.getLogger(__name__)


def local_sgd_train_loop(
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
                print(f"stepping {local_sgd._local_step}", flush=True)
                loss.backward()

                optimizer.step()

                if manager.current_step() >= 4:
                    break

                runner.failure_injector.check(rank, manager.current_step())

        # return state_dict so we can check consistency
        return state_dict()
    return {}


def diloco_train_loop(
    rank: int,
    store_port: int,
    device: torch.device,
    runner: Runner,
) -> Dict[str, Dict[str, object]]:
    with ExitStack() as stack:
        # Declare the model and optimizers
        m: nn.Module = MyModel()
        model_state_dict: Dict[str, Any] = runner.train_loop_args["model_state_dict"]
        m.load_state_dict(model_state_dict)
        m = m.to(device)

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
            # diloco._backup_parameters = state_dict["backup_params"]
            # for param in diloco._backup_parameters.values():
            #     param = param.to(device)
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
            manager,
            m,
            inner_optimizer,
            outer_optimizer,
            backup_device=device,
            sync_every=2,
        ) as diloco:
            print("starting training", flush=True)
            while True:
                inputs = torch.rand(2, 3).to(device)
                labels = torch.randint(4, (2,)).to(device)

                out = m(inputs)
                loss = criterion(out, labels)

                inner_optimizer.zero_grad()
                loss.backward()
                print(f"stepping {diloco._local_step}", flush=True)
                inner_optimizer.step()
                manager_step_str = str(manager.current_step())
                all_state_dicts[manager_step_str] = state_dict()

                # after 4 model updates then break
                if manager.current_step() >= 4:
                    break

                runner.failure_injector.check(rank, manager.current_step())

        # return state_dict so we can check consistency
        return all_state_dicts
    return {}


class LocalSGDIntegTest(TestCase):
    @parameterized.expand(
        [
            (False,),
            (True,),
        ]
    )
    def test_local_sgd_recovery(self, use_cuda: bool) -> None:
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

        with ThreadPoolExecutor(max_workers=num_replicas) as executor:
            for replica_id, failure_injector in zip(
                range(num_replicas), failure_injectors
            ):
                runner = Runner(
                    replica_id=replica_id,
                    num_replicas=num_replicas,
                    lighthouse_address=lighthouse.address(),
                    failure_injector=failure_injector,
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

        self.assertEqual(failure_injectors[1].count, 1)

    @parameterized.expand(
        [
            (False,),
            (True,),
        ]
    )
    def test_diloco_healthy(self, use_cuda: bool) -> None:
        lighthouse = Lighthouse(bind="[::]:0", min_replicas=2)
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
                    num_replicas=num_replicas,
                    lighthouse_address=lighthouse.address(),
                    failure_injector=failure_injector,
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

        for replica_group in state_dicts:
            for step, state_dict in replica_group.items():
                # inner optimizer will be different, outer optimizer and model should be the same
                torch.testing.assert_close(
                    state_dict["backup_params"],
                    state_dicts[0][str(step)]["backup_params"],
                    check_device=False,
                )
                torch.testing.assert_close(
                    state_dict["outer_optim"],
                    state_dicts[0][str(step)]["outer_optim"],
                    check_device=False,
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
                    num_replicas=num_replicas,
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
