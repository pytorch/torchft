# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import concurrent
import multiprocessing
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional
from unittest import TestCase
from unittest.mock import MagicMock, create_autospec, patch

import torch
import torch.distributed as dist
from torch.distributed import TCPStore

from torchft._torchft import (
    FailureStream,
    LighthouseClient,
    LighthouseServer,
    QuorumResult,
)
from torchft.checkpointing.transport import CheckpointTransport
from torchft.manager import (
    MANAGER_ADDR_KEY,
    REPLICA_ID_KEY,
    ExceptionWithTraceback,
    Manager,
    WorldSizeMode,
    _failure_listener_process_main,
)
from torchft.process_group import ProcessGroup, ProcessGroupGloo, _DummyWork


def mock_should_commit(
    rank: int, step: int, should_commit: bool, timeout: timedelta
) -> bool:
    return should_commit


class TestManager(TestCase):
    store: TCPStore  # pyre-fixme[13]: never initialized
    load_state_dict: MagicMock  # pyre-fixme[13]: never initialized
    manager: Optional[Manager]  # pyre-fixme[13]: never initialized

    def tearDown(self) -> None:
        # Manager cleanup might be handled by _create_manager
        if hasattr(self, "manager") and self.manager is not None:
            self.manager.shutdown(wait=False)

    def _create_manager(
        self,
        use_async_quorum: bool = True,
        min_replica_size: int = 2,
        world_size_mode: WorldSizeMode = WorldSizeMode.DYNAMIC,
        timeout: timedelta = timedelta(seconds=10),
        init_sync: bool = True,
        max_retries: Optional[int] = None,
        proactive_recovery: bool = False,
    ) -> Manager:
        pg = create_autospec(ProcessGroup)
        pg.errored.return_value = None

        self.store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        self.store.set(MANAGER_ADDR_KEY, "dummy")
        self.store.set(REPLICA_ID_KEY, "dummy_id")
        with patch(
            "os.environ",
            {
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": self.store.port,
                "RANK": "1",
                "WORLD_SIZE": "2",
            },
        ):
            self.load_state_dict = MagicMock()
            manager = Manager(
                pg=pg,
                min_replica_size=min_replica_size,
                load_state_dict=self.load_state_dict,
                state_dict=lambda: {},
                use_async_quorum=use_async_quorum,
                world_size_mode=world_size_mode,
                timeout=timeout,
                init_sync=init_sync,
                max_retries=max_retries,
                proactive_recovery=proactive_recovery,
            )
            self.manager = manager
        return manager

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_manager(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        self.assertEqual(client_mock.call_count, 1)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_state_dict(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()

        state_dict = manager.state_dict()
        self.assertEqual(
            state_dict,
            {
                "step": 0,
                "batches_committed": 0,
            },
        )

        manager.load_state_dict(
            {
                "step": 1234,
                "batches_committed": 2345,
            }
        )
        self.assertEqual(manager.current_step(), 1234)
        self.assertEqual(manager.batches_committed(), 2345)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_user_state_dict(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()

        self.assertEqual(
            manager._manager_state_dict(),
            {
                "user": {},
                "torchft": {
                    "step": 0,
                    "batches_committed": 0,
                },
            },
        )

        manager.set_state_dict_fns(
            self.load_state_dict,
            lambda: {"new_state": 1},
        )

        self.assertEqual(
            manager._manager_state_dict(),
            {
                "user": {"new_state": 1},
                "torchft": {
                    "step": 0,
                    "batches_committed": 0,
                },
            },
        )

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_happy(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        client_mock().should_commit = mock_should_commit

        quorum = QuorumResult()
        quorum.quorum_id = 123
        quorum.replica_rank = 1
        quorum.replica_world_size = 2
        quorum.recover_src_manager_address = "manager address"
        quorum.store_address = f"localhost:{self.store.port}"
        quorum.max_step = 1
        quorum.max_replica_rank = 1
        quorum.max_world_size = 2
        quorum.heal = False

        client_mock()._quorum.return_value = quorum

        self.assertEqual(manager._quorum_id, -1)
        self.assertEqual(manager.current_step(), 0)
        self.assertEqual(manager.batches_committed(), 0)

        manager.start_quorum()
        manager.allreduce(torch.tensor([1.0])).wait()
        self.assertEqual(len(manager._pending_work), 1)
        self.assertTrue(manager.should_commit())
        self.assertEqual(len(manager._pending_work), 0)

        self.assertEqual(manager._quorum_id, 123)
        self.assertEqual(manager.current_step(), 1)
        # pyre-ignore[16]: _pg is mocked
        self.assertEqual(manager._pg.allreduce.call_count, 1)

        manager.start_quorum()
        self.assertEqual(manager.batches_committed(), 2)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_heal_sync(self, client_mock: MagicMock) -> None:
        manager = self._create_manager(use_async_quorum=False)
        client_mock().should_commit = mock_should_commit

        quorum = QuorumResult()
        quorum.quorum_id = 123
        quorum.replica_rank = 1
        quorum.replica_world_size = 2
        quorum.recover_src_manager_address = "manager address"
        quorum.recover_src_replica_rank = 0
        quorum.store_address = f"localhost:{self.store.port}"
        quorum.max_step = 20
        quorum.max_replica_rank = None
        quorum.max_world_size = 2
        quorum.heal = True

        client_mock()._quorum.return_value = quorum

        # forcible increment checkpoint server to compute correct address
        manager._checkpoint_transport.send_checkpoint(
            dst_ranks=[],
            step=quorum.max_step,
            state_dict=manager._manager_state_dict(),
            timeout=timedelta(seconds=10),
        )
        client_mock()._checkpoint_metadata.return_value = (
            manager._checkpoint_transport.metadata()
        )

        self.assertEqual(manager._quorum_id, -1)
        self.assertEqual(manager.current_step(), 0)

        self.assertEqual(manager.num_participants(), 0)
        self.assertEqual(manager.participating_rank(), None)

        manager.start_quorum()
        manager.allreduce(torch.tensor([1.0])).wait()
        self.assertFalse(manager._healing)
        self.assertTrue(manager.is_participating())
        self.assertEqual(manager.num_participants(), 2)
        self.assertTrue(manager.should_commit())

        self.assertEqual(manager._quorum_id, 123)
        self.assertEqual(manager.current_step(), 21)
        # pyre-ignore[16]: _pg is mocked
        self.assertEqual(manager._pg.allreduce.call_count, 1)
        # pyre-ignore[16]: _pg is mocked
        self.assertEqual(manager._pg.allreduce.return_value.get_future.call_count, 1)

        self.assertEqual(self.load_state_dict.call_count, 1)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_heal_async_not_enough_participants(
        self, client_mock: MagicMock
    ) -> None:
        manager = self._create_manager(use_async_quorum=True, min_replica_size=2)
        client_mock().should_commit = mock_should_commit

        quorum = QuorumResult()
        quorum.quorum_id = 123
        quorum.replica_rank = 1
        quorum.replica_world_size = 2
        quorum.recover_src_manager_address = "manager address"
        quorum.recover_src_replica_rank = 0
        quorum.store_address = f"localhost:{self.store.port}"
        quorum.max_step = 20
        quorum.max_replica_rank = None
        quorum.max_world_size = 1
        quorum.heal = True

        client_mock()._quorum.return_value = quorum

        # forcible increment checkpoint server to compute correct address
        manager._checkpoint_transport.send_checkpoint(
            dst_ranks=[],
            step=quorum.max_step,
            state_dict=manager._manager_state_dict(),
            timeout=timedelta(seconds=10),
        )
        client_mock()._checkpoint_metadata.return_value = (
            manager._checkpoint_transport.metadata()
        )

        self.assertEqual(manager._quorum_id, -1)
        self.assertEqual(manager.current_step(), 0)

        manager.start_quorum()
        assert manager._quorum_future is not None
        manager._quorum_future.result()
        self.assertTrue(manager._healing)
        self.assertFalse(manager.is_participating())
        self.assertEqual(manager.num_participants(), 1)

        grad = torch.tensor([1.0])
        manager.allreduce(grad).wait()
        torch.testing.assert_close(grad, torch.zeros_like(grad))
        # don't commit since num_max < min_replica_size
        self.assertFalse(manager.should_commit())
        self.assertEqual(manager.current_step(), 20)

        self.assertEqual(manager._quorum_id, 123)
        self.assertEqual(manager.current_step(), 20)
        # pyre-ignore[16]: _pg is mocked
        self.assertEqual(manager._pg.allreduce.call_count, 1)
        # pyre-ignore[16]: _pg is mocked
        self.assertEqual(manager._pg.allreduce.return_value.get_future.call_count, 1)

        self.assertEqual(self.load_state_dict.call_count, 1)

        # failed to commit so no step
        quorum.heal = False
        manager.start_quorum()
        self.assertEqual(manager.current_step(), 20)
        self.assertEqual(manager.batches_committed(), 0)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_heal_async_zero_grad(self, client_mock: MagicMock) -> None:
        manager = self._create_manager(use_async_quorum=True, min_replica_size=1)
        client_mock().should_commit = mock_should_commit

        quorum = QuorumResult()
        quorum.quorum_id = 123
        quorum.replica_rank = 1
        quorum.replica_world_size = 2
        quorum.recover_src_manager_address = "manager address"
        quorum.recover_src_replica_rank = 0
        quorum.store_address = f"localhost:{self.store.port}"
        quorum.max_step = 20
        quorum.max_replica_rank = None
        quorum.max_world_size = 1
        quorum.heal = True

        client_mock()._quorum.return_value = quorum

        # forceable increment checkpoint server to compute correct address
        manager._checkpoint_transport.send_checkpoint(
            dst_ranks=[],
            step=quorum.max_step,
            state_dict=manager._manager_state_dict(),
            timeout=timedelta(seconds=10),
        )
        client_mock()._checkpoint_metadata.return_value = (
            manager._checkpoint_transport.metadata()
        )

        self.assertEqual(manager._quorum_id, -1)
        self.assertEqual(manager.current_step(), 0)

        manager.start_quorum()
        assert manager._quorum_future is not None
        manager._quorum_future.result()
        self.assertTrue(manager._healing)

        grad = torch.tensor([1.0])
        manager.allreduce(grad).wait()
        torch.testing.assert_close(grad, torch.zeros_like(grad))
        # don't commit since num_max < min_replica_size
        self.assertTrue(manager.should_commit())
        self.assertEqual(manager.num_participants(), 1)
        self.assertTrue(manager.current_step(), 21)

        self.assertEqual(manager._quorum_id, 123)
        # pyre-ignore[16]: _pg is mocked
        self.assertEqual(manager._pg.allreduce.call_count, 1)
        # pyre-ignore[16]: _pg is mocked
        self.assertEqual(manager._pg.allreduce.return_value.get_future.call_count, 1)

        self.assertEqual(self.load_state_dict.call_count, 1)

        # healed
        quorum.heal = False
        manager.start_quorum()
        self.assertEqual(manager.current_step(), 21)
        self.assertEqual(manager.batches_committed(), 1)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_allreduce_error(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        client_mock().should_commit = mock_should_commit

        quorum = QuorumResult()
        quorum.quorum_id = 123
        quorum.replica_rank = 1
        quorum.replica_world_size = 2
        quorum.recover_src_manager_address = "manager address"
        quorum.store_address = f"localhost:{self.store.port}"
        quorum.max_step = 1
        quorum.max_replica_rank = 1
        quorum.max_world_size = 2
        quorum.heal = False

        client_mock()._quorum.return_value = quorum

        self.assertEqual(manager._quorum_id, -1)
        self.assertEqual(manager.current_step(), 0)

        manager.start_quorum()
        manager.allreduce(torch.tensor([1.0])).wait()
        # pyre-ignore[16]: _pg is mocked
        self.assertEqual(manager._pg.allreduce.call_count, 1)

        # inject failure when work queued
        # pyre-ignore[16]: _pg is mocked
        manager._pg.allreduce.side_effect = RuntimeError("injected failure")
        manager.allreduce(torch.tensor([1.0])).wait()
        self.assertTrue(manager._errored)
        # this should be skipped due to error
        manager.allreduce(torch.tensor([1.0])).wait()
        self.assertEqual(manager._pg.allreduce.call_count, 2)
        # pyre-ignore[16]: _pg is mocked
        self.assertEqual(manager._pg.allreduce.return_value.get_future.call_count, 1)

        self.assertFalse(manager.should_commit())
        self.assertTrue(manager._errored)

        # cleanup
        manager._pg.allreduce.side_effect = None

        # inject failure when worked waited
        quorum.max_step = 2

        manager.start_quorum()

        self.assertFalse(manager._errored)

        bad_fut = torch.futures.Future()  # pyre-fixme[29]: not a function
        bad_fut.set_exception(RuntimeError("injected failure"))
        manager._pg.allreduce.return_value.get_future.return_value = bad_fut
        manager.allreduce(torch.tensor([1.0])).wait()
        self.assertEqual(manager._pg.allreduce.return_value.get_future.call_count, 2)
        self.assertTrue(manager._errored)
        self.assertFalse(manager.should_commit())
        self.assertTrue(manager._errored)

        # cleanup
        manager._pg.allreduce.reset_mock(return_value=True)

        # recover on next step
        quorum.max_step = 3

        manager.start_quorum()
        manager.allreduce(torch.tensor([1.0])).wait()
        self.assertTrue(manager.should_commit())

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_pg_errored(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()
        client_mock().should_commit = mock_should_commit

        quorum = QuorumResult()
        quorum.quorum_id = 123
        quorum.replica_rank = 1
        quorum.replica_world_size = 2
        quorum.recover_src_manager_address = "manager address"
        quorum.store_address = f"localhost:{self.store.port}"
        quorum.max_step = 1
        quorum.max_replica_rank = 1
        quorum.max_world_size = 2
        quorum.heal = False

        client_mock()._quorum.return_value = quorum

        self.assertEqual(manager._quorum_id, -1)
        self.assertEqual(manager.current_step(), 0)

        manager.start_quorum()

        injected_failure = RuntimeError("injected failure")

        # pyre-ignore[16]: _pg is mocked
        manager._pg.errored.return_value = injected_failure

        self.assertFalse(manager.should_commit())
        assert manager._errored is not None
        self.assertEqual(manager._errored.original_exception, injected_failure)
        # pyre-ignore[16]: _pg is mocked
        self.assertEqual(manager._pg.errored.call_count, 1)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_fixed_world_size(self, client_mock: MagicMock) -> None:
        # test active and spares
        for rank in [1, 2]:
            manager = self._create_manager(
                min_replica_size=2,
                world_size_mode=WorldSizeMode.FIXED_WITH_SPARES,
            )
            client_mock().should_commit = mock_should_commit

            quorum = QuorumResult()
            quorum.quorum_id = 123
            quorum.replica_rank = rank
            quorum.replica_world_size = 3
            quorum.recover_src_manager_address = "manager address"
            quorum.store_address = f"localhost:{self.store.port}"
            quorum.max_step = 1
            quorum.max_replica_rank = rank
            quorum.max_world_size = 3
            quorum.heal = False

            client_mock()._quorum.return_value = quorum

            self.assertEqual(manager._quorum_id, -1)
            self.assertEqual(manager.current_step(), 0)
            self.assertEqual(manager.batches_committed(), 0)

            manager.start_quorum()
            manager.allreduce(torch.tensor([1.0])).wait()

            self.assertEqual(manager.is_participating(), rank != 2)
            self.assertEqual(manager.num_participants(), 2)

            self.assertTrue(manager.should_commit())
            self.assertEqual(manager.batches_committed(), 2)
            self.assertEqual(manager.current_step(), 1)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_no_healing(self, client_mock: MagicMock) -> None:
        manager = self._create_manager(
            min_replica_size=2,
        )
        client_mock().should_commit = mock_should_commit

        quorum = QuorumResult()
        quorum.quorum_id = 123
        quorum.replica_rank = 0
        quorum.replica_world_size = 3
        quorum.recover_src_manager_address = "manager address"
        quorum.recover_src_replica_rank = 1
        quorum.store_address = f"localhost:{self.store.port}"
        quorum.max_step = 1
        quorum.max_replica_rank = None
        quorum.max_world_size = 2
        quorum.heal = True
        client_mock()._quorum.return_value = quorum

        self.assertEqual(manager._quorum_id, -1)
        self.assertEqual(manager.current_step(), 0)
        self.assertEqual(manager.batches_committed(), 0)

        manager.start_quorum(allow_heal=False)
        manager.allreduce(torch.tensor([1.0])).wait()

        self.assertFalse(manager.is_participating())
        self.assertEqual(manager.num_participants(), 2)

        self.assertTrue(manager.should_commit())
        self.assertEqual(manager.batches_committed(), 2)
        self.assertEqual(manager.current_step(), 1)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_manager_report_error(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()

        self.assertIsNone(manager.errored())
        e = RuntimeError("some error")
        manager.report_error(e)
        error = manager.errored()
        assert error is not None
        self.assertIs(error.original_exception, e)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_manager_wrap_future(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()

        self.assertIsNone(manager.errored())

        fut = torch.futures.Future()  # pyre-fixme[29]: not a function
        wrapped_fut = manager.wrap_future(fut, 2)
        self.assertIsNone(manager.errored())

        e = RuntimeError("injected failure")
        fut.set_exception(e)
        error = manager.errored()
        assert error is not None
        self.assertIs(error.original_exception, e)
        self.assertEqual(wrapped_fut.value(), 2)

        self.assertEqual(manager._pending_work, [wrapped_fut])

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_manager_wrap_future_timeout(self, client_mock: MagicMock) -> None:
        manager = self._create_manager(timeout=timedelta(seconds=0.01))

        self.assertFalse(manager.errored())

        fut = torch.futures.Future()  # pyre-fixme[29]: not a function
        wrapped_fut = manager.wrap_future(fut, 2)
        wrapped_fut.wait()
        error = manager.errored()
        assert error is not None
        with self.assertRaisesRegex(
            TimeoutError, "future did not complete within.*0.01"
        ):
            raise error.original_exception

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_manager_numerics(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()

        manager._quorum_future = quorum_future = MagicMock(
            spec=concurrent.futures.Future
        )
        manager._participating_replica_rank = 1
        manager._participating_replica_world_size = 5
        self.assertEqual(manager.num_participants(), 5)
        self.assertEqual(quorum_future.result.call_count, 1)
        self.assertEqual(manager.participating_rank(), 1)
        self.assertEqual(quorum_future.result.call_count, 2)

        # pyre-ignore[16]: _pg is mocked
        manager._pg.allreduce.return_value = _DummyWork(None)

        self.assertTrue(manager.is_participating())
        fut = torch.futures.Future()  # pyre-fixme[29]: not a function
        fut = manager.allreduce(torch.tensor([1.0]))
        result = fut.value()
        torch.testing.assert_close(result, torch.tensor([1.0 / 5]))

        # check healing numerics
        manager._healing = True
        self.assertFalse(manager.is_participating())
        fut = torch.futures.Future()  # pyre-fixme[29]: not a function
        fut = manager.allreduce(torch.tensor([1.0]))
        result = fut.value()
        torch.testing.assert_close(result, torch.tensor([0.0]))

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_happy_timeouts(self, client_mock: MagicMock) -> None:
        manager = self._create_manager(use_async_quorum=False)

        quorum = QuorumResult()
        quorum.quorum_id = 123
        quorum.replica_rank = 1
        quorum.replica_world_size = 2
        quorum.recover_src_manager_address = "manager address"
        quorum.store_address = f"localhost:{self.store.port}"
        quorum.max_step = 1
        quorum.max_replica_rank = 1
        quorum.max_world_size = 2
        quorum.heal = False

        client_mock()._quorum.return_value = quorum

        manager.start_quorum(timeout=timedelta(seconds=12))
        self.assertEqual(
            client_mock()._quorum.call_args.kwargs["timeout"], timedelta(seconds=12)
        )

        self.assertTrue(manager.should_commit(timeout=timedelta(seconds=23)))
        self.assertEqual(
            client_mock().should_commit.call_args.kwargs["timeout"],
            timedelta(seconds=23),
        )

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_skip_init(self, client_mock: MagicMock) -> None:
        manager = self._create_manager(
            use_async_quorum=False,
            init_sync=False,
        )

        self.assertFalse(manager._init_sync)

        quorum = QuorumResult()
        quorum.quorum_id = 123
        quorum.replica_rank = 1
        quorum.replica_world_size = 2
        quorum.recover_src_manager_address = "manager address"
        quorum.store_address = f"localhost:{self.store.port}"
        quorum.max_step = 1
        quorum.max_replica_rank = 1
        quorum.max_world_size = 2
        quorum.heal = False

        client_mock()._quorum.return_value = quorum

        manager.start_quorum()
        self.assertEqual(client_mock()._quorum.call_args.kwargs["init_sync"], False)

        manager._init_sync = True
        manager.start_quorum()
        self.assertEqual(client_mock()._quorum.call_args.kwargs["init_sync"], True)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_checkpoint_errors(self, client_mock: MagicMock) -> None:
        manager = self._create_manager(use_async_quorum=True)
        client_mock().should_commit = MagicMock(return_value=False)

        transport = MagicMock(spec=CheckpointTransport)
        transport.send_checkpoint.side_effect = RuntimeError("send failure")
        transport.recv_checkpoint.side_effect = RuntimeError("recv failure")
        manager._checkpoint_transport = transport

        quorum = QuorumResult()
        quorum.quorum_id = 123
        quorum.replica_rank = 1
        quorum.replica_world_size = 2
        quorum.recover_src_manager_address = "manager address"
        quorum.recover_src_replica_rank = 0
        quorum.store_address = f"localhost:{self.store.port}"
        quorum.max_step = 20
        quorum.max_replica_rank = None
        quorum.max_world_size = 2
        quorum.heal = True

        client_mock()._quorum.return_value = quorum

        manager.start_quorum()
        manager.wait_quorum()
        self.assertFalse(manager.should_commit())

        error = manager.errored()
        assert error is not None
        with self.assertRaisesRegex(RuntimeError, "recv failure"):
            raise error.original_exception

        quorum.recover_dst_replica_ranks = [0]
        manager.start_quorum()
        manager.wait_quorum()
        self.assertFalse(manager.should_commit())

        error = manager.errored()
        assert error is not None
        with self.assertRaisesRegex(RuntimeError, "send failure"):
            raise error.original_exception

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_quorum_configure_errors(self, client_mock: MagicMock) -> None:
        manager = self._create_manager(use_async_quorum=True)
        client_mock().should_commit = MagicMock(return_value=False)

        # pyre-ignore[16]: mock
        manager._pg.configure.side_effect = RuntimeError("configure failure")

        quorum = QuorumResult()
        quorum.quorum_id = 123
        quorum.replica_rank = 1
        quorum.replica_world_size = 2
        quorum.recover_src_manager_address = "manager address"
        quorum.recover_src_replica_rank = 0
        quorum.store_address = f"localhost:{self.store.port}"
        quorum.max_step = 20
        quorum.max_replica_rank = None
        quorum.max_world_size = 2

        client_mock()._quorum.return_value = quorum

        manager.start_quorum()
        manager.wait_quorum()
        self.assertFalse(manager.should_commit())

        error = manager.errored()
        assert error is not None
        with self.assertRaisesRegex(RuntimeError, "configure failure"):
            raise error.original_exception

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_max_retries(self, client_mock: MagicMock) -> None:
        # Create a manager with max_retries=2
        manager = self._create_manager(max_retries=2)

        # Setup quorum for testing
        quorum = QuorumResult()
        quorum.quorum_id = 123
        quorum.replica_rank = 1
        quorum.replica_world_size = 2
        quorum.recover_src_manager_address = "manager address"
        quorum.store_address = f"localhost:{self.store.port}"
        quorum.max_step = 1
        quorum.max_replica_rank = 1
        quorum.max_world_size = 2
        quorum.heal = False
        client_mock()._quorum.return_value = quorum

        # Make should_commit always return False to simulate failures
        client_mock().should_commit = MagicMock(return_value=False)

        # Start quorum
        manager.start_quorum()

        # First failure
        self.assertFalse(manager.should_commit())
        self.assertEqual(manager._commit_failures, 1)

        # Second failure
        self.assertFalse(manager.should_commit())
        self.assertEqual(manager._commit_failures, 2)

        # Third failure - should raise exception
        with self.assertRaises(RuntimeError) as context:
            manager.should_commit()

        self.assertIn("exceeding max_retries=2", str(context.exception))
        self.assertEqual(manager._commit_failures, 3)

        # Now test that success resets the counter
        manager._commit_failures = 2  # Reset to just before failure threshold
        client_mock().should_commit = MagicMock(return_value=True)  # Now succeed

        # This should succeed and reset the counter
        self.assertTrue(manager.should_commit())
        self.assertEqual(manager._commit_failures, 0)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_manager_error_handler(self, client_mock: MagicMock) -> None:
        """Test that the Manager correctly processes exceptions sent from the failure_listener_process."""
        # Create a manager
        manager = self._create_manager()

        # Create an exception simulating what would be sent from _failure_listener_process_main
        error = Exception("Peer failure detected: replica failed_replica has failed")
        exception = ExceptionWithTraceback(error)

        # Directly test the error handling mechanism
        manager._error_handler(error)

        # Verify the error was properly processed
        captured_error = manager.errored()
        self.assertIsNotNone(captured_error)
        self.assertEqual(str(captured_error.original_exception), str(error))

    def test_direct_error_pipe(self) -> None:
        """Test sending an exception to the Manager's _error_pipe."""
        # Create a manager with proactive_recovery=True to ensure it has an error pipe
        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=1,
            join_timeout_ms=100,
        )

        # Create a manager that tries to join
        store = dist.TCPStore(
            host_name="localhost",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )
        pg = ProcessGroupGloo()
        manager = Manager(
            pg=pg,
            min_replica_size=1,
            load_state_dict=lambda x: None,
            state_dict=lambda: None,
            replica_id=f"lighthouse_test",
            store_addr="localhost",
            store_port=store.port,
            rank=0,
            world_size=1,
            use_async_quorum=False,
            lighthouse_addr=lighthouse.address(),
            proactive_recovery=True,
        )

        # Make sure the error pipe is created
        self.assertIsNotNone(manager._error_pipe, "Manager should have an error pipe")
        time.sleep(1)
        # Create a mock error message
        mock_error_msg = "Test failure detected from direct pipe test"
        test_exception = Exception(mock_error_msg)

        # Create an ExceptionWithTraceback and send it through the pipe
        exc_with_tb = ExceptionWithTraceback(test_exception)
        manager._error_remote.send(exc_with_tb)

        # Wait a short time for the error processor thread to process the message
        time.sleep(1)

        # Verify that the error was properly processed by the Manager
        error_obj = manager.errored()
        self.assertIsNotNone(
            error_obj, "Error should have been captured by the Manager"
        )

        # Clean up
        manager.shutdown(wait=True)

    def test_manager_failure_e2e(self) -> None:
        """Test that the Manager correctly handles errors from the failure_listener_process."""
        # Create a manager with proactive_recovery=True to ensure it has an error pipe
        lighthouse = LighthouseServer(
            bind="[::]:0",
            min_replicas=1,
            join_timeout_ms=100,
        )

        # Create a manager that tries to join
        store = dist.TCPStore(
            host_name="localhost",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )
        pg = ProcessGroupGloo()
        manager = Manager(
            pg=pg,
            min_replica_size=1,
            load_state_dict=lambda x: None,
            state_dict=lambda: None,
            replica_id=f"lighthouse_test",
            store_addr="localhost",
            store_port=store.port,
            rank=0,
            world_size=1,
            use_async_quorum=False,
            lighthouse_addr=lighthouse.address(),
            proactive_recovery=True,
        )

        time.sleep(1.5)

        failed_replica_id = "failed_replica"
        lighthouse.inject_failure(failed_replica_id)

        time.sleep(1.5)  # Prevent flakyness
        error_obj = manager.errored()

        # Verify that the manager received the error notification
        self.assertIsNotNone(error_obj, "Manager should have captured the failure")
        self.assertIn(
            failed_replica_id,
            str(error_obj.original_exception),
            f"Error should mention the failed replica: {error_obj.original_exception}",
        )

        # Clean up resources
        manager.shutdown(wait=True)
