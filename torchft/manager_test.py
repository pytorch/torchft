# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import concurrent
from datetime import timedelta
from typing import Optional
from unittest import TestCase
from unittest.mock import MagicMock, create_autospec, patch

import torch
from torch.distributed import TCPStore

from torchft._torchft import QuorumResult
from torchft.manager import MANAGER_ADDR_KEY, REPLICA_ID_KEY, Manager, WorldSizeMode
from torchft.process_group import ProcessGroup, _DummyWork


def mock_should_commit(
    rank: int, step: int, should_commit: bool, timeout: timedelta
) -> bool:
    return should_commit


class TestManager(TestCase):
    store: TCPStore  # pyre-fixme[13]: never initialized
    load_state_dict: MagicMock  # pyre-fixme[13]: never initialized
    manager: Optional[Manager]  # pyre-fixme[13]: never initialized

    def tearDown(self) -> None:
        manager = self.manager
        if manager is not None:
            manager.shutdown(wait=False)

    def _create_manager(
        self,
        use_async_quorum: bool = True,
        min_replica_size: int = 2,
        world_size_mode: WorldSizeMode = WorldSizeMode.DYNAMIC,
        timeout: timedelta = timedelta(seconds=10),
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
        quorum.max_rank = 1
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
        quorum.recover_src_rank = 0
        quorum.store_address = f"localhost:{self.store.port}"
        quorum.max_step = 20
        quorum.max_rank = None
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
        quorum.recover_src_rank = 0
        quorum.store_address = f"localhost:{self.store.port}"
        quorum.max_step = 20
        quorum.max_rank = None
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
        quorum.recover_src_rank = 0
        quorum.store_address = f"localhost:{self.store.port}"
        quorum.max_step = 20
        quorum.max_rank = None
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
        quorum.max_rank = 1
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
        quorum.max_rank = 1
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
        self.assertEqual(manager._errored, injected_failure)
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
            quorum.max_rank = rank
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
        quorum.recover_src_rank = 1
        quorum.store_address = f"localhost:{self.store.port}"
        quorum.max_step = 1
        quorum.max_rank = None
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
        self.assertIs(manager.errored(), e)

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_manager_wrap_future(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()

        self.assertIsNone(manager.errored())

        fut = torch.futures.Future()  # pyre-fixme[29]: not a function
        wrapped_fut = manager.wrap_future(fut, 2)
        self.assertIsNone(manager.errored())

        e = RuntimeError("injected failure")
        fut.set_exception(e)
        self.assertIs(manager.errored(), e)
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
        self.assertIsNotNone(error)
        with self.assertRaisesRegex(
            TimeoutError, "future did not complete within.*0.01"
        ):
            raise error

    @patch("torchft.manager.ManagerClient", autospec=True)
    def test_manager_numerics(self, client_mock: MagicMock) -> None:
        manager = self._create_manager()

        manager._quorum_future = quorum_future = MagicMock(
            spec=concurrent.futures.Future
        )
        manager._participating_rank = 1
        manager._participating_world_size = 5
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
        quorum.max_rank = 1
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
