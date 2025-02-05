# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
import unittest
import urllib.error
from datetime import timedelta
from unittest import TestCase
from unittest.mock import MagicMock

from torchft.checkpointing import CheckpointServer, _timed_acquire


class TestCheckpointing(TestCase):
    def test_checkpoint_server(self) -> None:
        expected = {"state": "dict"}
        state_dict_fn = MagicMock()
        state_dict_fn.return_value = expected
        server = CheckpointServer(
            timeout=timedelta(seconds=10),
        )

        server.send_checkpoint(
            dst_ranks=[],
            step=1234,
            state_dict=expected,
            timeout=timedelta(seconds=10),
        )

        metadata = server.metadata()

        out = server.recv_checkpoint(
            src_rank=0, metadata=metadata, step=1234, timeout=timedelta(seconds=10)
        )
        self.assertEqual(out, expected)

        # test timeout
        with self.assertRaisesRegex(urllib.error.URLError, r"urlopen error"):
            server.recv_checkpoint(
                src_rank=0, metadata=metadata, step=1234, timeout=timedelta(seconds=0.0)
            )

        # test mismatch case
        server.send_checkpoint(
            dst_ranks=[],
            step=2345,
            state_dict=expected,
            timeout=timedelta(seconds=10),
        )

        with self.assertRaisesRegex(urllib.error.HTTPError, r"Error 400"):
            server.recv_checkpoint(
                src_rank=0, metadata=metadata, step=1234, timeout=timedelta(seconds=10)
            )

        server.shutdown()

    def test_checkpoint_server_locking(self) -> None:
        server = CheckpointServer(
            timeout=timedelta(seconds=10),
        )

        # server should start up in a disallowed state this will block incoming
        # requests until allow_checkpoint is called
        self.assertTrue(server._checkpoint_lock.locked())
        self.assertTrue(server._disallowed)
        self.assertEqual(server._step, -1)

        # allow requests
        server.allow_checkpoint(1)

        self.assertFalse(server._checkpoint_lock.locked())
        self.assertFalse(server._disallowed)
        self.assertEqual(server._step, 1)

        # duplicate allow/disallow is fine
        server.allow_checkpoint(2)
        self.assertEqual(server._step, 2)

        server.disallow_checkpoint()
        server.disallow_checkpoint()
        self.assertTrue(server._checkpoint_lock.locked())
        self.assertTrue(server._disallowed)

        server.shutdown()

    def test_timed_acquire(self) -> None:
        lock = threading.Lock()

        with _timed_acquire(lock, timedelta(seconds=10)):
            self.assertTrue(lock.locked())

        self.assertFalse(lock.locked())

        lock.acquire()

        with self.assertRaisesRegex(
            TimeoutError, r"timed out acquiring lock after 0.0"
        ):
            with _timed_acquire(lock, timedelta(seconds=0.0)):
                pass

        self.assertTrue(lock.locked())


if __name__ == "__main__":
    unittest.main()
