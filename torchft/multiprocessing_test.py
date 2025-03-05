from multiprocessing.connection import Connection
from unittest import TestCase

import torch.multiprocessing as mp

from torchft.multiprocessing import _MonitoredPipe


def pipe_get(q: "Connection[object, object]") -> None:
    q.recv()


def pipe_put(q: "Connection[object, object]") -> None:
    q.recv()
    q.send(1)


class MultiprocessingTest(TestCase):
    def test_monitored_queue_put(self) -> None:
        ctx = mp.get_context("fork")
        local, remote = ctx.Pipe()
        p = ctx.Process(target=pipe_get, args=(remote,), daemon=True)
        p.start()
        del remote

        mq = _MonitoredPipe(local)
        mq.send(1)
        with self.assertRaisesRegex(ConnectionResetError, "Connection reset by peer"):
            while True:
                mq.send(1)

        mq.close()
        assert mq.closed()

    def test_monitored_queue_get(self) -> None:
        ctx = mp.get_context("fork")
        local, remote = ctx.Pipe()
        p = ctx.Process(target=pipe_put, args=(remote,), daemon=True)
        p.start()
        del remote

        mq = _MonitoredPipe(local)

        with self.assertRaisesRegex(TimeoutError, "timed out after 0.0 seconds"):
            mq.recv(timeout=0.0)

        # continue
        mq.send(1)

        self.assertEqual(mq.recv(timeout=10), 1)
        with self.assertRaises(EOFError):
            mq.recv(timeout=10)

        mq.close()
        assert mq.closed()
