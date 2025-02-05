from unittest import TestCase

import torch.multiprocessing as mp

from torchft.multiprocessing import _MonitoredQueue


def queue_get(q: mp.Queue) -> None:
    q.get()


def queue_put(q: mp.Queue) -> None:
    q.put(1)


class MultiprocessingTest(TestCase):
    def test_monitored_queue_put(self) -> None:
        ctx = mp.get_context("fork")
        q = ctx.Queue(maxsize=1)
        p = ctx.Process(target=queue_get, args=(q,), daemon=True)
        p.start()

        mq = _MonitoredQueue(p, q)
        mq.put(1, timeout=10)
        mq.put(1, timeout=10)
        with self.assertRaisesRegex(RuntimeError, "process is not alive 0"):
            mq.put(1, timeout=10)

        with self.assertRaisesRegex(TimeoutError, "timed out after 0.0 seconds"):
            mq.put(1, timeout=0.0)

        mq.close()

    def test_monitored_queue_get(self) -> None:
        ctx = mp.get_context("fork")
        q = ctx.Queue(maxsize=1)
        p = ctx.Process(target=queue_put, args=(q,), daemon=True)
        p.start()

        mq = _MonitoredQueue(p, q)
        self.assertEqual(mq.get(timeout=10), 1)
        with self.assertRaisesRegex(RuntimeError, "process is not alive 0"):
            mq.get(timeout=10)

        with self.assertRaisesRegex(TimeoutError, "timed out after 0.0 seconds"):
            mq.get(timeout=0.0)

        mq.close()
