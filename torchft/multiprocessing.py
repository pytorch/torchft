import queue
import time
from datetime import timedelta
from typing import Union

import torch.multiprocessing as mp


class _MonitoredQueue:
    def __init__(
        self,
        p: mp.Process,
        q: mp.Queue,
        poll_interval: timedelta = timedelta(seconds=1),
    ) -> None:
        """
        Args:
            p: process to monitor
            q: queue to monitor
            poll_interval: interval to poll the Process health when calling get/put
        """
        self._p = p
        self._q = q
        self._poll_interval_s: float = poll_interval.total_seconds()

    def get(self, timeout: Union[float, timedelta]) -> object:
        """
        Get an item from the queue. If the process is not alive, raise RuntimeError.
        If the queue is empty, wait for up to timeout seconds for an item to be
        available. If no item is available after timeout seconds, raise TimeoutError.

        Args:
            timeout: timeout in seconds
        """

        if isinstance(timeout, timedelta):
            timeout = timeout.total_seconds()

        start = time.perf_counter()
        while True:
            elapsed = time.perf_counter() - start
            if elapsed > timeout:
                raise TimeoutError(f"queue.get() timed out after {timeout} seconds")
            if not self._p.is_alive():
                raise RuntimeError(f"process is not alive {self._p.exitcode}")

            try:
                v = self._q.get(timeout=self._poll_interval_s)
                break
            except queue.Empty:
                continue

        if isinstance(v, Exception):
            raise v
        return v

    def put(self, obj: object, timeout: Union[float, timedelta]) -> None:
        """
        Put an item into the queue. If the process is not alive, raise RuntimeError.
        If the queue is full, wait for up to timeout seconds for an item to be
        available. If queue is full after timeout seconds, raise TimeoutError.

        If an exception is put into the queue, it will be raised when calling get().

        Args:
            obj: object to put into the queue
            timeout: timeout in seconds
        """
        if isinstance(timeout, timedelta):
            timeout = timeout.total_seconds()

        start = time.perf_counter()
        while True:
            elapsed = time.perf_counter() - start
            if elapsed > timeout:
                raise TimeoutError(f"queue.put() timed out after {timeout} seconds")
            if not self._p.is_alive():
                raise RuntimeError(f"process is not alive {self._p.exitcode}")

            try:
                self._q.put(obj, timeout=self._poll_interval_s)
                break
            except queue.Full:
                continue

    def close(self) -> None:
        self._q.close()

    def closed(self) -> bool:
        # pyre-ignore[16]: no attribute _closed
        return self._q._closed
