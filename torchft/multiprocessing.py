import queue
import time
from datetime import timedelta
from multiprocessing.connection import Connection
from typing import Union

import torch.multiprocessing as mp


class _MonitoredPipe:
    def __init__(self, pipe: "Connection[object, object]") -> None:
        self._pipe = pipe

    def send(self, obj: object) -> None:
        self._pipe.send(obj)

    def recv(self, timeout: Union[float, timedelta]) -> object:
        if isinstance(timeout, timedelta):
            timeout = timeout.total_seconds()
        if self._pipe.poll(timeout):
            out = self._pipe.recv()
            if isinstance(out, Exception):
                raise out
            return out
        else:
            raise TimeoutError(f"pipe.recv() timed out after {timeout} seconds")

    def close(self) -> None:
        self._pipe.close()

    def closed(self) -> bool:
        return self._pipe.closed
