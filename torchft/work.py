from contextlib import nullcontext
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist


class _DummyWork(dist._Work):
    def __init__(self, result: object) -> None:
        super().__init__()
        self.result_ = result
        # pyre-fixme[29]: Future is not a function
        self.future_: torch.futures.Future[object] = torch.futures.Future()
        self.future_.set_result(result)

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        return True

    def get_future(self) -> torch.futures.Future[object]:
        return self.future_


class _WorkWrapper(dist._Work):
    def __init__(
        self, work: dist._Work, fut: torch.futures.Future[torch.Tensor]
    ) -> None:
        super().__init__()
        self._work = work
        self._fut = fut

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        return True

    def get_future(self) -> torch.futures.Future[torch.Tensor]:
        return self._fut
