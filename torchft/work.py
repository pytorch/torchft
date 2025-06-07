import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Callable, Optional

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import Work

logger: logging.Logger = logging.getLogger(__name__)


class DummyWork(dist._Work):
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


class ErrorSwallowingWork(Work):
    def __init__(
        self,
        work: Work,
        report_error: Callable[[Exception], None],
        default_result: object,
    ) -> None:
        super().__init__()

        self._work = work
        self._default_result = default_result
        self._report_error = report_error

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        try:
            self._work.wait()
        except Exception as e:
            self._report_error(e)

        return True

    def get_future(self) -> torch.futures.Future[object]:
        return self._work.get_future()
