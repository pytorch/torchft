import asyncio
import queue
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from typing import Callable, Generator, Optional, TypeVar
from unittest.mock import Mock

import torch
from torch.futures import Future

T = TypeVar("T")


class _TimerHandle:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._timer_handle: Optional[asyncio.TimerHandle] = None
        self._cancelled = False

    def set_timer_handle(self, timer_handle: asyncio.TimerHandle) -> None:
        with self._lock:
            if self._cancelled:
                timer_handle.cancel()
                self._timer_handle = None
            else:
                self._timer_handle = timer_handle

    def cancel(self) -> None:
        with self._lock:
            assert not self._cancelled, "timer can only be cancelled once"
            self._cancelled = True
            if self._timer_handle is not None:
                self._timer_handle.cancel()
                self._timer_handle = None


class _TimeoutManager:
    """
    This class manages timeouts for code blocks, futures and CUDA events. It
    uses a background thread with an event loop to schedule the timeouts and
    call the callback function when the timeout is reached.

    Generally there is a single instance of this class that is used for all
    timeouts. The callbacks should not block otherwise other timeouts may not
    be processed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._event_loop_thread: Optional[threading.Thread] = None
        self._next_timer_id = 0

        # Ensures `_event_loop_thread` is not stuck
        self._watchdog_thread: Optional[threading.Thread] = None

        # Give this much time the the `_event_loop_thread` to confirm that
        # it is not stuck
        self._watchdog_interval = timedelta(seconds=30)

        # This queue is used to delete events on the main thread as cudaEventDestroy
        # can block if the CUDA queue is full.
        self._del_queue: queue.SimpleQueue[object] = queue.SimpleQueue()

    def _maybe_start_event_loop(self) -> asyncio.AbstractEventLoop:
        """
        Start the event loop if it has not already been started.
        """
        with self._lock:
            if self._event_loop is None:
                self._event_loop = asyncio.new_event_loop()
                self._event_loop_thread = threading.Thread(
                    target=self._event_loop.run_forever,
                    daemon=True,
                    name="TimeoutManager",
                )
                self._event_loop_thread.start()

                self._watchdog_thread = threading.Thread(
                    target=self._watchdog_loop, daemon=True
                )
                self._watchdog_thread.start()

            # pyre-fixme[7]: optional
            return self._event_loop

    def _watchdog_loop(self) -> None:
        while True:
            is_healthy = False

            def updated_health() -> None:
                nonlocal is_healthy
                is_healthy = True

            with self._lock:
                if self._event_loop is None:
                    return

                # The method passed to the event loop should finish fast.
                # It just updates a bool, which is also thread safe.
                self._event_loop.call_soon_threadsafe(updated_health)

            time.sleep(self._watchdog_interval.total_seconds())

            if not is_healthy:
                print("TimeoutManager is stuck. Exiting.")
                sys.exit(1)
                # Needed becuase `sys.exit` is mocked in unit tests.
                # If we don't return here, we don't break out of the loop.
                return

    def shutdown(self) -> None:
        """
        Shutdown the event loop and cancel all pending timeouts.
        """
        watchdog_thread = None
        with self._lock:
            if self._event_loop is not None:
                self._event_loop.call_soon_threadsafe(self._event_loop.stop)
                assert self._event_loop_thread is not None
                self._event_loop_thread.join()
                self._event_loop = None
                self._event_loop_thread = None

                # We can't join the watchdog thread here because it grabs `lock_`
                watchdog_thread = self._watchdog_thread

        if watchdog_thread is not None:
            # If `_maybe_start_event_loop` is called again, the it is possible the `join`
            # below will never finish.
            # This class assumes `_maybe_start_event_loop` will not be called after `shutdown`.
            # If this functionality is required in the future, we could change the class to
            # support this. Or create multiple instances of this class.
            watchdog_thread.join()

    def register(self, fut: Future[T], timeout: timedelta) -> Future[T]:
        """
        Registers a future that will be cancelled after the specified timeout.
        """
        # bypass timeout for mock futures
        if isinstance(fut, Mock):
            return fut

        self._clear_del_queue()

        loop = self._maybe_start_event_loop()

        timed_fut: Future[T] = Future()
        handle: _TimerHandle = _TimerHandle()
        loop.call_soon_threadsafe(
            self._register_callback,
            loop,
            lambda: timed_fut.set_exception(
                # pyre-fixme[6]: e is not T
                TimeoutError(f"future did not complete within {timeout}")
            ),
            timeout,
            handle,
        )

        stream: Optional[torch.Stream] = (
            torch.accelerator.current_stream() if torch.accelerator.is_available() else None
        )

        def callback(fut: Future[T]) -> None:
            if stream is not None:
                if torch.cuda.is_available():
                    context = torch.cuda.stream(stream)
                elif torch.xpu.is_available():
                    context = torch.xpu.stream(stream)
                else:
                    context = nullcontext()
            else:
                context = nullcontext()
            with context:
                handle.cancel()
                try:
                    timed_fut.set_result(fut.wait())
                except Exception as e:
                    try:
                        # this can throw if the future is already done
                        # pyre-fixme[6]: e is not T
                        timed_fut.set_exception(e)
                    except Exception:
                        pass

        fut.add_done_callback(callback)
        return timed_fut

    def stream_timeout(self, callback: Callable[[], None], timeout: timedelta) -> None:
        self._clear_del_queue()

        loop = self._maybe_start_event_loop()

        event: torch.Event = torch.Event()
        event.record()

        def handler() -> None:
            if not event.query():
                callback()

            # cudaEventDestroy can block so we never want to delete in the event
            # loop. Put it on the del queue so we can delete it in the main
            # thread.
            self._del_queue.put(event)

        loop.call_soon_threadsafe(
            self._register_callback, loop, handler, timeout, _TimerHandle()
        )

    @classmethod
    def _register_callback(
        cls,
        loop: asyncio.AbstractEventLoop,
        callback: Callable[[], None],
        timeout: timedelta,
        handle: _TimerHandle,
    ) -> None:
        timer_handle = loop.call_later(
            timeout.total_seconds(),
            callback,
        )
        handle.set_timer_handle(timer_handle)

    @contextmanager
    def context_timeout(
        self, callback: Callable[[], None], timeout: timedelta
    ) -> Generator[None, None, None]:
        self._clear_del_queue()

        loop = self._maybe_start_event_loop()
        handle = _TimerHandle()

        loop.call_soon_threadsafe(
            self._register_callback, loop, callback, timeout, handle
        )

        yield

        handle.cancel()

    def _clear_del_queue(self) -> int:
        """
        Clear the queue of futures to be deleted.

        Returns the number of items deleted.
        """
        count = 0
        while True:
            try:
                # get and immediately discard item
                item = self._del_queue.get_nowait()
                refcount = sys.getrefcount(item)
                assert (
                    # 1 from item, 1 from getrefcount
                    refcount
                    == 2
                ), f"items in del_queue reference should not have other references, found {refcount=}"
                del item

                count += 1
            except queue.Empty:
                break

        return count


_TIMEOUT_MANAGER = _TimeoutManager()


def future_timeout(fut: Future[T], timeout: timedelta) -> Future[T]:
    """
    Return a Future that completes with the result of the given Future within
    the given timeout or with a TimeoutError.

    Args:
        fut: The Future to wait for
        timeout: The timeout to wait for the Future to complete

    Returns:
        The future with a timeout
    """
    return _TIMEOUT_MANAGER.register(fut, timeout)


def future_wait(fut: Future[T], timeout: timedelta) -> T:
    """
    Wait for a Future to complete up to a timeout.

    Args:
        fut: The Future to wait for
        timeout: The timeout to wait for the Future to complete

    Returns:
        The result of the Future if it completed within the timeout.

    Raises:
        TimeoutError if the Future did not complete within the timeout.
        Any other exception that occurred in the Future.
    """

    event: threading.Event = threading.Event()

    def callback(fut: Future[T]) -> T:
        event.set()
        return fut.wait()

    fut = fut.then(callback)

    if not event.wait(timeout=timeout.total_seconds()):
        raise TimeoutError(f"future did not complete within {timeout}")

    return fut.wait()


def stream_timeout(callback: Callable[[], None], timeout: timedelta) -> None:
    """
    Registers a callback that will be called after the specified timeout if
    the current stream doesn't complete in time.

    This uses a cuda Event to track the completion of the current stream. If
    the stream is not complete after the timeout, the callback is called.

    Args:
        callback: The callback to call if the stream doesn't complete in time.
        timeout: The timeout to wait for the stream to complete.
    """
    _TIMEOUT_MANAGER.stream_timeout(callback, timeout)


@contextmanager
def context_timeout(
    callback: Callable[[], None], timeout: timedelta
) -> Generator[None, None, None]:
    """
    Registers a callback that will be called after the specified timeout if
    the current contextmanager doesn't exit in time.

    Args:
        callback: The callback to call if we time out.
        timeout: How long to wait for the contextmanager to exit.
    """

    with _TIMEOUT_MANAGER.context_timeout(callback, timeout):
        yield
