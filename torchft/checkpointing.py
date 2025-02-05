# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Checkpointing
==============

This module implements methods for checkpointing and resuming training from a checkpoint.
"""

import io
import logging
import socket
import threading
import urllib.request
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import timedelta
from http.server import BaseHTTPRequestHandler
from typing import Generator, Generic, List, Optional, TypeVar, cast

from torchft.http import _IPv6HTTPServer
from torchft.serialization import streaming_load, streaming_save

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


class CheckpointTransport(Generic[T], ABC):
    @abstractmethod
    def metadata(self) -> str:
        """
        Returns a string that will be used by the remote CheckpointTransport to fetch the checkpoint.
        """
        ...

    @abstractmethod
    def send_checkpoint(
        self, dst_ranks: List[int], step: int, state_dict: T, timeout: timedelta
    ) -> None:
        """
        Sends the checkpoint, only called when there is a rank that is behind.

        This may be async.

        Args:
            dst_ranks: the ranks to send to
            step: the step number to send
            state_dict: the state dict to send
            timeout: the timeout to wait for the checkpoint to be sent
        """
        ...

    def disallow_checkpoint(self) -> None:
        """
        Called after send_checkpoint to wait for the checkpoint to be sent.

        Once this returns, the state_dict may be mutated so no further data should be sent.
        """
        ...

    @abstractmethod
    def recv_checkpoint(
        self, src_rank: int, metadata: str, step: int, timeout: timedelta
    ) -> T:
        """
        Receives the checkpoint from the given rank.

        Args:
            src_rank: the rank to receive the checkpoint from
            metadata: the metadata returned by the remote CheckpointTransport
            step: the step number to receive
            timeout: the timeout to wait for the checkpoint
        """
        ...

    def shutdown(self, wait: bool = True) -> None:
        """
        Called to shutdown the checkpoint transport.

        Args:
            wait: whether to wait for the transport to shutdown
        """


@contextmanager
def _timed_acquire(
    lock: threading.Lock, timeout: timedelta
) -> Generator[None, None, None]:
    """
    Acquire a lock with a timeout.

    Args:
        lock: the lock to acquire
        timeout: the timeout to acquire the lock
    """
    if not lock.acquire(timeout=timeout.total_seconds()):
        raise TimeoutError(f"timed out acquiring lock after {timeout}")
    try:
        yield
    finally:
        lock.release()


class CheckpointServer(CheckpointTransport[T]):
    """
    This is an HTTP server that can be used to transfer checkpoints
    between workers.

    This allows for fast recovery of workers by fetching the current weights
    from an existing worker.

    Args:
        state_dict: a callable that returns the state dict to be transferred
    """

    def __init__(self, timeout: timedelta) -> None:
        self._checkpoint_lock = threading.Lock()
        self._disallowed = False
        self._step = -1
        self._timeout = timeout
        self._state_dict: Optional[T] = None

        # We don't allow checkpoints until the first send_checkpoint to avoid
        # serving the default step=-1 invalid checkpoint.
        self.disallow_checkpoint()

        ckpt_server = self

        class RequestHandler(BaseHTTPRequestHandler):
            # set request socket timeout to avoid hanging forever
            timeout = self._timeout.total_seconds()

            def do_GET(self):
                try:
                    # validate socket timeout is actually set
                    assert self.connection.gettimeout() == self.timeout

                    with _timed_acquire(
                        ckpt_server._checkpoint_lock, ckpt_server._timeout
                    ):
                        step = ckpt_server._step

                        if self.path != f"/checkpoint/{step}":
                            self.send_response(400)
                            self.send_header("Content-type", "text/plain")
                            self.end_headers()
                            self.err(
                                f"invalid checkpoint requested, serving {step} but got {self.path}"
                            )
                            return

                        self.send_response(200)
                        self.send_header("Content-type", "application/octet-stream")
                        self.end_headers()

                        state_dict = ckpt_server._state_dict

                        streaming_save(state_dict, self.wfile)
                except Exception as e:
                    logger.exception(
                        f"Exception in checkpoint server when handling {self.path=}: {e}",
                    )
                    self.send_response(500, str(e))
                    self.end_headers()

            def err(self, msg: str) -> None:
                logger.error(msg)
                self.wfile.write(msg.encode())

        server_address = ("", 0)
        self._server = _IPv6HTTPServer(server_address, RequestHandler)
        logger.info(f"Started CheckpointServer on {self.address()}...")

        self._thread = threading.Thread(
            target=self._serve,
            args=(),
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def load_from_address(cls, address: str, timeout: timedelta) -> T:
        """
        Loads a checkpoint from the given address.

        Args:
            address: the HTTP address to load the checkpoint from
        """
        logger.info(f"fetching checkpoint from {address}")

        with urllib.request.urlopen(address, timeout=timeout.total_seconds()) as f:
            data = f.read()

        reader = io.BytesIO(data)
        state_dict = streaming_load(reader)
        return cast(T, state_dict)

    def address(self) -> str:
        """
        Returns the HTTP address to fetch a checkpoint from this server. Step must be appended to the end of the address.

        Format: http://host:port/checkpoint/1234

        Returns:
            an HTTP address
        """
        port = self._server.socket.getsockname()[1]
        return f"http://{socket.gethostname()}:{port}/checkpoint/"

    def _serve(self) -> None:
        try:
            self._server.serve_forever()
        except Exception as e:
            logger.exception("got exception in checkpoint server")

    def disallow_checkpoint(self) -> None:
        """
        Disallows serving the checkpoint.

        All requests will block until allow_checkpoint is called.
        """
        if not self._disallowed:
            self._disallowed = True
            self._checkpoint_lock.acquire()

    def allow_checkpoint(self, step: int) -> None:
        """
        Allows serving the checkpoint with the specified step number.

        Args:
            step: the step number to serve
        """
        self._step = step

        if self._disallowed:
            self._disallowed = False
            self._checkpoint_lock.release()

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the server.
        """
        if not wait:
            # hack for nonblocking shutdown of socketserver threads
            # pyre-fixme[16]: no attribute `__shutdown_request`.
            self._server.__shutdown_request = True
        if wait:
            self._server.shutdown()
            self._thread.join()

    def metadata(self) -> str:
        return self.address()

    def send_checkpoint(
        self, dst_ranks: List[int], step: int, state_dict: T, timeout: timedelta
    ) -> None:
        self._state_dict = state_dict
        self.allow_checkpoint(step)

    def recv_checkpoint(
        self, src_rank: int, metadata: str, step: int, timeout: timedelta
    ) -> T:
        return self.load_from_address(f"{metadata}{step}", timeout)
