import io
import warnings
from typing import IO

import torch


def _fallback_save(obj: object, f: IO[bytes]) -> None:
    warnings.warn(
        "using slow fallback torch.save implementation, please upgrade to PT 2.7+ for fast streaming saves"
    )

    torch.save(obj, f)


def _fallback_load(f: IO[bytes], weights_only: bool = True) -> object:
    warnings.warn(
        "using slow fallback torch.load implementation, please upgrade to PT 2.7+ for fast streaming loads"
    )

    # torch.load requires a seekable file object
    buf = f.read()
    reader = io.BytesIO(buf)

    return torch.load(reader, weights_only=weights_only)


try:
    # upgrade to PT 2.7 once released
    from torch.distributed._serialization import _streaming_load, _streaming_save
except ImportError:
    _streaming_load = _fallback_load
    _streaming_save = _fallback_save
