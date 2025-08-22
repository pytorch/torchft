# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for TorchFT.
"""

from contextlib import nullcontext
from typing import Any, Optional, Union

import torch


def get_stream_context(
    stream: Optional[torch.Stream],
) -> Union[torch.cuda.StreamContext, torch.xpu.StreamContext, nullcontext[None]]:
    """
    Get the appropriate stream context for the given stream.

    This function provides a unified way to handle stream contexts across different
    accelerator types (CUDA, XPU).

    Args:
        stream: The stream to create a context for. If None, returns nullcontext.

    Returns:
        The appropriate stream context for the accelerator type, or nullcontext
        if stream is None or no accelerator is available.
    """
    if stream is not None:
        if torch.cuda.is_available():
            # pyre-fixme[6]: Expected `Optional[streams.Stream]` but got `_C.Stream`
            return torch.cuda.stream(stream)
        elif torch.xpu.is_available():
            # pyre-fixme[6]: Expected `Optional[streams.Stream]` but got `_C.Stream`
            return torch.xpu.stream(stream)
        else:
            return nullcontext()
    else:
        return nullcontext()
