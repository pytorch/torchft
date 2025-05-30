# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import torch

# pyre-ignore[21]: Could not find a module corresponding to import `triton`
import triton
from torch import cuda
from torch.distributed.distributed_c10d import (
    AllgatherOptions,
    AllreduceOptions,
    AllToAllOptions,
    ReduceOp,
)
from torch.futures import Future

if TYPE_CHECKING:
    from torchft.process_group import ProcessGroup

from torchft.quantization import (
    fused_dequantize_from_fp8,
    fused_quantize_into_fp8,
    fused_reduce_fp8,
)


def _to_alltoall_options(opts: AllreduceOptions) -> AllToAllOptions:
    alltoall_opts = AllToAllOptions()
    alltoall_opts.timeout = opts.timeout
    return alltoall_opts


def _to_allgather_options(opts: AllreduceOptions) -> AllgatherOptions:
    allgather_opts = AllgatherOptions()
    allgather_opts.timeout = opts.timeout
    return allgather_opts


def allreduce_quantized(
    tensors: list[torch.Tensor],
    opts: AllreduceOptions | ReduceOp,
    process_group: "ProcessGroup",
    sync_stream: cuda.Stream | None = None,
) -> Future[None]:
    """
    Performs a quantized all-reduce operation on a list of tensors.

    This function implements an optimized all-reduce that reduces communication
    overhead by quantizing tensors to FP8 format before sending them over the
    network. The algorithm works as follows:

    1. Quantize input tensors to FP8 format
    2. Distribute chunks of quantized tensors to all ranks using all-to-all
    3. Reduce chunks locally in higher precision after dequantization
    4. Collect reduced chunks from all ranks using all-gather
    5. Dequantize the result back to the original precision

    This implementation only supports the AVG reduce operation.

    Args:
        tensors: List of tensors to be reduced. All tensors must be on the same
            CUDA device and have the same dtype.
        opts: Options for the all-reduce operation. Can be either an
            AllreduceOptions object or a ReduceOp enum. If a ReduceOp is
            provided, it must be ReduceOp.AVG.
        process_group: The process group to perform the all-reduce on.
        sync_stream: Optional CUDA stream to use for synchronization. If None,
            a new stream will be created.

    Returns:
        A Future that can be used to wait for the operation to complete and
        clean up intermediate buffers.

    Raises:
        NotImplementedError: If the reduce operation is not ReduceOp.AVG.
    """
    if isinstance(opts, ReduceOp):
        allreduce_opts = AllreduceOptions()
        allreduce_opts.reduceOp = opts
    else:
        allreduce_opts = opts

    # Check if the reduceOp is AVG, as only AVG is supported
    if allreduce_opts.reduceOp != ReduceOp.AVG:
        raise NotImplementedError(
            f"ReduceOp {allreduce_opts.reduceOp} is not supported "
            f"for quantized allreduce, only AVG is supported"
        )

    rank = process_group.rank()
    world_size = process_group.size()

    if sync_stream is None:
        sync_stream = cuda.Stream()

    assert sync_stream is not None
    # Ensure that all operations are completed on the current stream
    # before proceeding with all-reduce
    sync_stream.wait_stream(cuda.current_stream())
    with cuda.stream(sync_stream):
        # Quantize tensoers and compute their scales, all inlined in the
        # output tensor.
        quantized_tensors = fused_quantize_into_fp8(tensors, world_size)

        # Allocate output tensor where all-reduce results will be stored
        quantized_tensors_out = torch.zeros_like(quantized_tensors)
        # Collect chunks and their scales from other ranks
        process_group.alltoall_base(
            quantized_tensors_out.view(world_size, -1),
            quantized_tensors.view(world_size, -1),
            [],
            [],
            _to_alltoall_options(allreduce_opts),
        ).wait()

        # Reduce chunks locally in higher precision after dequantization.
        # The output is again quantized.
        fused_reduce_fp8(
            tensors,
            quantized_tensors_out,
            world_size,
            rank,
        )

        # Collect reduced chunks from other ranks.
        process_group.allgather_into_tensor_coalesced(
            [quantized_tensors.view(world_size, -1)],
            [torch.split(quantized_tensors_out.view(world_size, -1), 1)[rank]],
            _to_allgather_options(allreduce_opts),
        ).wait()

        # Dequantize and copy to output buffer.
        fused_dequantize_from_fp8(tensors, quantized_tensors, world_size)

        class QuantizedAllReduceFuture(Future[None]):
            def __init__(
                self,
                sync_stream: cuda.Stream,
                quantized_tensors: torch.Tensor,
                quantized_tensors_out: torch.Tensor,
            ) -> None:
                super().__init__()
                self._sync_stream = sync_stream
                self._quantized_tensors = quantized_tensors
                self._quantized_tensors_out = quantized_tensors_out

            def wait(self) -> None:
                # Wait for the synchronization to complete.
                cuda.current_stream().wait_stream(self._sync_stream)
                # Clean up intermediate buffers.
                del self._quantized_tensors_out
                del self._quantized_tensors

        # pyre-ignore[29]
        return QuantizedAllReduceFuture(
            sync_stream, quantized_tensors, quantized_tensors_out
        )
