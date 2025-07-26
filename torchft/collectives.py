# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import TYPE_CHECKING

import torch

# pyre-ignore[21]: Could not find a module corresponding to import `triton`
import triton
from torch import cuda
from torch.distributed import ReduceOp
from torch.distributed.distributed_c10d import (
    AllgatherOptions,
    AllreduceOptions,
    AllToAllOptions,
    ReduceScatterOptions,
)
from torch.futures import Future

if TYPE_CHECKING:
    from torchft.process_group import ProcessGroup

from torchft.quantization import (
    fused_dequantize_from_fp8,
    fused_quantize_into_fp8,
    fused_reduce_fp8,
)


def _to_alltoall_options(
    opts: AllreduceOptions | ReduceScatterOptions,
) -> AllToAllOptions:
    alltoall_opts = AllToAllOptions()
    alltoall_opts.timeout = opts.timeout
    return alltoall_opts


def _to_allgather_options(
    opts: AllreduceOptions | ReduceScatterOptions,
) -> AllgatherOptions:
    allgather_opts = AllgatherOptions()
    allgather_opts.timeout = opts.timeout
    return allgather_opts


def get_padded_sizes(
    tensors: list[torch.Tensor],
    world_size: int,
) -> list[torch.Size]:
    """
    Calculate padded sizes for tensors to ensure they can be evenly
    divided across ranks.

    This function computes padded tensor sizes by rounding up the
    first dimension of each tensor to be a multiple of the world_size.
    This ensures that when tensors are split across ranks
    in distributed operations, each process receives an equal
    number of elements.

    Args:
        tensors: List of tensors whose sizes need to be padded
        world_size: Number of ranks in the distributed setup

    Returns:
        List of torch.Size objects with the first dimension padded
        to be a multiple of world_size

    Note:
        For 1D tensors, they are treated as 2D tensors with a
        second dimension of 1
    """
    padded_sizes = []
    for tensor in tensors:
        size = tensor.size()
        if len(size) == 1:
            size = (size[0], 1)
        padded_m = math.ceil(size[0] / world_size) * world_size
        padded_sizes.append(torch.Size((padded_m, *size[1:])))
    return padded_sizes


def allocate_reduce_scatter_output(
    tensors: list[torch.Tensor],
    world_size: int,
) -> tuple[torch.Tensor, list[torch.Size]]:
    """
    Allocate tensor for the output of a reduce-scatter operation.

    This function creates a single contiguous tensor to hold the results of a
    reduce-scatter operation across multiple ranks. It ensures that the tensor
    is properly sized and shaped to accommodate the results, where each rank
    will receive a portion of the reduced data.

    Args:
        tensors: List of input tensors for the reduce-scatter operation.
                All tensors must be on the same device and have the same
                data type.
        world_size: Number of ranks in the distributed setup

    Returns:
        A tuple containing:
        - A single contiguous tensor allocated for the reduce-scatter output
        - A list of padded sizes for the input tensors that were split across
          ranks

    Raises:
        AssertionError: If the input tensors are not all on the same device or
                       do not all have the same data type
    """
    device = tensors[0].device
    dtype = tensors[0].dtype
    for i in range(1, len(tensors)):
        assert (
            tensors[i].device == tensors[i - 1].device
        ), "All inputs must be on the same device"
        assert (
            tensors[i].dtype == tensors[i - 1].dtype
        ), "All inputs must be on the same dtype"

    padded_sizes = get_padded_sizes(tensors, world_size)

    chunks = []
    numels = [size.numel() // world_size for size in padded_sizes]
    tensor = torch.empty(
        (sum(numels),),
        device=device,
        dtype=dtype,
    )
    for split, padded_size in zip(torch.split(tensor, numels), padded_sizes):
        chunks.append(split.view(padded_size[0] // world_size, *padded_size[1:]))
    return tensor, padded_sizes


class _QuantizedOpFuture(Future[list[torch.Tensor]]):
    def __init__(
        self,
        sync_stream: cuda.Stream,
        keep_alive_tensors: list[torch.Tensor],
        return_tensors: list[torch.Tensor],
    ) -> None:
        super().__init__()
        self._sync_stream = sync_stream
        self._keep_alive_tensors = keep_alive_tensors
        self._return_tensors = return_tensors

    def wait(self) -> list[torch.Tensor]:
        # Wait for the synchronization to complete.
        cuda.current_stream().wait_stream(self._sync_stream)
        # Clean up intermediate buffers.
        del self._keep_alive_tensors
        return self._return_tensors


def reduce_scatter_quantized(
    output: torch.Tensor,
    inputs: list[torch.Tensor],
    opts: ReduceScatterOptions | ReduceOp,
    process_group: "ProcessGroup",
    sync_stream: cuda.Stream | None = None,
) -> Future[None]:
    """
    Performs a quantized reduce-scatter operation on a list of tensors.

    This function implements an optimized reduce-scatter that reduces communication
    overhead by quantizing tensors to FP8 format before sending them over the
    network. The algorithm works as follows:

    1. Quantize input tensors to FP8 format
    2. Distribute chunks of quantized tensors to all ranks using all-to-all
    3. Reduce chunks locally in higher precision after dequantization
    4. Dequantize the result back to the original precision for the current rank

    This implementation only supports the AVG and SUM reduce operations.

    Args:
        output: Pre-allocated tensor to store the output of the reduce-scatter operation
        inputs: List of tensors to be reduced and scattered. All tensors must be on
                the same CUDA device and have the same dtype.
        opts: Options for the reduce-scatter operation. Can be either a
              ReduceScatterOptions object or a ReduceOp enum.
        process_group: The process group to perform the reduce-scatter on.
        sync_stream: Optional CUDA stream to use for synchronization. If None,
                    a new stream will be created.

    Returns:
        A Future that can be used to wait for the operation to complete and
        clean up intermediate buffers.

    Raises:
        NotImplementedError: If the reduce operation is not ReduceOp.AVG or ReduceOp.SUM.
    """

    if isinstance(opts, ReduceOp):
        reducescatter_opts = ReduceScatterOptions()
        reducescatter_opts.reduceOp = opts
    else:
        reducescatter_opts = opts

    # Check if the reduceOp is AVG or SUM
    if reducescatter_opts.reduceOp not in {
        ReduceOp(ReduceOp.AVG),
        ReduceOp(ReduceOp.SUM),
    }:
        raise NotImplementedError(
            f"ReduceOp {reducescatter_opts.reduceOp} is not supported "
            f"for quantized reduce-scatter, only AVG and SUM are supported"
        )

    rank = process_group.rank()
    world_size = process_group.size()

    reduce_output_sizes = [
        torch.Size((s[0] // world_size, *s[1:]))
        for s in get_padded_sizes(inputs, world_size)
    ]
    reduce_output_numels = [s.numel() for s in reduce_output_sizes]
    reduce_outputs = [
        o.view(s)
        for o, s in zip(
            output.split(reduce_output_numels),
            reduce_output_sizes,
        )
    ]

    if sync_stream is None:
        sync_stream = cuda.Stream()

    assert sync_stream is not None
    # Ensure that all operations are completed on the current stream
    # before proceeding with all-reduce
    sync_stream.wait_stream(cuda.current_stream())
    with cuda.stream(sync_stream):
        # Quantize tensoers and compute their scales, all inlined in the
        # output tensor.
        quantized_inputs = fused_quantize_into_fp8(inputs, world_size)

        # Allocate output tensor where all-reduce results will be stored
        quantized_inputs_out = torch.zeros_like(quantized_inputs)
        # Collect chunks and their scales from other ranks
        process_group.alltoall_base(
            quantized_inputs_out.view(world_size, -1),
            quantized_inputs.view(world_size, -1),
            [],
            [],
            _to_alltoall_options(reducescatter_opts),
        ).wait()

        # Reduce chunks locally in higher precision after dequantization.
        # The output is again quantized.
        fused_reduce_fp8(
            inputs,
            quantized_inputs_out,
            world_size,
            rank,
            reducescatter_opts.reduceOp,
        )

        # Get view into the output tensor that corresponds to the
        # current rank
        quantized_reduce_scatter = (
            quantized_inputs_out.view(world_size, -1).split(1)[rank].squeeze(0)
        )
        # Dequantize the result back to the original precision for
        # the current rank
        fused_dequantize_from_fp8(
            reduce_outputs,
            quantized_reduce_scatter,
            1,
        )

        # pyre-ignore[29]
        return _QuantizedOpFuture(
            sync_stream,
            [
                quantized_inputs,
                quantized_inputs_out,
            ],
            [output],
        )


def allreduce_quantized(
    tensors: list[torch.Tensor],
    opts: AllreduceOptions | ReduceOp,
    process_group: "ProcessGroup",
    sync_stream: cuda.Stream | None = None,
) -> Future[list[torch.Tensor]]:
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

    # Check if the reduceOp is AVG or SUM
    if allreduce_opts.reduceOp not in {
        ReduceOp(ReduceOp.AVG),
        ReduceOp(ReduceOp.SUM),
    }:
        raise NotImplementedError(
            f"ReduceOp {allreduce_opts.reduceOp} is not supported "
            f"for quantized allreduce, only AVG and SUM are supported"
        )

    rank = process_group.rank()
    world_size: int = process_group.size()

    if sync_stream is None:
        sync_stream = cuda.Stream()

    assert sync_stream is not None
    # Ensure that all operations are completed on the current stream
    # before proceeding with all-reduce
    sync_stream.wait_stream(cuda.current_stream())
    with cuda.stream(sync_stream):
        # Quantize tensoers and compute their scales, all inlined in the
        # output tensor.
        quantized_tensors: torch.Tensor = fused_quantize_into_fp8(tensors, world_size)

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
            allreduce_opts.reduceOp,
        )

        # Collect reduced chunks from other ranks.
        work = process_group.allgather_into_tensor_coalesced(
            [quantized_tensors.view(world_size, -1)],
            [torch.split(quantized_tensors_out.view(world_size, -1), 1)[rank]],
            _to_allgather_options(allreduce_opts),
        )
        work.wait()
        fut = work.get_future()

        def callback(fut: Future[list[torch.Tensor]]) -> list[torch.Tensor]:
            # Dequantize and copy to output buffer.
            nonlocal tensors, quantized_tensors, world_size, sync_stream

            with torch.cuda.stream(sync_stream):
                # Setup stream dependency
                fut.wait()
                # Dequantize the result back to the original precision
                fused_dequantize_from_fp8(tensors, quantized_tensors, world_size)
                return tensors

        fut = fut.then(callback)
        return fut
