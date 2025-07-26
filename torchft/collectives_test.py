# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Callable
from unittest import TestCase, skipUnless

import torch
import torch.distributed as dist
from parameterized import parameterized
from torch import cuda
from torch.distributed import ReduceOp, ReduceScatterOptions

from torchft import _test_utils
from torchft.process_group import ProcessGroup
from torchft.process_group_test import MultiPgBaseTest

try:
    # pyre-ignore[21]: Could not find a module corresponding to import `triton`
    import triton
except ImportError:
    pass
else:
    from torchft.collectives import (
        allocate_reduce_scatter_output,
        allreduce_quantized,
        get_padded_sizes,
        reduce_scatter_quantized,
    )

    def _check_result_tolerance(
        actual: torch.Tensor, expected: torch.Tensor, tolerance: float
    ) -> None:
        diff = torch.abs(
            (expected - actual).div(expected.to(torch.float32) + 0.0000001)
        )
        mean_diff = diff.mean().item()

        if mean_diff > tolerance:
            print(f"Diff: {diff=}\n{expected=}\n{actual=}")
            raise AssertionError(f"Results not within tolerance {tolerance}")

    @skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= 2,
        "2 CUDA devices are required for this test",
    )
    class QuantizedAllReduceTest(MultiPgBaseTest):
        BACKEND = "nccl"
        WORLD_SIZE = 2

        def _run_parallel_collectives(
            self, collective: Callable[[ProcessGroup, int, str], None]
        ) -> None:
            futures = []
            for rank in range(self.WORLD_SIZE):
                pg = self.pg_pool[rank]
                device = f"cuda:{rank}"
                fut = self.executor.submit(collective, pg, rank, device)
                futures.append(fut)

            self._collect(futures)

        def _run_all_reduce_collective(
            self,
            pg: ProcessGroup,
            device: str,
            tensors_num: int,
            tensor_size: int,
            multiplier: float,
            tolerance: float,
            reduce_op: ReduceOp,
            dtype: torch.dtype,
        ) -> None:
            cuda.set_device(device)
            inp = (
                torch.rand(
                    tensors_num * tensor_size,
                    dtype=dtype,
                    device=device,
                )
                * multiplier
            )
            for split in _test_utils.gen_splits(inp, tensor_size):
                actual = inp.clone()
                expected = inp.clone()
                tensors = [
                    i.view(*s)
                    for s, i in zip(
                        split,
                        torch.split(actual, tensor_size),
                    )
                ]

                work = allreduce_quantized(tensors, reduce_op, pg)
                work.get_future().wait()

                work = pg.allreduce([expected], reduce_op)
                work.get_future().wait()

                _check_result_tolerance(actual, expected, tolerance)

        def _run_reduce_scatter_collective(
            self,
            pg: ProcessGroup,
            device: str,
            tensors_num: int,
            tensor_size: int,
            multiplier: float,
            tolerance: float,
            reduce_op: ReduceOp,
            dtype: torch.dtype,
        ) -> None:
            cuda.set_device(device)
            inp = (
                torch.rand(
                    tensors_num * tensor_size,
                    dtype=dtype,
                    device=device,
                )
                * multiplier
            )
            world_size = pg.size()
            for split in _test_utils.gen_splits(inp, tensor_size):
                actual = inp.clone()
                tensors = [
                    i.view(*s)
                    for s, i in zip(
                        split,
                        torch.split(actual, tensor_size),
                    )
                ]

                actual_output, _ = allocate_reduce_scatter_output(
                    tensors,
                    world_size,
                )

                opts = ReduceScatterOptions()
                opts.reduceOp = reduce_op

                fut = reduce_scatter_quantized(actual_output, tensors, opts, pg)
                fut.wait()

                padded_sizes = get_padded_sizes(tensors, world_size)
                padded_numel = sum(s.numel() for s in padded_sizes)

                padded_input = torch.empty(padded_numel, dtype=dtype, device=device)
                torch._chunk_cat(
                    tensors, dim=0, num_chunks=world_size, out=padded_input
                )

                expected_output = torch.empty(
                    padded_numel // world_size, dtype=dtype, device=device
                )

                work = pg.reduce_scatter([expected_output], [[padded_input]], opts)
                work.get_future().wait()

                _check_result_tolerance(actual_output, expected_output, tolerance)

        END_TO_END_CONFIGS: list[tuple[int, float, ReduceOp, torch.dtype]] = [
            (ts, m, o, t)
            for ts in [256, 1024, 2048]
            for m in [1.0, 100.0, 1000.0]
            for o in [ReduceOp.AVG, ReduceOp.SUM]
            for t in [torch.float32, torch.float16, torch.bfloat16]
        ]

        @parameterized.expand(END_TO_END_CONFIGS)
        def test_all_reduce_collective(
            self,
            tensor_size: int,
            multiplier: float,
            reduce_op: ReduceOp,
            dtype: torch.dtype,
        ) -> None:
            self._run_parallel_collectives(
                lambda pg, _, device: self._run_all_reduce_collective(
                    pg,
                    device,
                    2,
                    tensor_size,
                    multiplier,
                    0.04,
                    reduce_op,
                    dtype,
                )
            )

        @parameterized.expand(END_TO_END_CONFIGS)
        def test_reduce_scatter_collective(
            self,
            tensor_size: int,
            multiplier: float,
            reduce_op: ReduceOp,
            dtype: torch.dtype,
        ) -> None:
            self._run_parallel_collectives(
                lambda pg, _, device: self._run_reduce_scatter_collective(
                    pg,
                    device,
                    2,
                    tensor_size,
                    multiplier,
                    0.05,
                    reduce_op,
                    dtype,
                )
            )
