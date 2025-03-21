# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import os
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from datetime import timedelta
from typing import Any, Callable, Dict, List, cast
from unittest import TestCase, skipUnless
from unittest.mock import Mock

import torch
import torch.distributed as dist
from parameterized import parameterized
from torch import nn
from torch._C._distributed_c10d import (
    AllgatherOptions,
    AllreduceCoalescedOptions,
    AllreduceOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    ReduceOp,
    ReduceScatterOptions,
    _resolve_process_group,
)
from torch.distributed import (
    ReduceOp,
    TCPStore,
    _functional_collectives,
    get_world_size,
)
from torch.distributed.device_mesh import init_device_mesh

from torchft.manager import Manager
from torchft.process_group import (
    ErrorSwallowingProcessGroupWrapper,
    ManagedProcessGroup,
    ProcessGroup,
    ProcessGroupBabyGloo,
    ProcessGroupBabyNCCL,
    ProcessGroupDummy,
    ProcessGroupGloo,
    ProcessGroupNCCL,
    ProcessGroupWrapper,
    _DummyWork,
    _ErrorSwallowingWork,
    _ManagedWork,
    extend_device_mesh,
    ft_init_device_mesh,
)


def dummy_init_pg() -> None:
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, store=dist.HashStore()
        )


def _test_pg(
    pg: ProcessGroup,
    example_tensor: torch.Tensor = torch.randn((2, 3), dtype=torch.float32),
) -> Dict[str, dist._Work]:
    """
    Helper function to test a set of collective operations on a given process group.
    """

    shape: torch.Size = example_tensor.shape
    dtype: torch.dtype = example_tensor.dtype

    # Create some dummy tensors for testing
    input_tensor = example_tensor.clone()
    output_tensors = [
        [torch.empty_like(input_tensor) for _ in range(get_world_size(pg))]
    ]
    tensor_list = [torch.empty_like(input_tensor)]

    def check_tensors(arg: Any) -> None:  # pyre-ignore[2]
        """Recursively check tensors for expected shape and dtype."""
        if isinstance(arg, torch.Tensor):
            assert arg.dtype == dtype, f"Output dtype mismatch: {arg.dtype} != {dtype}"
            assert arg.shape == shape, f"Output shape mismatch: {arg.shape} != {shape}"
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                check_tensors(item)

    # Test collectives. send/recv require multiple processes to test, so we skip them here
    collectives = [
        ("allreduce", ([input_tensor], AllreduceOptions())),
        ("allreduce", ([input_tensor], ReduceOp.SUM)),
        ("allreduce_coalesced", ([input_tensor], AllreduceCoalescedOptions())),
        ("allgather", (output_tensors, [input_tensor], AllgatherOptions())),
        (
            "allgather_into_tensor_coalesced",
            (output_tensors[0], [input_tensor], AllgatherOptions()),
        ),
        (
            "alltoall_base",
            (
                output_tensors[0][0],
                input_tensor,
                [input_tensor.shape[0]],
                [input_tensor.shape[0]],
                AllToAllOptions(),
            ),
        ),
        ("barrier", (BarrierOptions(),)),
        ("broadcast", (tensor_list, BroadcastOptions())),
        ("broadcast_one", (input_tensor, 0)),
        (
            "reduce_scatter",
            (output_tensors[0], [[input_tensor]], ReduceScatterOptions()),
        ),
        (
            "reduce_scatter_tensor_coalesced",
            (output_tensors[0], [input_tensor], ReduceScatterOptions()),
        ),
    ]
    works: Dict[str, dist._Work] = {}

    for coll_str, args in collectives:
        try:
            coll = getattr(pg, coll_str)
            work = coll(*args)
            works[coll_str] = work
            work.wait()
            fut = work.get_future()
            fut.wait()
            # Check that all tensor arguments have the expected shapes and dtypes
            check_tensors(args)
        except RuntimeError as e:
            if f"does not support {coll_str}" in str(e):
                # Skip collectives that are not supported by the backend.
                continue
            raise e

    return works


def run_allgather_test(pg: ProcessGroup, rank: int, tensor: torch.Tensor) -> None:
    """Test allgather collective operation.

    Suppose each rank's local tensor = [rank+1, rank+2],
    we allgather => gather onto a list of length world_sz.
    """
    world_sz = pg.size()
    to_gather = torch.stack([tensor, tensor + 1], dim=0)
    # shape: (2,)
    to_gather = to_gather.reshape(-1)

    # Gathers as follows: [ [ recv0 ], [ recv1 ], ... [ recv_{sz-1} ] ]
    # Each recv is shape (2,)
    output_list = [
        torch.zeros(2, device=tensor.device, dtype=tensor.dtype)
        for _ in range(world_sz)
    ]

    work = pg.allgather([output_list], [to_gather], AllgatherOptions())
    work.wait()

    for r in range(world_sz):
        expected = torch.tensor(
            [r + 1, r + 2], device=tensor.device, dtype=tensor.dtype
        )
        torch.testing.assert_close(output_list[r], expected)


def run_allgather_into_tensor_coalesced_test(
    pg: ProcessGroup, rank: int, tensor: torch.Tensor
) -> None:
    """Test allgather tensor coalesced collective operation.

    This example gathers two local tensors, T0 and T1, from each rank into corresponding
    output tensors.

    For world_sz = n, each rank r has:
        T0 = [r+1],
        T1 = [r+10]

    After allgather_into_tensor_coalesced, we result in two tensors: out0, out1,
    both length n.

    out0 gathers T0 from all ranks, out1 gathers T1 from all ranks.

    We verify that out0[k] == [k+1] and out1[k] == [k+10] for all k.

    """
    world_sz = pg.size()

    if world_sz < 2:
        return

    t0 = torch.tensor([rank + 1], device=tensor.device, dtype=tensor.dtype)
    t1 = torch.tensor([rank + 10], device=tensor.device, dtype=tensor.dtype)

    out0 = torch.zeros(world_sz, device=tensor.device, dtype=tensor.dtype)
    out1 = torch.zeros(world_sz, device=tensor.device, dtype=tensor.dtype)

    work = pg.allgather_into_tensor_coalesced(
        [out0, out1], [t0, t1], AllgatherOptions()
    )
    work.wait()

    for r in range(world_sz):
        expected0 = torch.tensor([r + 1], device=t0.device, dtype=t0.dtype)
        torch.testing.assert_close(out0[r], expected0[0])
        expected1 = torch.tensor([r + 10], device=t1.device, dtype=t1.dtype)
        torch.testing.assert_close(out1[r], expected1[0])


def run_allreduce_test(pg: ProcessGroup, rank: int, tensor: torch.Tensor) -> None:
    """Test allreduce collective operation.

    Assume each rank's tensor has value = rank + 1.
    The final result after allreduce(SUM) should be sum(r=1,...,world_sz-1).
    """
    tc = tensor.clone()
    world_sz = pg.size()
    work = pg.allreduce([tc], ReduceOp.SUM)
    work.wait()
    expected_val = sum(r + 1 for r in range(world_sz))
    torch.testing.assert_close(tc, torch.tensor([expected_val], device=tensor.device))


def run_allreduce_coalesced_test(
    pg: ProcessGroup, rank: int, tensor: torch.Tensor
) -> None:
    """Test allreduce_coalesced collective operation.

    Assume each rank's tensor has value = rank + 1.
    We coalesce 1 tensors:
    - t0 = [rank + 1]
    - t1 = [rank + 2]

    Our final sum should be sum(r=1,...,world_sz-1) + sum(r=2,...,world_sz-1).
    """
    world_sz = pg.size()
    t0 = tensor.clone()
    t1 = tensor.clone() + 1
    work = pg.allreduce_coalesced([t0, t1], AllreduceCoalescedOptions())
    work.wait()
    sum_t0 = sum(r + 1 for r in range(world_sz))
    sum_t1 = sum(r + 2 for r in range(world_sz))
    torch.testing.assert_close(t0, torch.tensor([sum_t0], device=t0.device))
    torch.testing.assert_close(t1, torch.tensor([sum_t1], device=t1.device))


def run_alltoall_test(pg: ProcessGroup, rank: int, tensor: torch.Tensor) -> None:
    """Test all-to-all collective operation.

    Suppose each rank's local tensor = [rank*ws+1, rank*ws+2, ..., rank*ws + n]

    e.g.:
    rank=0 => [1,2]
    rank=1 => [3,4]

    After all-to-all, rank r's output[k] = the element from rank k that is destined for rank r,
    e.g.: (k*n) + (r+1):

    rank=0 => [1,3]
    rank=1 => [2,4]

    """
    world_sz = pg.size()
    if world_sz < 2:
        return

    input_tensor = torch.arange(
        start=rank * world_sz + 1,
        end=rank * world_sz + 1 + world_sz,
        device=tensor.device,
        dtype=tensor.dtype,
    )
    output_tensor = torch.empty(world_sz, device=tensor.device, dtype=tensor.dtype)

    send_sz = [1] * world_sz
    recv_sz = [1] * world_sz

    alltoall_work = pg.alltoall_base(
        output_tensor, input_tensor, send_sz, recv_sz, AllToAllOptions()
    )
    alltoall_work.wait()

    expected = torch.empty(world_sz, device=tensor.device, dtype=tensor.dtype)
    for k in range(world_sz):
        val = k * world_sz + (rank + 1)
        expected[k] = val

    torch.testing.assert_close(output_tensor, expected)


def run_broadcast_test(pg: ProcessGroup, rank: int, tensor: torch.Tensor) -> None:
    """Test broadcast collective operation.

    rank0 will broadcast a known value and all other ranks should get it.
    """
    broadcast_tensor = tensor.clone() if rank == 0 else torch.zeros_like(tensor)
    broadcast_work = pg.broadcast([broadcast_tensor], BroadcastOptions())
    broadcast_work.wait()
    expected_broadcast = torch.tensor([1], device=tensor.device)
    torch.testing.assert_close(broadcast_tensor, expected_broadcast)


def run_broadcast_one_test(pg: ProcessGroup, rank: int, tensor: torch.Tensor) -> None:
    """Test broadcast_one collective operation.

    rank0 will broadcast a known value and all other ranks should get it.
    """
    broadcast_one_tensor = tensor.clone() if rank == 0 else torch.zeros_like(tensor)
    broadcast_one_work = pg.broadcast_one(broadcast_one_tensor, 0)
    broadcast_one_work.wait()
    torch.testing.assert_close(
        broadcast_one_tensor, torch.tensor([1], device=tensor.device)
    )


def run_barrier_test(pg: ProcessGroup, rank: int, tensor: torch.Tensor) -> None:
    """Test barrier collective operation."""
    opts = BarrierOptions()
    if tensor.is_cuda:
        device_id = tensor.device.index
        opts.device_ids = [device_id]
    barrier_work = pg.barrier(opts)
    barrier_work.wait()


def run_send_recv_test(pg: ProcessGroup, rank: int, tensor: torch.Tensor) -> None:
    """Test send/recv point-to-point operations.

    Simple point-to-point between ranks 0 and 1, ignored for other ranks.
    """
    if pg.size() < 2:
        return
    if rank == 0:
        send_tensor = tensor.clone()
        send_work = pg.send([send_tensor], 1, 0)
        send_work.wait()
    elif rank == 1:
        recv_tensor = torch.zeros_like(tensor)
        recv_work = pg.recv([recv_tensor], 0, 0)
        recv_work.wait()
        expected = torch.tensor([1], device=tensor.device)
        torch.testing.assert_close(recv_tensor, expected)


def run_reduce_scatter_test(pg: ProcessGroup, rank: int, tensor: torch.Tensor) -> None:
    """Test reduce_scatter collective operation.

    Assume each rank creates a matrix where each row r contains values:
        [r * world_sz + 1, ..., r * world_sz + world_sz]

    For example, with world_size=2:
        [[1, 2],
         [3, 4]]

    The reduce_scatter operation then:
        - Reduces (sums) corresponding rows across all ranks
        - Scatters the results so each rank gets one row of the final sum
        - Since all ranks had the same initial data, the expected result for each rank r is:
            rank r receives: [rworld_sz + 1, ..., rworld_sz + world_sz] * world_sz

    For example, with 2 ranks:
        rank 0 gets: [1, 2] * 2 = [2, 4] (first row)
        rank 1 gets: [3, 4] * 2 = [6, 8] (second row)
    """
    if tensor.device.type == "cpu":
        return
    # reduce scatter not supported on GLOO
    world_sz = pg.size()
    if world_sz < 2:
        return

    local_data = []
    for r in range(world_sz):
        row_vals = torch.arange(
            start=r * world_sz + 1,
            end=r * world_sz + world_sz + 1,
            device=tensor.device,
            dtype=torch.float32,
        )
        local_data.append(row_vals)

    out = torch.zeros(world_sz, device=tensor.device, dtype=torch.float32)
    opts = ReduceScatterOptions()
    opts.reduceOp = ReduceOp.SUM
    work = pg.reduce_scatter([out], [local_data], opts)
    work.wait()

    expected_row = torch.arange(
        start=rank * world_sz + 1,
        end=rank * world_sz + world_sz + 1,
        device=tensor.device,
        dtype=torch.float32,
    )
    expected_sum = expected_row * world_sz
    torch.testing.assert_close(out, expected_sum)


def run_reduce_scatter_tensor_coalesced_test(
    pg: ProcessGroup, rank: int, tensor: torch.Tensor
) -> None:
    """Test reduce_scatter tensor coalesced collective operation.

      We define two 2D tensors, each shaped [world_sz, world_sz] which is replicated on each rank.

      reduce_scatter coalesced will reduce each row of each tensor, then scatter the results to each rank.
      Because these are replicated on all ranks, the reduced sum for each row is:
          [r*world_sz + 1, ..., r*world_sz + world_sz] * world_sz

      For example, with 2 ranks:
          rank 0 gets: [1, 2] * 2 = [2, 4] (first row)
          rank 1 gets: [3, 4] * 2 = [6, 8] (second row)
    For example, with 2 ranks:
          rank 0 gets: [1, 2] * 2 = [2, 4] (first row)
          rank 1 gets: [3, 4] * 2 = [6, 8] (second row)

    """
    world_sz = pg.size()
    if world_sz < 2:
        return  # skip trivial

    # Build m0, m1 (each is a list of n rows) fully replicated on all ranks
    m0 = []
    m1 = []
    for r in range(world_sz):
        row0 = torch.arange(
            start=r * world_sz + 1,
            end=r * world_sz + world_sz + 1,
            device=tensor.device,
            dtype=torch.float32,
        )
        row1 = torch.arange(
            start=r * world_sz + 100,
            end=r * world_sz + 100 + world_sz,
            device=tensor.device,
            dtype=torch.float32,
        )
        m0.append(row0)
        m1.append(row1)

    # Each rank receives one "row" for m0, one row for m1, after reduce_scatter_coalesced
    out0 = torch.zeros(world_sz, device=tensor.device, dtype=torch.float32)
    out1 = torch.zeros(world_sz, device=tensor.device, dtype=torch.float32)

    opts = ReduceScatterOptions()
    opts.reduceOp = ReduceOp.SUM

    m0 = torch.stack(m0)
    m1 = torch.stack(m1)

    work = pg.reduce_scatter_tensor_coalesced([out0, out1], [m0, m1], opts)
    work.wait()

    base0 = (
        torch.arange(
            start=rank * world_sz + 1,
            end=rank * world_sz + world_sz + 1,
            device=tensor.device,
            dtype=torch.float32,
        )
        * world_sz
    )
    base1 = (
        torch.arange(
            start=rank * world_sz + 100,
            end=rank * world_sz + 100 + world_sz,
            device=tensor.device,
            dtype=torch.float32,
        )
        * world_sz
    )

    torch.testing.assert_close(out0, base0)
    torch.testing.assert_close(out1, base1)


_COLLECTIVE_TO_FUNC: Dict[str, Callable[[ProcessGroup, int, torch.Tensor], None]] = {
    "allgather": run_allgather_test,
    "allgather_into_tensor_coalesced": run_allgather_into_tensor_coalesced_test,
    "allreduce": run_allreduce_test,
    "allreduce_coalesced": run_allreduce_coalesced_test,
    "alltoall_base": run_alltoall_test,
    "barrier": run_barrier_test,
    "broadcast": run_broadcast_test,
    "broadcast_one": run_broadcast_one_test,
    "reduce_scatter": run_reduce_scatter_test,
    "reduce_scatter_tensor_coalesced": run_reduce_scatter_tensor_coalesced_test,
    "send/recv": run_send_recv_test,
}
_ALL_COLLECTIVES: List[str] = list(_COLLECTIVE_TO_FUNC.keys())


class ProcessGroupTest(TestCase):
    def test_gloo_apis(self) -> None:
        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )

        store_addr = f"localhost:{store.port}/prefix"
        pg = ProcessGroupGloo()
        pg.configure(store_addr, 0, 1)

        self.assertEqual(pg.size(), 1)

        _test_pg(pg)

        m = nn.Linear(3, 4)
        m = torch.nn.parallel.DistributedDataParallel(m, process_group=pg)
        m(torch.rand(2, 3))

    def test_gloo_timeout(self) -> None:
        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )

        store_addr = f"localhost:{store.port}/prefix"
        pg = ProcessGroupGloo(timeout=timedelta(seconds=0.01))
        with self.assertRaisesRegex(
            RuntimeError, "(timeout after 10ms|Socket Timeout)"
        ):
            pg.configure(store_addr, 0, 2)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_nccl_apis(self) -> None:
        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        device = "cuda"

        store_addr = f"localhost:{store.port}/prefix"
        pg = ProcessGroupNCCL()
        pg.configure(store_addr, 0, 1)

        self.assertEqual(pg.size(), 1)

        _test_pg(pg, torch.tensor([2], device=device))

        m = nn.Linear(3, 4).to(device)
        m = torch.nn.parallel.DistributedDataParallel(m, process_group=pg)
        m(torch.rand(2, 3, device=device))

        # reconfigure
        store_addr = f"localhost:{store.port}/prefix2"
        pg.configure(store_addr, 0, 1)

        _test_pg(pg, torch.tensor([2], device=device))

        torch.cuda.synchronize()

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    @skipUnless(
        torch.cuda.is_available(),
        "needs CUDA",
    )
    def test_nccl_init_timeout(self) -> None:
        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        store_addr = f"localhost:{store.port}/prefix"
        del store

        pg = ProcessGroupNCCL(timeout=timedelta(seconds=0.01))

        with self.assertRaisesRegex(RuntimeError, "timed out after 10ms"):
            pg.configure(store_addr, 0, 2)

    def test_baby_gloo_timeout(self) -> None:
        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )

        store_addr = f"localhost:{store.port}/prefix"

        a = ProcessGroupBabyGloo(timeout=timedelta(seconds=0.01))
        with self.assertRaisesRegex(TimeoutError, "timed out after 0.01 seconds"):
            a.configure(store_addr, 0, 2)

    def test_reconfigure_baby_process_group(self) -> None:
        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        store_addr = f"localhost:{store.port}/prefix"

        a = ProcessGroupBabyGloo()
        a.configure(store_addr, 0, 1)
        future_thread_1 = a._future_thread
        future_pipe_1 = a._future_pipe
        p_1 = a._p

        store_addr = f"localhost:{store.port}/prefix2"
        a.configure(store_addr, 0, 1)
        future_thread_2 = a._future_thread
        future_pipe_2 = a._future_pipe
        p_2 = a._p

        self.assertNotEqual(future_thread_1, future_thread_2)
        self.assertNotEqual(future_pipe_1, future_pipe_2)
        self.assertNotEqual(p_1, p_2)

        assert future_thread_1 is not None
        self.assertFalse(future_thread_1.is_alive())
        assert future_pipe_1 is not None
        self.assertTrue(future_pipe_1.closed())
        assert p_1 is not None
        self.assertFalse(p_1.is_alive())

        assert future_thread_2 is not None
        self.assertTrue(future_thread_2.is_alive())
        assert future_pipe_2 is not None
        self.assertFalse(future_pipe_2.closed())
        assert p_2 is not None
        self.assertTrue(p_2.is_alive())

    def test_baby_gloo_apis(self) -> None:
        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )

        store_addr = f"localhost:{store.port}/prefix"

        a = ProcessGroupBabyGloo(timeout=timedelta(seconds=10))
        try:
            a.configure(store_addr, 0, 1)

            _test_pg(a)

            # force collection to ensure no BabyWork objects remain
            gc.collect()

            self.assertEqual(a.num_active_work(), 0)

        finally:
            a.shutdown()

        t = torch.zeros(10)
        with self.assertRaisesRegex(OSError, "handle is closed"):
            a.allreduce([t], AllreduceOptions()).wait()

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_baby_nccl_apis(self) -> None:
        # set to 1 if more than >=2 gpus
        device_id = 1 % torch.cuda.device_count()
        torch.cuda.set_device(device_id)

        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )

        store_addr = f"localhost:{store.port}/prefix"

        a = ProcessGroupBabyNCCL(timeout=timedelta(seconds=10))
        try:
            a.configure(store_addr, 0, 1)

            _test_pg(a, torch.randn((2, 3), device="cuda"))

            torch.cuda.synchronize()

            # force collection to ensure no BabyWork objects remain
            gc.collect()

            self.assertEqual(a.num_active_work(), 0)
        finally:
            a.shutdown()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        t = torch.zeros(10)
        with self.assertRaisesRegex(OSError, "handle is closed"):
            a.allreduce([t], AllreduceOptions()).wait()

    def test_dummy(self) -> None:
        pg = ProcessGroupDummy(0, 1)
        m = nn.Linear(3, 4)
        m = torch.nn.parallel.DistributedDataParallel(m, process_group=pg)
        m(torch.rand(2, 3))

    def test_device_mesh(self) -> None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(0)
        os.environ["RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(1)

        mesh_1d = init_device_mesh("cpu", mesh_shape=(1,), mesh_dim_names=("tp",))

        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        store_addr = f"localhost:{store.port}/prefix"

        pg = ProcessGroupGloo()
        pg.register("test_device_mesh")
        pg.configure(store_addr, 0, 1)

        mesh_2d = extend_device_mesh(mesh_1d, pg)
        mesh_2d.get_group("dp")
        assert mesh_2d.ndim == 2

        pg.unregister()

    def test_functional_collectives(self) -> None:
        dummy_init_pg()

        store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        store_addr = f"localhost:{store.port}/prefix"

        pg = ProcessGroupGloo().register("test_func_col")
        pg.configure(store_addr, 0, 1)

        self.assertEqual(pg.group_name, str(dist.get_pg_count() - 1))

        self.assertIs(_resolve_process_group(pg.group_name), pg)

        try:
            t = torch.zeros(10)
            _functional_collectives.all_reduce(t, "sum", pg).wait()
        finally:
            pg.unregister()

    def test_process_group_wrapper(self) -> None:
        pg = ProcessGroupDummy(0, 1)
        wrapper = ProcessGroupWrapper(pg=pg)
        self.assertIs(wrapper.parent, pg)

        wrapper.configure("addr", 0, 1)
        self.assertEqual(pg.configure_count, 1)

        self.assertEqual(repr(wrapper), "ProcessGroupWrapper(pg=ProcessGroupDummy())")

    def test_error_swallowing_process_group_wrapper(self) -> None:
        pg = ProcessGroupDummy(0, 1)
        wrapper = ErrorSwallowingProcessGroupWrapper(pg)
        self.assertIs(wrapper.parent, pg)

        works = _test_pg(wrapper)
        self.assertIsInstance(list(works.values())[0], _ErrorSwallowingWork)

        err = RuntimeError("test")
        wrapper.report_error(err)
        self.assertEqual(wrapper.error(), err)

        works = _test_pg(wrapper)
        for work in works.values():
            self.assertIsInstance(work, _DummyWork)

    def test_managed_process_group(self) -> None:
        manager = Mock(spec=Manager)
        manager.errored.return_value = None
        manager._pg = ProcessGroupDummy(0, 1)
        pg = ManagedProcessGroup(manager)
        manager.num_participants.return_value = 123

        self.assertEqual(pg.size(), 123)

        works = _test_pg(pg)
        self.assertIsInstance(list(works.values())[0], _ManagedWork)

        self.assertEqual(manager.report_error.call_count, 0)
        self.assertEqual(manager.wrap_future.call_count, 2)
        self.assertEqual(manager.wait_quorum.call_count, 2)


class DeviceMeshTest(TestCase):
    @staticmethod
    def _test_init_device_mesh(world_size: int, rank: int) -> None:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(12346)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(4)

        testcase = TestCase()

        manager = Mock(spec=Manager)
        # Even though we only have 4 workers, we can still initialize (2, 4) mesh.
        # That's because the replicate group is NOT phystically created in the
        # real mesh but is virtually added to the mesh via ManagedDeviceMesh.
        device_mesh = ft_init_device_mesh(
            device_type="cpu",
            mesh_shape=(2, world_size),
            mesh_dim_names=("dp_replicate", "dp_shard"),
            replicate_dim=0,
            manager=manager,
        )

        testcase.assertTrue(
            isinstance(device_mesh.get_group("dp_replicate"), ManagedProcessGroup)
        )
        testcase.assertTrue(
            not isinstance(device_mesh.get_group("dp_shard"), ManagedProcessGroup)
        )
        replicate_group = device_mesh.get_group("dp_replicate")
        testcase.assertEqual(
            cast(ManagedProcessGroup, replicate_group)._manager, manager
        )
        replicate_mesh = device_mesh["dp_replicate"]
        testcase.assertEqual(replicate_mesh.get_group(), replicate_group)
        flatten_mesh = device_mesh._flatten("dp")
        manager.num_participants.return_value = 1
        testcase.assertEqual(flatten_mesh.size(), world_size)
        testcase.assertEqual(flatten_mesh.get_local_rank(), dist.get_rank())

    def test_init_device_mesh(self) -> None:
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(4):
                future = executor.submit(self._test_init_device_mesh, 4, i)
                futures.append(future)


class MultiPgBaseTest(TestCase):
    """
    A base test that creates N processes (via ThreadPoolExecutor) sharing
    a single ProcessGroup. Each test_* method will reuse the same PG.

    Subclasses can specify:
    - BACKEND: the backend to use for the ProcessGroup ("gloo" or "nccl")
    - WORLD_SIZE: how many ranks to simulate
    - Additional config for the PG, i.e. timeouts.
    """

    BACKEND = "gloo"
    WORLD_SIZE = 2

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.store = TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        cls.store_addr = f"localhost:{cls.store.port}/prefix"

        cls.pg_pool: List[ProcessGroup] = []

        cls.executor = ThreadPoolExecutor(max_workers=cls.WORLD_SIZE)

        def init_pg(rank: int) -> ProcessGroup:
            pg = cls._create_pg(cls.BACKEND)
            pg.configure(cls.store_addr, rank, cls.WORLD_SIZE)
            return pg

        futures = [cls.executor.submit(init_pg, rank) for rank in range(cls.WORLD_SIZE)]
        cls.pg_pool = [future.result() for future in futures]

    @classmethod
    def tearDownClass(cls) -> None:
        # Cleanup
        for pg in cls.pg_pool:
            shutdown = getattr(pg, "shutdown", None)
            if shutdown is not None:
                shutdown()
        cls.executor.shutdown(wait=True)
        super().tearDownClass()

    @classmethod
    def _create_pg(cls, backend: str) -> ProcessGroup:
        """
        Helper that creates a new ProcessGroup of the specified type.

        NCCL groups aren't currently supported - we prefer to test
        BabyNCCLGroups as they spin up their own subprocesses.
        """
        if backend == "gloo":
            return ProcessGroupGloo(timeout=timedelta(seconds=1))
        elif backend == "baby_gloo":
            return ProcessGroupBabyGloo(timeout=timedelta(seconds=10))
        elif backend == "nccl":
            return ProcessGroupNCCL(timeout=timedelta(seconds=1))
        elif backend == "baby_nccl":
            return ProcessGroupBabyNCCL(timeout=timedelta(seconds=10))
        elif backend == "dummy":
            return ProcessGroupDummy(0, 1)
        else:
            raise NotImplementedError(f"Unsupported backend: {backend}")

    def _run_parallel(self, collective: str, device: str = "cpu") -> None:
        """
        Helper to run on all ranks in parallel, returning a list
        of results or raising an exception if any fail.
        """
        func = _COLLECTIVE_TO_FUNC[collective]

        futures = []
        for rank in range(self.WORLD_SIZE):
            pg = self.pg_pool[rank]
            # Each worker calls `func(pg=pg, rank=rank, tensor=tensor, *args, **kwargs)`
            if "cuda" in device:
                device = f"cuda:{rank}"
            tensor = torch.tensor([rank + 1], device=device)
            fut = self.executor.submit(func, pg, rank, tensor)
            futures.append(fut)

        self._collect(futures)

    def _collect(self, futs: list[Future]) -> None:
        for i, f in enumerate(futs):
            try:
                res = f.result()  # timeout=10)
                if res:
                    print(f"Rank {i}: {res}")
            except Exception as e:
                print(f"Rank {i}: {e}")
                raise

    def _run_with_resiliency(self, collective: str, device: str = "cpu") -> None:
        """
        Run a collective with resiliency:
        - fault_rank (last rank) simulates a crash.
        - surviving ranks detect the error, then reconfigure PG to exclude fault_rank.
        - surviving ranks run the same collective again successfully.
        """

        def worker(pg: ProcessGroup, rank: int, dev: str) -> str:
            if dev == "cuda":
                torch.cuda.set_device(rank)
                # Use a separate stream to avoid deadlocks between threads.
                torch.cuda.set_stream(torch.cuda.Stream())

            fault_rank = self.WORLD_SIZE - 1
            test = _COLLECTIVE_TO_FUNC[collective]

            # Re-configure the PG to exclude the fault rank
            new_store_addr = f"localhost:{self.store.port}/reconfig_{collective}"

            pg.configure(new_store_addr, rank, self.WORLD_SIZE)

            # run the same collective again successfully
            t2 = torch.tensor([rank + 1], device=dev)
            test(pg, rank, t2)

            # Simulate a failure

            t1 = torch.tensor([rank + 1], device=dev)
            # Simulate failure on the fault rank, but other ranks should still succeed.
            if rank == fault_rank:
                pg.shutdown()
                return f"Rank{rank} crashed"

            # We hardcode the list of expected errors.
            # gloo: Connection closed by peer, timed out waiting, no error, read error
            # nccl: Tensor-likes are not equal/not close (due to abort)
            with self.assertRaisesRegex(
                Exception,
                r"(Connection closed by peer|Timed out waiting|no error|Read error|not equal|not close)",
            ):
                test(pg, rank, t1.clone())
                raise RuntimeError("no error")

            if err := pg.errored():
                with self.assertRaisesRegex(RuntimeError, "aborted"):
                    raise err

            return f"Rank{rank} final success."

        # run in parallel
        futs = [
            self.executor.submit(worker, self.pg_pool[r], r, device)
            for r in range(self.WORLD_SIZE)
        ]
        self._collect(futs)


class GlooMultiPgTest(MultiPgBaseTest):
    BACKEND = "gloo"
    WORLD_SIZE = 3
    SKIP = [
        "alltoall_base",
        "reduce_scatter",
        "reduce_scatter_tensor_coalesced",
    ]
    COLLECTIVES: List[str] = list(set(_ALL_COLLECTIVES) - set(SKIP))

    @parameterized.expand(COLLECTIVES)
    def test_collective(self, collective: str) -> None:
        self._run_parallel(collective, device="cpu")

    # pyre-fixme[56]: Pyre was not able to infer the type of the decorator
    @skipUnless(
        torch.__version__ >= "2.7",
        "torch 2.6 has a bug with destructing PyWork objects",
    )
    @parameterized.expand(COLLECTIVES)
    def test_collective_with_resiliency(self, collective: str) -> None:
        self._run_with_resiliency(collective, device="cpu")


class BabyGlooMultiPgTest(MultiPgBaseTest):
    BACKEND = "baby_gloo"
    WORLD_SIZE = 3
    SKIP = [
        "alltoall_base",
        "reduce_scatter",
        "reduce_scatter_tensor_coalesced",
    ]
    COLLECTIVES: List[str] = list(set(_ALL_COLLECTIVES) - set(SKIP))

    @parameterized.expand(COLLECTIVES)
    def test_collective(self, collective: str) -> None:
        self._run_parallel(collective, device="cpu")

    @parameterized.expand(COLLECTIVES)
    def test_collective_with_resiliency(self, collective: str) -> None:
        self._run_with_resiliency(collective, device="cpu")


@skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() >= 2, "needs 2 CUDA devices"
)
class BabyNcclMultiPgTest(MultiPgBaseTest):
    BACKEND = "baby_nccl"
    WORLD_SIZE = 2

    @parameterized.expand(_ALL_COLLECTIVES)
    def test_collective(self, collective: str) -> None:
        self._run_parallel(collective, device="cuda")

    # @parameterized.expand(_ALL_COLLECTIVES)
    # def test_collective_with_resiliency(self, collective: str) -> None:
    #    self._run_with_resiliency(collective, device="cuda")


@skipUnless(
    torch.cuda.is_available()
    and torch.cuda.device_count() >= 2
    and torch.cuda.nccl.version() >= (2, 25),
    "needs 2 CUDA devices and NCCL >=2.25",
)
class NormalNcclMultiPgTest(MultiPgBaseTest):
    BACKEND = "nccl"
    WORLD_SIZE = 2

    @parameterized.expand(_ALL_COLLECTIVES)
    def test_collective(self, collective: str) -> None:
        self._run_parallel(collective, device="cuda")

    @parameterized.expand(_ALL_COLLECTIVES)
    def test_collective_with_resiliency(self, collective: str) -> None:
        self._run_with_resiliency(collective, device="cuda")
