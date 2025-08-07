# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import types
import unittest
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, cast

# Define a type variable for the Future's value type
T = TypeVar("T")

import parameterized
import torch
from torch.distributed.distributed_c10d import Work
from torch.futures import Future

from torchft.manager import Manager, _ManagedWork


class SimpleWork(Work):
    """A simple implementation of torch.distributed.Work for testing."""

    def __init__(self, tensors: List[torch.Tensor]) -> None:
        super().__init__()
        self._tensors = tensors
        self._future: Future[List[torch.Tensor]] = torch.futures.Future()
        self._is_completed: bool = False

    def wait(self, timeout: Optional[timedelta] = None) -> bool:
        self._is_completed = True
        self._future.set_result(self._tensors)
        return True

    def get_future(self) -> Future[List[torch.Tensor]]:
        return self._future


class TestManagedWork(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("cpu", torch.device("cpu")),
            ("cuda", torch.device("cuda:0")),
        ]
    )
    def test_callbacks_execute_after_wait(
        self, name: str, device: torch.device
    ) -> None:
        """Test that callbacks are only executed after wait() is called."""
        # Skip if CUDA is requested but not available
        if device.type == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Create a tensor to work with
        tensor: torch.Tensor = torch.ones(1, dtype=torch.float32, device=device)

        # Create a simple work object
        work = SimpleWork([tensor])

        # Create a minimal manager object with just the wrap_future method
        manager = Manager.__new__(Manager)  # Create instance without calling __init__
        # We're using types.MethodType to attach a method to the manager instance
        # This is just for testing purposes
        manager.wrap_future = types.MethodType(  # type: ignore
            lambda self, fut, default, timeout=None: fut, manager
        )

        # Create the managed work
        managed_work = _ManagedWork(manager, work, [tensor])

        # Track callback execution
        callback_executed: bool = False

        def callback(fut: Future[object]) -> List[torch.Tensor]:
            # Cast to the expected type
            nonlocal callback_executed, tensor
            callback_executed = True
            # Multiply tensor by 2 to verify the callback ran
            tensor.mul_(2)
            return [tensor]

        # Add the callback
        fut = managed_work.get_future()
        fut = fut.then(callback)

        # Verify callback hasn't executed yet
        self.assertFalse(callback_executed)
        self.assertEqual(tensor.item(), 1.0)

        # Call wait() which should trigger the callback
        managed_work.wait()

        # Verify callback has executed
        self.assertTrue(callback_executed)
        self.assertEqual(tensor.item(), 2.0)

    @parameterized.parameterized.expand(
        [
            ("cpu", torch.device("cpu")),
            ("cuda", torch.device("cuda:0")),
        ]
    )
    def test_multiple_callbacks_execute_in_order(
        self, name: str, device: torch.device
    ) -> None:
        """Test that multiple callbacks are executed in the order they were added."""
        # Skip if CUDA is requested but not available
        if device.type == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Create a tensor to work with
        tensor: torch.Tensor = torch.ones(1, dtype=torch.float32, device=device)

        # Create a simple work object
        work = SimpleWork([tensor])

        # Create a minimal manager object with just the wrap_future method
        manager = Manager.__new__(Manager)  # Create instance without calling __init__
        manager.wrap_future = types.MethodType(  # type: ignore
            lambda self, fut, default, timeout=None: fut, manager
        )

        # Create the managed work
        managed_work = _ManagedWork(manager, work, [tensor])

        # Track execution order
        execution_order: List[int] = []

        def callback1(fut: Future[list[torch.Tensor]]) -> List[torch.Tensor]:
            nonlocal tensor
            execution_order.append(1)
            tensor.add_(1)
            return [tensor]

        def callback2(fut: Future[list[torch.Tensor]]) -> List[torch.Tensor]:
            nonlocal tensor
            execution_order.append(2)
            tensor.add_(2)
            return [tensor]

        def callback3(fut: Future[list[torch.Tensor]]) -> List[torch.Tensor]:
            nonlocal tensor
            execution_order.append(3)
            tensor.add_(3)
            return [tensor]

        # Add callbacks
        fut = managed_work.get_future()
        fut = cast(Future[list[torch.Tensor]], fut)
        fut = fut.then(callback1)
        fut = fut.then(callback2)
        fut = fut.then(callback3)

        # Verify no callbacks have executed yet
        self.assertEqual(len(execution_order), 0)
        self.assertEqual(tensor.item(), 1.0)

        # Call wait() which should trigger the callbacks
        managed_work.wait()

        # Verify callbacks executed in order
        self.assertEqual(execution_order, [1, 2, 3])

        # Each callback adds to the tensor, so final value should be 1 + 1 + 2 + 3 = 7
        self.assertEqual(tensor.item(), 7.0)

    @parameterized.parameterized.expand(
        [
            ("cpu", torch.device("cpu")),
            ("cuda", torch.device("cuda:0")),
        ]
    )
    def test_future_then_api(self, name: str, device: torch.device) -> None:
        """Test that the future's then API works correctly with ManagedWork."""
        # Skip if CUDA is requested but not available
        if device.type == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Create a tensor to work with
        tensor: torch.Tensor = torch.ones(1, dtype=torch.float32, device=device)

        # Create a simple work object
        work = SimpleWork([tensor])

        # Create a minimal manager object with just the wrap_future method
        manager = Manager.__new__(Manager)  # Create instance without calling __init__
        manager.wrap_future = types.MethodType(  # type: ignore
            lambda self, fut, default, timeout=None: fut, manager
        )

        # Create the managed work
        managed_work = _ManagedWork(manager, work, [tensor])

        # Get the future
        future = managed_work.get_future()

        # Track callback execution
        callback_executed: bool = False

        def callback(fut: Future[object]) -> List[torch.Tensor]:
            # Cast to the expected type
            nonlocal callback_executed, tensor
            callback_executed = True
            # Multiply tensor by 3 to verify the callback ran
            tensor.mul_(3)
            return [tensor]

        # Use the then API
        future = future.then(callback)

        # Verify callback hasn't executed yet
        self.assertFalse(callback_executed)
        self.assertEqual(tensor.item(), 1.0)

        # Call wait() on the managed_work first to set up the future properly
        managed_work.wait()

        # Verify callback has executed
        self.assertTrue(callback_executed)
        self.assertEqual(tensor.item(), 3.0)

    @parameterized.parameterized.expand(
        [
            ("cpu", torch.device("cpu")),
            ("cuda", torch.device("cuda:0")),
        ]
    )
    def test_callbacks_changing_return_types(
        self, name: str, device: torch.device
    ) -> None:
        """
        Test that callbacks can change return types and that tensors are modified in-place.
        This test demonstrates:
        1. Callbacks changing return types (List[Tensor] -> Dict -> Tuple)
        2. Using Future.value() instead of nonlocal
        3. Verifying tensors are modified in-place for both approaches
        """
        # Skip if CUDA is requested but not available
        if device.type == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Create tensors to work with
        tensor1: torch.Tensor = torch.ones(1, dtype=torch.float32, device=device)
        tensor2: torch.Tensor = torch.ones(1, dtype=torch.float32, device=device) * 2

        # Store original tensor memory addresses to verify in-place modification
        tensor1_address = tensor1.data_ptr()
        tensor2_address = tensor2.data_ptr()

        # Create a simple work object
        work = SimpleWork([tensor1, tensor2])

        # Create a minimal manager object with just the wrap_future method
        manager = Manager.__new__(Manager)  # Create instance without calling __init__
        manager.wrap_future = types.MethodType(  # type: ignore
            lambda self, fut, default, timeout=None: fut, manager
        )

        # Create the managed work
        managed_work = _ManagedWork(manager, work, [tensor1, tensor2])

        # Get the future
        future = managed_work.get_future()
        future = cast(Future[List[torch.Tensor]], future)

        # First callback: Takes List[Tensor] and returns Dict[str, Tensor]
        # Uses nonlocal to modify tensor1
        def callback1(fut: Future[List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
            tensors = fut.value()
            nonlocal tensor1
            # Modify tensor1 in-place using nonlocal
            tensor1.mul_(3)
            # Return a dictionary instead of a list
            return {"first": tensors[0], "second": tensors[1]}

        # Second callback: Takes Dict[str, Tensor] and returns Tuple[Tensor, float]
        # Uses Future.value() to modify tensor2
        def callback2(
            fut: Future[Dict[str, torch.Tensor]]
        ) -> Tuple[torch.Tensor, float]:
            data = fut.value()
            # Modify tensor2 in-place using the value from the future
            data["second"].add_(5)  # Should modify tensor2 in-place
            # Return a tuple instead of a dict
            return (data["second"], data["first"].item())

        # Third callback: Takes Tuple[Tensor, float] and returns a single Tensor
        def callback3(fut: Future[Tuple[torch.Tensor, float]]) -> torch.Tensor:
            tensor, value = fut.value()
            # Create a new tensor based on the tuple values
            result = tensor * value
            return result

        # Chain the callbacks
        future = future.then(callback1)
        future = future.then(callback2)
        future = future.then(callback3)

        # Call wait() to trigger the callbacks
        managed_work.wait()

        # Verify tensor1 was modified in-place (using nonlocal)
        self.assertEqual(tensor1.item(), 3.0)  # 1 * 3 = 3
        self.assertEqual(tensor1.data_ptr(), tensor1_address)  # Same memory address

        # Verify tensor2 was modified in-place (using Future.value())
        self.assertEqual(tensor2.item(), 7.0)  # 2 + 5 = 7
        self.assertEqual(tensor2.data_ptr(), tensor2_address)  # Same memory address

        # Get the final result from the future
        final_result = future.wait()

        # The final result should be tensor2 * tensor1.item() = 7 * 3 = 21
        self.assertEqual(final_result.item(), 21.0)


if __name__ == "__main__":
    unittest.main()
