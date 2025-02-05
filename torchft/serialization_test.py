from io import BytesIO
from typing import cast
from unittest import TestCase

import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh, DTensor, distribute_tensor

from torchft.serialization import streaming_load, streaming_save


class MyClass:
    def __init__(self, a: int) -> None:
        self.a = a

    def __eq__(self, other: "MyClass") -> bool:
        return self.a == other.a


class TestCheckpointingSerialization(TestCase):
    def test_scalar_tensor(self) -> None:
        tensor = torch.tensor(42, dtype=torch.int32)
        state_dict = {"scalar": tensor}
        file = BytesIO()
        streaming_save(state_dict, file)
        file.seek(0)

        result = streaming_load(file)
        torch.testing.assert_close(result, state_dict)

    def test_strided_tensor(self) -> None:
        base_tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        strided_tensor = base_tensor[::2, ::2]
        state_dict = {"strided": strided_tensor}
        file = BytesIO()
        streaming_save(state_dict, file)
        file.seek(0)

        result = streaming_load(file)
        torch.testing.assert_close(result, state_dict)

    def test_tensor_with_offset(self) -> None:
        base_tensor = torch.arange(10, dtype=torch.float64)
        offset_tensor = base_tensor[2:]
        state_dict = {"offset": offset_tensor}
        file = BytesIO()
        streaming_save(state_dict, file)
        file.seek(0)

        result = streaming_load(file)
        torch.testing.assert_close(result, state_dict)

    def test_nested_tensors(self) -> None:
        tensor1 = torch.tensor([1, 2, 3], dtype=torch.int32)
        tensor2 = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float64)
        state_dict = {"nested": {"tensor1": tensor1, "tensor2": tensor2}}
        file = BytesIO()
        streaming_save(state_dict, file)
        file.seek(0)

        result = streaming_load(file)
        torch.testing.assert_close(result, state_dict)

    def test_various_data_types(self) -> None:
        tensor_float32 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        tensor_int16 = torch.tensor([1, 2, 3], dtype=torch.int16)
        tensor_bool = torch.tensor([True, False, True], dtype=torch.bool)
        state_dict = {
            "float32": tensor_float32,
            "int16": tensor_int16,
            "bool": tensor_bool,
        }
        file = BytesIO()
        streaming_save(state_dict, file)
        file.seek(0)

        result = streaming_load(file)
        torch.testing.assert_close(result, state_dict)

    def test_dtensor(self) -> None:
        dist.init_process_group(
            backend="gloo", rank=0, world_size=1, store=dist.HashStore()
        )

        device_mesh = DeviceMesh("cpu", 1)
        tensor = torch.randn(4, 4, device="cuda")
        dtensor = distribute_tensor(tensor, device_mesh, [])
        state_dict = dtensor
        file = BytesIO()
        streaming_save(state_dict, file)
        file.seek(0)

        result = cast(DTensor, streaming_load(file))
        torch.testing.assert_close(result.to_local(), state_dict.to_local())

    def test_python_object(self) -> None:
        state_dict = {
            "obj": MyClass(42),
        }

        file = BytesIO()
        streaming_save(state_dict, file)
        file.seek(0)

        result = streaming_load(file)
        self.assertEqual(result, state_dict)
