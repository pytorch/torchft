# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from datetime import timedelta
import torch
import torch.utils.data
from torch import nn, optim
from torch.distributed import ReduceOp
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.pipelining import SplitPoint, pipeline

from torchft import Manager, ProcessGroupGloo, ProcessGroupNCCL
from torchft.checkpointing.pg_transport import PGTransport
from torchft.local_sgd import DiLoCo

from torchft.collectives import allreduce_quantized

logging.basicConfig(level=logging.INFO)

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=10000, feature_dim=128, num_classes=10):
        """
        Create a dummy dataset suitable for MLP models.
        
        Args:
            size: Number of samples in the dataset
            feature_dim: Dimension of the feature vector (should match d_hid in MultiMLP)
            num_classes: Number of output classes
        """
        self.size = size
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Generate random feature vector (1D) instead of image (3D)
        features = torch.rand(self.feature_dim)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return features, label

# MLP Layer
class MLPModule(torch.nn.Module):
    def __init__(self, d_hid: int):
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x

class MultiMLP(torch.nn.Module):
    def __init__(self, d_hid: int, n_layers: int = 2, num_classes: int = 10):
        super().__init__()
        self.layers = torch.nn.ModuleList([MLPModule(d_hid) for _ in range(n_layers)])
        # Add a final classification layer
        self.classifier = torch.nn.Linear(d_hid, num_classes)
        # For demonstration purposes only, this should be defined by user
        self.split_spec = {
            f"layers.{i}": SplitPoint.BEGINNING for i in range(1, n_layers)
        }

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # Apply the classification layer to get logits
        x = self.classifier(x)
        return x


REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))
NUM_REPLICA_GROUPS = int(os.environ.get("NUM_REPLICA_GROUPS", 2))
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

print(f"{CUDA_VISIBLE_DEVICES=}, REPLICA_GROUP_ID: {REPLICA_GROUP_ID}")
print(f"{NUM_REPLICA_GROUPS=}")
torch.cuda.set_device(0)

# Get number of classes from the dataset
d_hid = 128  # Feature dimension for the MLP
n_layers = 8  # Number of MLP layers

# Create dummy dataset with random data matching the model's input dimension
dataset_size = 10000
trainset = DummyDataset(size=dataset_size, feature_dim=d_hid)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, num_workers=2, shuffle=True
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pg = (
    ProcessGroupNCCL(
        timeout=timedelta(seconds=30),
    )
    if torch.cuda.is_available()
    else ProcessGroupGloo(timeout=timedelta(seconds=5))
)
print(f"{device=} {pg=}")

transport = PGTransport(
    pg,
    timeout=timedelta(seconds=10),
    device=device,
)

num_classes = trainset.num_classes
m = MultiMLP(d_hid=d_hid, n_layers=n_layers, num_classes=num_classes).to(device)
inner_optimizer = optim.AdamW(
    m.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
)
outer_optimizer = optim.SGD(m.parameters(), lr=0.7, momentum=0.9, nesterov=True)
criterion = nn.CrossEntropyLoss()

print(m)
num_params = sum(p.numel() for p in m.parameters())
print(f"DiLoCo: Total number of parameters: {num_params}")

@record
def regular_diloco() -> None:
    def load_state_dict(state_dict):
        m.load_state_dict(state_dict["model"])
        m.to(device)
        diloco.original_parameters = state_dict["original_params"]
        for name in diloco.original_parameters.keys():
            diloco.original_parameters[name] = diloco.original_parameters[name].to(
                device
            )
        inner_optimizer.load_state_dict(state_dict["inner_optim"])
        outer_optimizer.load_state_dict(state_dict["outer_optim"])

    def state_dict():
        return {
            "model": m.state_dict(),
            "original_params": diloco.original_parameters,
            "inner_optim": inner_optimizer.state_dict(),
            "outer_optim": outer_optimizer.state_dict(),
        }


    manager = Manager(
        pg=pg,
        min_replica_size=1,
        load_state_dict=load_state_dict,
        state_dict=state_dict,
        replica_id=f"regular_diloco_{REPLICA_GROUP_ID}",
        timeout=timedelta(seconds=30),
        checkpoint_transport=transport,
        use_async_quorum=False,
    )

    num_local_steps = 0
    sync_every = 100
    max_outer_steps = 10

    with DiLoCo(
        manager,
        m,
        inner_optimizer,
        outer_optimizer,
        backup_device=device,
        sync_every=sync_every,
    ) as diloco:
        while True:
            for i, (inputs, labels) in enumerate(trainloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                inner_optimizer.zero_grad()

                out = m(inputs)
                loss = criterion(out, labels)
                loss.backward()

                inner_optimizer.step()
                num_local_steps += 1

                if num_local_steps % sync_every == 0:
                    print(
                        f"DiLoCo: Number of inner optimizer steps completed: {num_local_steps}"
                    )
                    print(
                        f"DiLoCo: Number of outer optimizer steps completed: {manager.current_step()} loss = {loss.item()}"
                    )

                if manager.current_step() >= max_outer_steps:
                    exit()

@record
def streaming_diloco() -> None:
    def load_state_dict(state_dict):
        m.load_state_dict(state_dict["model"])
        m.to(device)
        inner_optimizer.load_state_dict(state_dict["inner_optim"])
        outer_optimizer.load_state_dict(state_dict["outer_optim"])

    def state_dict():
        return {
            "model": m.state_dict(),
            "inner_optim": inner_optimizer.state_dict(),
            "outer_optim": outer_optimizer.state_dict(),
        }

    manager = Manager(
        pg=pg,
        min_replica_size=1,
        load_state_dict=load_state_dict,
        state_dict=state_dict,
        replica_id=f"streaming_diloco_{REPLICA_GROUP_ID}",
        timeout=timedelta(seconds=30),
        checkpoint_transport=transport,
        use_async_quorum=False,
    )

    # Part 1, more easily specify model partitions using pipeline APIs?
    # TODO: how to map partition back to original model
    example_input, _ = next(iter(trainloader))
    pipe = pipeline(module=m, mb_args=(example_input.to(device),), split_spec=m.split_spec)
    module_partitions = [pipe.get_stage_module(idx) for idx in range(n_layers)]
    # for module in module_partitions:
    #     print(f"DiLoCo: {module=}, params: {[p for p in module.parameters()]}")

    # Part 2, run DiLoCo as usual
    num_local_steps = 0
    sync_every = 100
    max_outer_steps = 5

    for i, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        inner_optimizer.zero_grad()

        out = m(inputs)
        loss = criterion(out, labels)
        loss.backward()

        inner_optimizer.step()
        num_local_steps += 1

        if num_local_steps % sync_every == 0:
            print(
                f"DiLoCo: Number of inner optimizer steps completed: {num_local_steps}"
            )
            print(
                f"DiLoCo: Number of outer optimizer steps completed: {manager.current_step()} loss = {loss.item()}"
            )
            manager.start_quorum()
            # On sync step, we need to sync the model weights across the manager (we only do part of it)
            params_data = []
            for p in module_partitions[0].parameters():
                tensor = p.data
                # TODO: only 2D tensors supported for quantization?
                # replica_0/0     File "/data/users/howardhuang/torchft/torchft/quantization.py", line 450, in _prepare_quantize_fp8
                # replica_0/0       assert len(inputs[i].shape) == 2, "Only 2D tensors are supported"
                # replica_0/0   AssertionError: Only 2D tensors are supported
                if tensor.dim() == 1:
                    # Convert 1D tensors to 2D by adding a dimension
                    tensor = tensor.unsqueeze(0)
                params_data.append(tensor)
            # print(f"Transfering {params_data=} tensors")
            print(f"param shapes {[(p.shape) for p in params_data]}")
            # TODO: error blocking
            # replica_1/0     File "/data/users/howardhuang/torchft/torchft/quantization.py", line 531, in fused_quantize_into_fp8
            # replica_1/0       _fused_kernel_quantize_into_fp8[grid](
            # replica_1/0     File "/home/howardhuang/.conda/envs/torchft/lib/python3.10/site-packages/triton/runtime/jit.py", line 499, in run
            # replica_1/0       if key not in self.cache[device]:
            # replica_1/0   TypeError: unhashable type: 'constexpr'
            fut = allreduce_quantized(params_data, ReduceOp.AVG, pg)
            # TODO: add allreduce_quantized as a manager collective option
            fut.wait()
            print("finished")

        if manager.current_step() >= max_outer_steps:
            print("exiting")
            exit()

if __name__ == "__main__":
    # regular_diloco()
    streaming_diloco()
