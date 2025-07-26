# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from datetime import timedelta

REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))
os.environ["CUDA_VISIBLE_DEVICES"] = str(REPLICA_GROUP_ID % 4)
os.environ["NCCL_HOSTID"] = str(REPLICA_GROUP_ID)

USE_STREAMING = os.getenv("USE_STREAMING", "False") == "True"
USE_NCCL = os.getenv("USE_NCCL", "False") == "True"

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.pipelining import SplitPoint, pipeline
from torch.export import export
from torch.utils.tensorboard import SummaryWriter
from torchdata.stateful_dataloader import StatefulDataLoader

from torchft import (
    DistributedSampler,
    Manager,
    ProcessGroupBabyNCCL,
    ProcessGroupGloo,
    ProcessGroupNCCL,
)
from torchft.checkpointing.http_transport import HTTPTransport
from torchft.local_sgd import DiLoCo

logging.basicConfig(level=logging.INFO)


@record
def main() -> None:
    REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))
    RUN = int(os.environ.get("RUN", 0))

    output_folder = f"output/replica-{REPLICA_GROUP_ID}"

    writer = SummaryWriter(f"{output_folder}/tensorboard", max_queue=1000)

    def load_state_dict(state_dict):
        m.load_state_dict(state_dict["model"])
        inner_optimizer.load_state_dict(state_dict["inner_optim"])
        outer_optimizer.load_state_dict(state_dict["outer_optim"])

    def state_dict():
        return {
            "model": m.state_dict(),
            "inner_optim": inner_optimizer.state_dict(),
            "outer_optim": outer_optimizer.state_dict(),
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pg = (
        ProcessGroupNCCL(
            timeout=timedelta(seconds=10),
        )
        if torch.cuda.is_available() and USE_NCCL
        else ProcessGroupGloo(timeout=timedelta(seconds=10))
    )

    transport = HTTPTransport(
        timeout=timedelta(seconds=10),
        num_chunks=0,
    )

    manager = Manager(
        pg=pg,
        use_async_quorum=False,
        min_replica_size=1,
        load_state_dict=load_state_dict,
        state_dict=state_dict,
        replica_id=f"train_diloco_{REPLICA_GROUP_ID}",
        timeout=timedelta(seconds=30),
        checkpoint_transport=transport,
    )

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

            target_size = 500 * (1 << 20)
            self.useless = nn.Embedding(target_size // d_hid // 4, d_hid)

        def forward(self, x):
            x = self.net1(x)
            x = self.relu(x)
            x = self.net2(x)
            x += self.useless.weight[0]
            return x

    class MultiMLP(torch.nn.Module):
        def __init__(self, d_hid: int, n_layers: int = 2, num_classes: int = 10):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [MLPModule(d_hid) for _ in range(n_layers)]
            )
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

    n_layers = 2
    d_hid = 128

    m = MultiMLP(d_hid, n_layers).to(device)

    trainset = DummyDataset(size=10000, feature_dim=d_hid)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, num_workers=2, shuffle=True
    )

    example_input, _ = next(iter(trainloader))

    pipe = pipeline(
        module=m, mb_args=(example_input.to(device),), split_spec=m.split_spec
    )
    module_partitions = [pipe.get_stage_module(idx) for idx in range(n_layers)]

    inner_optimizer: optim.Optimizer = torch.optim.AdamW(
        m.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
    )
    outer_optimizer: optim.Optimizer = torch.optim.SGD(
        m.parameters(), lr=0.7, momentum=0.9, nesterov=True
    )

    criterion = nn.CrossEntropyLoss()

    num_params = sum(p.numel() for p in m.parameters())
    print(f"Total number of parameters: {num_params}")

    def trace_handler(p):
        dir = f"{output_folder}/profiles"
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

        p.export_chrome_trace(f"{dir}/step-{p.step_num}.json")

    # You can use an epoch based training but with faults it's easier to use step
    # based training.
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=60, repeat=2),
        on_trace_ready=trace_handler,
        record_shapes=False,
        profile_memory=False,
    )

    prof.start()
    tensorboard_key_prefix = f"Run:{RUN}"
    with DiLoCo(
        manager,
        module_partitions if USE_STREAMING else [m],
        inner_optimizer,
        outer_optimizer,
        backup_device=device,
        sync_every=20 if USE_STREAMING else 20,
        fragment_sync_delay=5 if USE_STREAMING else 0,
        should_quantize=False,
    ) as diloco:
        while True:
            for i, (inputs, labels) in enumerate(trainloader):
                prof.step()

                inputs = inputs.to(device)
                labels = labels.to(device)

                inner_optimizer.zero_grad()

                out = m(inputs)
                loss = criterion(out, labels)

                writer.add_scalar(f"{tensorboard_key_prefix}/loss", loss, i)

                loss.backward()

                inner_optimizer.step()

                writer.add_scalar(
                    f"{tensorboard_key_prefix}/num_participants",
                    manager.num_participants(),
                    i,
                )
                writer.add_scalar(
                    f"{tensorboard_key_prefix}/current_step", manager.current_step(), i
                )
                if manager.current_step() % 100 == 0:
                    print(f"[{manager.current_step()}] loss = {loss.item()}")

                if manager.current_step() >= 15:
                    # complete training
                    prof.stop()
                    writer.flush()
                    exit()


if __name__ == "__main__":
    main()
