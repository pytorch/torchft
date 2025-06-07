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

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.distributed.elastic.multiprocessing.errors import record
from torchdata.stateful_dataloader import StatefulDataLoader

from torchft import (
    DistributedDataParallel,
    DistributedSampler,
    Manager,
    Optimizer,
    ProcessGroupGloo,
    ProcessGroupNCCL,
)
from torchft.checkpointing.pg_transport import PGTransport

logging.basicConfig(level=logging.INFO)


@record
def main() -> None:
    REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))
    NUM_REPLICA_GROUPS = int(os.environ.get("NUM_REPLICA_GROUPS", 2))

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./cifar", train=True, download=True, transform=transform
    )

    # This shards the training set across all ranks and replica groups. We manage
    # the dataloaders on a per replica group basis with the assumption that the
    # majority of groups will be available so few batches will be dropped.
    sampler = DistributedSampler(
        trainset,
        replica_rank=REPLICA_GROUP_ID,
        num_replica_groups=NUM_REPLICA_GROUPS,
        group_rank=0,
        # for DDP we can use replica groups of size 1, FSDP/PP/CP would need more.
        num_replicas=1,
        shuffle=True,
    )

    # This uses the torchdata StatefulDataLoader to be able to checkpoint and
    # restore the per worker dataloader position.
    trainloader = StatefulDataLoader(
        trainset, batch_size=64, num_workers=2, sampler=sampler
    )

    def load_state_dict(state_dict):
        m.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optim"])

    def state_dict():
        return {
            "model": m.state_dict(),
            "optim": optimizer.state_dict(),
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pg = (
        ProcessGroupNCCL(
            timeout=timedelta(seconds=30),
        )
        if torch.cuda.is_available()
        else ProcessGroupGloo(timeout=timedelta(seconds=5))
    )

    transport = PGTransport(
        pg,
        timeout=timedelta(seconds=10),
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )

    manager = Manager(
        pg=pg,
        min_replica_size=1,
        load_state_dict=load_state_dict,
        state_dict=state_dict,
        replica_id=f"train_ddp_{REPLICA_GROUP_ID}",
        timeout=timedelta(seconds=30),
        checkpoint_transport=transport,
    )

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 6, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )

            final_dim = 10
            # We add a useless 1GB intermediate layer so we spend more time in dist
            # communication so injected failures are more likely to cause issues
            # if they exist.
            target_size = 1_000_000_000
            self.useless = nn.Embedding(target_size // final_dim // 4, final_dim)

            self.classifier = nn.Sequential(
                nn.Linear(16 * 5 * 5, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, final_dim),
            )

        def forward(self, x):
            x = self.cnn(x)
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = self.classifier(x)
            x += self.useless.weight[0]
            return x

    m = Net().to(device)
    m = DistributedDataParallel(manager, m)
    optimizer = Optimizer(manager, optim.AdamW(m.parameters()))
    criterion = nn.CrossEntropyLoss()

    print(m)
    num_params = sum(p.numel() for p in m.parameters())
    print(f"Total number of parameters: {num_params}")

    sort_by_keyword = "self_" + device + "_time_total"

    def trace_handler(p):
        output = p.key_averages().table(
            sort_by=sort_by_keyword,
            row_limit=100,
        )
        print(output)
        p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

    # You can use an epoch based training but with faults it's easier to use step
    # based training.
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=5, warmup=1, active=10, repeat=2),
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
    )

    prof.start()
    while True:
        for i, (inputs, labels) in enumerate(trainloader):
            prof.step()

            inputs = inputs.to(device)
            labels = labels.to(device)

            # must be called at the beginning of each train loop
            # Quorum computation is triggered here but only needed in the backwards pass.
            optimizer.zero_grad()

            out = m(inputs)
            loss = criterion(out, labels)

            # Gradient allreduce overlaps with the backwards pass.
            loss.backward()

            # must be called at the end of the train loop
            # This may not actually step the optimizer if an error occured during grad allreduce.
            optimizer.step()

            if manager.current_step() % 100 == 0:
                print(f"[{manager.current_step()}] loss = {loss.item()}")

            # TODO (by the user): periodically checkpoint model, optim, manager and dataloader

            # You typically want to checkpoint dataloader frequently (every step?) to
            # avoid repeated batches as it's replica group specific.

            # Model, optim and manager checkpoints can be done more infrequently as
            # they're shared across all groups and will load from existing replicas as
            # long as not every worker goes down.

            if manager.current_step() >= 10000:
                # complete training
                prof.stop()
                exit()


if __name__ == "__main__":
    main()
