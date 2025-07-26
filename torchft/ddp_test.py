# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase
from unittest.mock import create_autospec

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.distributed_c10d import Work
from torch.futures import Future

from torchft.ddp import DistributedDataParallel, PureDistributedDataParallel
from torchft.manager import Manager
from torchft.process_group import ProcessGroupBabyGloo, ProcessGroupGloo
from torchft.work import _DummyWork


class TestDDP(TestCase):
    def test_pure_ddp(self) -> None:
        manager = create_autospec(Manager)

        m = nn.Linear(3, 4)
        m = PureDistributedDataParallel(manager, m)

        inp = torch.rand(2, 3)
        out = m(inp)
        loss = out.mean()
        loss.backward()

        for p in m.parameters():
            self.assertIsNotNone(p.grad)

        self.assertEqual(manager.allreduce.call_count, len(list(m.parameters())))

    def test_ddp(self) -> None:
        manager = create_autospec(Manager)

        call_count = 0

        def allreduce(
            tensor: torch.Tensor,
        ) -> Work:
            nonlocal call_count

            call_count += 1

            return _DummyWork(tensor)

        manager.allreduce = allreduce

        m = nn.Linear(3, 4)
        m = DistributedDataParallel(manager, m)

        inp = torch.rand(2, 3)
        out = m(inp)
        loss = out.mean()
        loss.backward()

        for p in m.parameters():
            self.assertIsNotNone(p.grad)

        self.assertGreaterEqual(call_count, 1)
