# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Optimizers
============

This module implements an optimizer wrapper that works with the Manager to provide fault tolerance.

"""

from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

import torch
from torch.optim import Optimizer

if TYPE_CHECKING:
    from torchft.manager import Manager


class OptimizerWrapper(Optimizer):
    """
    This wraps any provided optimizer and in conjunction with the manager will provide fault tolerance.

    zero_grad() must be called at the start of the forwards pass and step() must
    be called at the end of the backwards pass.

    Depending on the state of the manager, the optimizer will either commit the
    gradients to the wrapped optimizer or ignore them.
    """

    def __init__(self, manager: "Manager", optim: Optimizer) -> None:
        self.optim = optim
        self.manager = manager

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        self.optim.add_param_group(param_group)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.optim.load_state_dict(state_dict)

    def state_dict(self) -> Dict[str, Any]:
        return self.optim.state_dict()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.manager.start_quorum()
        self.optim.zero_grad(set_to_none)

    def step(self, closure: Optional[object] = None) -> None:
        assert closure is None, "optimizers that use closures are not supported"
        if self.manager.should_commit():
            self.optim.step()

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        return self.optim.param_groups

    @property
    def state(self) -> Mapping[torch.Tensor, Any]:  # pyre-fixme[3]
        return self.optim.state
