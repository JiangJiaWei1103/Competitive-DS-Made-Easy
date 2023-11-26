"""
Demo model architecture.
Author: JiaWei Jiang
"""
from typing import Dict

import torch.nn as nn
from torch import Tensor


class DemoModel(nn.Module):
    """Demo model architecture.

    Args:
        in_dim: input dimension
        h_dim: hidden dimension
        out_dim: output dimension
    """

    def __init__(self, in_dim: int, h_dim: int, out_dim: int) -> None:
        self.name = self.__class__.__name__
        super().__init__()

        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim

        # Model blocks
        self.net = nn.Sequential(nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, out_dim))

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        """Forward pass.

        Args:
            inputs: model inputs

        Returns:
            output: prediction
        """
        output = self.net(inputs["x"]).squeeze(dim=-1)

        return output
