"""
Custom criterions.
Author: JiaWei Jiang
"""
from torch.nn.modules.loss import _Loss


class CustomLoss(_Loss):
    """Custom loss criterion."""

    def __init__(self) -> None:
        pass
