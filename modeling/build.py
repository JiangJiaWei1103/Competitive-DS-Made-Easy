"""
Model architecture building logic.
Author: JiaWei Jiang

The file contains a single function for model name switching and model
architecture building based on the model configuration.

To add in new model, users need to design custom model architecture,
put the file under the same directory, and import the corresponding
model below.
"""
from importlib import import_module
from typing import Any

from torch.nn import Module

from .demo.Demo import DemoModel


def build_model(model_name: str, **model_cfg: Any) -> Module:
    """Build and return the specified model architecture.

    Args:
        model_name: name of model architecture
        model_cfg: hyperparameters of the specified model

    Returns:
        model: model instance
    """
    model: Module
    if model_name == "DemoModel":
        model = DemoModel(**model_cfg)
    elif model_name.startswith("Exp"):
        # For quick dev and verification
        model_module = import_module(f"modeling.exp.{model_name}")
        model = model_module.Exp(**model_cfg)
    else:
        raise RuntimeError(f"{model_name} isn't registered.")

    return model
