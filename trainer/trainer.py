"""
Custom trainer definitions for different training processes.
Author: JiaWei Jiang

This file contains diversified trainers, whose training logics are
inherited from `BaseTrainer`.

* [ ] Fuse grad clipping mechanism into optimizer.
* [ ] Grad accumulation with drop_last or updating with the remaining
    samples.
* [ ] Support AMP.
"""
import gc
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from omegaconf.dictconfig import DictConfig
from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from base.base_trainer import BaseTrainer
from evaluating.evaluator import Evaluator


class MainTrainer(BaseTrainer):
    """Main trainer.

    It's better to define different trainers for different models if
    there exists significant difference within training and evaluation
    processes (e.g., model input, advanced data processing, graph node
    sampling, MTL criterion definition).

    Args:
        logger: message logger
        trainer_cfg: hyperparameters for training and evaluation processes
        model: model instance
        loss_fn: loss criterion
        optimizer: optimization algorithm
        lr_scheduler: learning rate scheduler
        scaler: scaling object
        train_loader: training data loader
        eval_loader: validation data loader
        use_wandb: if True, training and evaluation processes are
            tracked with wandb
    """

    def __init__(
        self,
        logger: Logger,
        trainer_cfg: DictConfig,
        model: Module,
        loss_fn: _Loss,
        optimizer: Optimizer,
        lr_skd: Union[_LRScheduler, lr_scheduler.ReduceLROnPlateau],
        ckpt_path: Path,
        evaluator: Evaluator,
        scaler: Any,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        use_wandb: bool = True,
    ):
        super(MainTrainer, self).__init__(
            logger,
            trainer_cfg,
            model,
            loss_fn,
            optimizer,
            lr_skd,
            ckpt_path,
            evaluator,
            use_wandb,
        )
        self.train_loader = train_loader
        self.eval_loader = eval_loader if eval_loader else train_loader
        self.scaler = scaler
        self.rescale = trainer_cfg["loss_fn"]["rescale"]

        self.loss_name = self.loss_fn.__class__.__name__

    def _train_epoch(self) -> float:
        """Run training process for one epoch.

        Returns:
            train_loss_avg: average training loss over batches
        """
        train_loss_total = 0

        self.model.train()
        for i, batch_data in enumerate(tqdm(self.train_loader)):
            if i % self.grad_accum_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)

            # Retrieve batched raw data
            inputs = {}
            for k, v in batch_data.items():
                if k != "y":
                    inputs[k] = v.to(self.device)
                else:
                    y = v.to(self.device)

            # Forward pass and derive loss
            output = self.model(inputs)

            # Derive loss
            loss = self.loss_fn(output, y)
            train_loss_total += loss.item()
            loss = loss / self.grad_accum_steps

            # Backpropagation
            loss.backward()
            if (i + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e-3)
                self.optimizer.step()
                if self.step_per_batch:
                    self.lr_skd.step()

            self._iter += 1

            # Free mem.
            del inputs, y, output
            _ = gc.collect()

        train_loss_avg = train_loss_total / len(self.train_loader)

        return train_loss_avg

    @torch.no_grad()
    def _eval_epoch(
        self,
        return_output: bool = False,
    ) -> Tuple[float, Dict[str, float], Optional[Tensor]]:
        """Run evaluation process for one epoch.

        Args:
            return_output: whether to return prediction

        Returns:
            eval_loss_avg: average evaluation loss over batches
            eval_result: evaluation performance report
            y_pred: prediction
        """
        eval_loss_total = 0
        y_true, y_pred = [], []

        self.model.eval()
        for i, batch_data in enumerate(self.eval_loader):
            # Retrieve batched raw data
            inputs = {}
            for k, v in batch_data.items():
                if k != "y":
                    inputs[k] = v.to(self.device)
                else:
                    y = v.to(self.device)

            # Forward pass
            output = self.model(inputs)

            # Derive loss
            loss = self.loss_fn(output, y)
            eval_loss_total += loss.item()

            # Record batched output
            y_true.append(y.detach().cpu())
            y_pred.append(output.detach().cpu())

            del inputs, y, output
            _ = gc.collect()

        eval_loss_avg = eval_loss_total / len(self.eval_loader)

        # Run evaluation with the specified evaluation metrics
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        eval_result = self.evaluator.evaluate(y_true, y_pred, self.scaler)

        if return_output:
            return eval_loss_avg, eval_result, y_pred
        else:
            return eval_loss_avg, eval_result, None
