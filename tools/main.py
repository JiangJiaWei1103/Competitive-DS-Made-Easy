"""
Main script for training and evaluation processes.
Author: JiaWei Jiang

This script is used to run training and evaluation processes given the
specified CV scheme. Moreover, evaluation on unseen (test) data is
optional.

* [ ] Dump prediction from the best ckpt or not.
* [ ] Write `cross_validate` function.
* [ ] Use `instantiate` to build objects (e.g., model, optimizer).
"""
import gc
import math
import warnings

import hydra
from omegaconf.dictconfig import DictConfig

import wandb
from base.base_trainer import BaseTrainer
from config.config import get_seeds, seed_everything
from criterion.build import build_criterion
from cv.build import build_cv
from data.build import build_dataloader
from data.data_processor import DataProcessor
from evaluating.build import build_evaluator
from experiment.experiment import Experiment
from modeling.build import build_model
from solver.build import build_lr_scheduler, build_optimizer
from trainer.trainer import MainTrainer
from utils.common import count_params

warnings.simplefilter("ignore")


@hydra.main(config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    """Run training and evaluation processes.

    Args:
        cfg: configuration driving training and evaluation processes
    """
    # Configure experiment
    experiment = Experiment(cfg)

    with experiment as exp:
        exp.dump_cfg(exp.cfg, "main")

        # Prepare data
        dp = DataProcessor(**exp.data_cfg["dp"])
        dp.run_before_splitting()
        data = dp.get_data_cv()

        # Run cross-validation
        cv = build_cv(**exp.data_cfg["cv"])
        one_fold_only = exp.cfg["one_fold_only"]
        for s_i, seed in enumerate(get_seeds(exp.cfg["n_seeds"])):
            exp.log(f"\nSeed the experiment with {seed}...")
            seed_everything(seed)
            cfg_seed = exp.cfg.copy()
            cfg_seed["seed"] = seed

            for fold, (tr_idx, val_idx) in enumerate(cv.split(X=data)):
                # Configure sub-entry for tracking current fold
                seed_name, fold_name = f"seed{s_i}", f"fold{fold}"
                proc_id = f"{seed_name}_{fold_name}"
                if exp.cfg["use_wandb"]:
                    tr_eval_run = exp.add_wnb_run(
                        cfg=cfg_seed,
                        job_type=fold_name if one_fold_only else seed_name,
                        name=seed_name if one_fold_only else fold_name,
                    )
                exp.log(f"== Train and Eval Process - Fold{fold} ==")

                # Build dataloaders
                data_tr, data_val = data.iloc[tr_idx].reset_index(drop=True), data.iloc[val_idx].reset_index(drop=True)
                data_tr, data_val, scaler = dp.run_after_splitting(data_tr, data_val)
                train_loader = build_dataloader(
                    data_tr, "train", exp.data_cfg["dataset"], **exp.trainer_cfg["dataloader"]
                )
                val_loader = build_dataloader(
                    data_val, "val", exp.data_cfg["dataset"], **exp.trainer_cfg["dataloader"]
                )

                # Build model
                model = build_model(exp.model_name, **exp.model_cfg["model_params"])
                model.to(exp.trainer_cfg["device"])
                if exp.cfg["use_wandb"]:
                    wandb.log({"model": {"n_params": count_params(model)}})
                    wandb.watch(model, log="all", log_graph=True)

                # Build criterion
                loss_fn = build_criterion(**exp.trainer_cfg["loss_fn"])

                # Build solvers
                optimizer = build_optimizer(model, **exp.trainer_cfg["optimizer"])
                num_training_steps = (
                    math.ceil(
                        len(train_loader.dataset)
                        / exp.trainer_cfg["dataloader"]["batch_size"]
                        / exp.trainer_cfg["grad_accum_steps"]
                    )
                    * exp.trainer_cfg["epochs"]
                )
                lr_skd = build_lr_scheduler(optimizer, num_training_steps, **exp.trainer_cfg["lr_skd"])

                # Build evaluator
                evaluator = build_evaluator(**exp.trainer_cfg["evaluator"])

                # Build trainer
                trainer: BaseTrainer = None
                trainer = MainTrainer(
                    logger=exp.logger,
                    trainer_cfg=exp.trainer_cfg,
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    lr_skd=lr_skd,
                    ckpt_path=exp.ckpt_path,
                    evaluator=evaluator,
                    scaler=None,
                    train_loader=train_loader,
                    eval_loader=val_loader,
                    use_wandb=exp.cfg["use_wandb"],
                )

                # Run main training and evaluation for one fold
                trainer.train_eval(fold)

                # Run evaluation on unseen test set
                if False:
                    data_test = dp.get_data_test()
                    test_loader = build_dataloader(
                        data_test, "test", exp.data_cfg["dataset"], **exp.trainer_cfg["dataloader"]
                    )
                    _ = trainer.test(fold, test_loader)

                # Dump output objects
                if scaler is not None:
                    exp.dump_trafo(scaler, f"scaler_{proc_id}")
                for model_path in exp.ckpt_path.glob("*.pth"):
                    if "seed" in str(model_path) or "fold" in str(model_path):
                        continue

                    # Rename model file
                    model_file_name_dst = f"{model_path.stem}_{proc_id}.pth"
                    model_path_dst = exp.ckpt_path / model_file_name_dst
                    model_path.rename(model_path_dst)

                # Free mem.
                del (data_tr, data_val, train_loader, val_loader, model, optimizer, lr_skd, evaluator, trainer)
                _ = gc.collect()

                if exp.cfg["use_wandb"]:
                    tr_eval_run.finish()
                if one_fold_only:
                    exp.log("Cross-validatoin stops at first fold!!!")
                    break


if __name__ == "__main__":
    # Launch main function
    main()
