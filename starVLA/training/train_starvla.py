# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 


"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate DDP, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).  
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.  
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).  
"""

# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple
from torch.utils.data import DataLoader
import time

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs, set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import get_scheduler

# Local Modules
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from starVLA.training.trainer_utils.trainer_tools import apply_training_freeze_policy
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups
from starVLA.training.trainer_utils.trainer_tools import build_per_group_scheduler
from starVLA.training.trainer_utils.config_tracker import wrap_config, AccessTrackedConfig

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
logger = get_logger(__name__)

LEGACY_METRIC_KEY_ALIASES = {
    "train_total_loss": "train_loss_total",
    "train_flow_loss": "train_loss_flow",
    "train_perceptual_loss": "train_loss_perceptual",
    "train_distill_loss": "train_loss_distill",
    "train_mse_loss": "train_loss_mse",
    "train_lpips_loss": "train_loss_lpips",
    "val_total_loss": "val_loss_total",
    "val_flow_loss": "val_loss_flow",
    "val_perceptual_loss": "val_loss_perceptual",
    "val_distill_loss": "val_loss_distill",
    "val_mse_loss": "val_loss_mse",
    "val_lpips_loss": "val_loss_lpips",
}

TRAIN_COMPONENT_METRIC_ALIASES = {
    "train_loss_total": ("total_loss", "loss_total"),
    "train_loss_flow": ("loss_flow", "flow_loss"),
    "train_loss_perceptual": ("loss_perceptual", "perceptual_loss"),
    "train_loss_distill": ("loss_distill", "distill_loss"),
    "train_loss_mse": ("loss_mse", "mse_loss"),
    "train_loss_lpips": ("loss_lpips", "lpips_loss"),
}

def _accumulate_eval_scalar(metric_numerator: torch.Tensor, metric_denominator: torch.Tensor, value) -> None:
    """Accumulate an optional scalar metric as float64 sum/count."""
    if not torch.is_tensor(value):
        return
    scalar = value.detach()
    if scalar.numel() != 1:
        scalar = scalar.mean()
    metric_numerator += scalar.to(dtype=torch.float64)
    metric_denominator += 1


def _coerce_config_bool(value, *, default: bool, field_name: str) -> bool:
    """Parse config booleans explicitly to avoid `bool("false") == True` surprises."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Invalid boolean value for `{field_name}`: {value!r}")


def build_accelerator(cfg) -> Accelerator:
    """build accelerator in DDP mode"""
    trainer_cfg = getattr(cfg, "trainer", None)
    gradient_accumulation_steps = int(getattr(trainer_cfg, "gradient_accumulation_steps", 1))
    ddp_find_unused_parameters = bool(getattr(trainer_cfg, "ddp_find_unused_parameters", True))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=ddp_find_unused_parameters)
    dataloader_config = DataLoaderConfiguration(non_blocking=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
        dataloader_config=dataloader_config,
    )
    accelerator.print(accelerator.state)
    return accelerator


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    base_run_id = str(cfg.run_id)
    run_timestamp = os.environ.get("LAWAM_RUN_TIMESTAMP")
    if dist.is_initialized():
        timestamp_list = [run_timestamp if dist.get_rank() == 0 else None]
        if dist.get_rank() == 0 and not timestamp_list[0]:
            timestamp_list[0] = time.strftime("%m%d_%H%M%S")
        dist.broadcast_object_list(timestamp_list, src=0)
        timestamp = timestamp_list[0]
    else:
        timestamp = run_timestamp or time.strftime("%m%d_%H%M%S")
    run_folder_name = f"{timestamp}+{base_run_id}"

    cfg.output_dir = os.path.join(cfg.run_root_dir, run_folder_name)
    cfg.log_dir = os.path.join(cfg.output_dir, "logs")
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
        os.makedirs(output_dir / "logs", exist_ok=True)

        # Save the full merged config so `.pt` checkpoints remain reloadable for inference.
        serialized_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.save(serialized_cfg, output_dir / "config.yaml")
        with open(output_dir / "config.yaml", "r", encoding="utf-8") as f_yaml, open(
            output_dir / "config.json", "w", encoding="utf-8"
        ) as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    return output_dir


# here changes need to 📦 encapsulate Dataloader
from starVLA.dataloader import build_dataloaders


def prepare_data(cfg, accelerator) -> tuple[DataLoader, Optional[DataLoader]]:
    """prepare training data"""
    # VLA data loader
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader, vla_val_dataloader = build_dataloaders(cfg=cfg)

    if accelerator.dataloader_config is not None:
        accelerator.dataloader_config.dispatch_batches = False
    # if dist.is_initialized():
    #     dist.barrier()

    return vla_train_dataloader, vla_val_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    base_warmup = int(cfg.trainer.num_warmup_steps)
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            warmup = group.get("num_warmup_steps", base_warmup)
            logger.info(
                f"LR Group {group['name']}: lr={group['lr']}, "
                f"warmup_steps={warmup}, num_params={len(group['params'])}"
            )

    # initialize learning rate scheduler (supports per-group warmup steps)
    has_per_group_warmup = any("num_warmup_steps" in g for g in optimizer.param_groups)
    if has_per_group_warmup:
        lr_scheduler = build_per_group_scheduler(optimizer=optimizer, cfg=cfg)
    else:
        lr_scheduler = get_scheduler(
            name=cfg.trainer.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=cfg.trainer.num_warmup_steps,
            num_training_steps=cfg.trainer.max_train_steps,
            scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
        )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(
        self,
        cfg,
        model,
        vla_train_dataloader,
        optimizer,
        lr_scheduler,
        accelerator,
        vla_val_dataloader=None,
    ):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vla_val_dataloader = vla_val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        # training status tracking
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        trackers = list(getattr(self.config, "trackers", [])) if hasattr(self.config, "trackers") else []
        self.use_wandb = "wandb" in trackers if trackers else True
    
    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Strict finetune initialization from a full model checkpoint, if configured.
        self._init_checkpointing()

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        if self.vla_val_dataloader is None:
            self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
                self.accelerator,  # must be the first param
                self.model,
                self.optimizer,
                self.vla_train_dataloader,
            )
        else:
            self.model, self.optimizer, self.vla_train_dataloader, self.vla_val_dataloader = (
                self.setup_distributed_training(
                    self.accelerator,  # must be the first param
                    self.model,
                    self.optimizer,
                    self.vla_train_dataloader,
                    self.vla_val_dataloader,
                )
            )

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        if not self.use_wandb:
            return
        if self.accelerator.is_main_process:
            try:
                # Force offline logging whenever wandb is enabled.
                os.environ["WANDB_MODE"] = "offline"
                wandb.init(
                    name=Path(self.config.output_dir).name,
                    dir=os.path.join(self.config.output_dir, "wandb"),
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    group="vla-train",
                    mode="offline",
                )
            except Exception as e:
                self.use_wandb = False
                logger.warning(f"W&B init failed, disable wandb logging for this run: {e}")

    def _init_checkpointing(self):
        """Initialize the checkpoint directory and optionally strict-load finetune init weights."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        load_pretrained_policy_flow = _coerce_config_bool(
            getattr(self.config.trainer, "load_pretrained_policy_flow", True),
            default=True,
            field_name="trainer.load_pretrained_policy_flow",
        )
        load_pretrained_lam = _coerce_config_bool(
            getattr(self.config.trainer, "load_pretrained_lam", True),
            default=True,
            field_name="trainer.load_pretrained_lam",
        )

        if pretrained_checkpoint:
            self.model = self.load_finetune_init_weights(
                self.model,
                checkpoint_path=pretrained_checkpoint,
                load_pretrained_policy_flow=load_pretrained_policy_flow,
                load_pretrained_lam=load_pretrained_lam,
            )
            self.completed_steps = 0
            logger.info(
                "Initialized model weights for finetune from `%s`; "
                "load_pretrained_policy_flow=%s; load_pretrained_lam=%s; training starts from step 0.",
                pretrained_checkpoint,
                load_pretrained_policy_flow,
                load_pretrained_lam,
            )
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _save_checkpoint(self):
        """save current training state"""

        if self.accelerator.is_main_process:

            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            # save model state
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")

            # save training metadata
            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
            # ✅ Save accessed configuration only
            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                # self.config.save_accessed_config(
                #     output_dir / "config.json", 
                #     use_original_values=False
                # )
                self.config.save_accessed_config(
                    output_dir / "config.yaml", 
                    use_original_values=False 
                )
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _get_learning_rate_metrics(self):
        """Collect learning-rate metrics for all optimizer parameter groups."""
        lr_metrics = {}
        for idx, group in enumerate(self.optimizer.param_groups):
            group_name = str(group.get("name") or f"group_{idx}")
            lr_metrics[f"learning_rate/{group_name}"] = float(group["lr"])

        if self.optimizer.param_groups:
            lr_metrics["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])

        return lr_metrics

    def _normalize_metric_aliases(self, metrics):
        """Collapse legacy metric spellings onto the canonical W&B keys."""
        normalized_metrics = dict(metrics)
        for alias_key, canonical_key in LEGACY_METRIC_KEY_ALIASES.items():
            if alias_key not in normalized_metrics:
                continue
            if canonical_key not in normalized_metrics:
                normalized_metrics[canonical_key] = normalized_metrics[alias_key]
            normalized_metrics.pop(alias_key, None)
        return normalized_metrics

    def _log_metrics(self, metrics):
        """record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                metrics = self._normalize_metric_aliases(metrics)
                # Log both the legacy scalar learning rate and named lr groups.
                metrics.update(self._get_learning_rate_metrics())

                # add epoch info
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)

                # record to W&B
                if self.use_wandb:
                    wandb.log(metrics, step=self.completed_steps)
                # debug output
                logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """create data iterators"""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        # create progress bar
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps),
            desc=str(getattr(self.config, "run_id", "train")),
            disable=not self.accelerator.is_local_main_process,
        )

        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # Wait for the next batch from the prepared dataloader.
            # Under Accelerate device placement, host-to-device copies happen before the batch is yielded,
            # so the displayed `data_times` includes any H2D transfer done by the dataloader wrapper.
            t_start_data_wait = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data_wait = time.perf_counter()

            # Pure training-step compute time for the current process.
            t_start_compute = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_compute = time.perf_counter()

            # update progress
            step_advanced = False
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
                step_advanced = True
            
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                        {
                            "data_times": f"{t_end_data_wait - t_start_data_wait:.3f}",
                            "model_times": f"{t_end_compute - t_start_compute:.3f}",
                        }
                    )

            if step_advanced:
                # evaluate model
                if self.completed_steps % self.config.trainer.eval_interval == 0:
                    step_metrics = self.eval_action_model(step_metrics)

                # record metrics
                step_metrics["data_time"] = t_end_data_wait - t_start_data_wait
                step_metrics["model_time"] = t_end_compute - t_start_compute
                self._log_metrics(step_metrics)

                # save checkpoint
                if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                    self._save_checkpoint()

            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

        # execute evaluation step

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """
        if step_metrics is None:
            step_metrics = {}

        if self.vla_val_dataloader is None:
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                logger.warning("Validation dataloader is None, skip eval_action_model for this step.")
            if dist.is_initialized():
                dist.barrier()
            return step_metrics

        eval_batches = int(getattr(self.config.trainer, "eval_batches", 50))
        if eval_batches <= 0:
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                logger.warning(f"`trainer.eval_batches` must be > 0, got {eval_batches}; skip eval.")
            if dist.is_initialized():
                dist.barrier()
            return step_metrics

        eval_model = self.accelerator.unwrap_model(self.model)
        model_was_training = bool(getattr(eval_model, "training", True))
        eval_model.eval()

        if not hasattr(eval_model, "policy_runner"):
            if self.accelerator.is_main_process:
                logger.warning(
                    "eval_action_model requires `model.policy_runner.infer_step_with_aligned_targets_from_train_batch`; "
                    "skip eval for framework `%s`.",
                    str(getattr(getattr(self.config, "framework", None), "name", "")),
                )
            if model_was_training:
                eval_model.train()
            if dist.is_initialized():
                dist.barrier()
            return step_metrics

        metric_numerator = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        metric_denominator = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        val_loss_perceptual_numerator = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        val_loss_perceptual_denominator = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        val_loss_distill_numerator = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        val_loss_distill_denominator = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        val_iter = iter(self.vla_val_dataloader)
        policy_backend = getattr(eval_model, "policy_backend", None)
        if policy_backend is not None and hasattr(policy_backend, "set_flow_train_step"):
            policy_backend.set_flow_train_step(self.completed_steps)

        try:
            with torch.no_grad():
                for _ in range(eval_batches):
                    try:
                        batch = next(val_iter)
                    except StopIteration:
                        break

                    try:
                        eval_output_dict = eval_model.forward(batch)
                    except Exception as exc:
                        if self.accelerator.is_main_process:
                            logger.warning(f"LatentWorld eval skip forward loss metrics due to runtime error: {exc}")
                    else:
                        _accumulate_eval_scalar(
                            val_loss_perceptual_numerator,
                            val_loss_perceptual_denominator,
                            eval_output_dict.get("loss_perceptual"),
                        )
                        _accumulate_eval_scalar(
                            val_loss_distill_numerator,
                            val_loss_distill_denominator,
                            eval_output_dict.get("loss_distill"),
                        )

                    try:
                        pred_actions, gt_actions, action_mask = (
                            eval_model.policy_runner.infer_step_with_aligned_targets_from_train_batch(batch)
                        )
                    except Exception as exc:
                        if self.accelerator.is_main_process:
                            logger.warning(f"LatentWorld eval skip action MSE metrics due to runtime error: {exc}")
                        continue

                    if pred_actions.shape != gt_actions.shape or pred_actions.shape != action_mask.shape:
                        if self.accelerator.is_main_process:
                            logger.warning(
                                "LatentWorld eval skip batch due to shape mismatch: "
                                f"pred={tuple(pred_actions.shape)}, "
                                f"gt={tuple(gt_actions.shape)}, "
                                f"mask={tuple(action_mask.shape)}"
                            )
                        continue

                    valid_count = action_mask.sum()
                    if int(valid_count.item()) <= 0:
                        if self.accelerator.is_main_process:
                            logger.warning("LatentWorld eval skip batch: empty valid action mask.")
                        continue

                    sq_error_sum = ((pred_actions - gt_actions) ** 2).masked_select(action_mask).sum()
                    metric_numerator += sq_error_sum.to(dtype=torch.float64)
                    metric_denominator += valid_count.to(dtype=torch.float64)
        finally:
            if model_was_training:
                eval_model.train()

        if dist.is_initialized():
            dist.all_reduce(metric_numerator, op=dist.ReduceOp.SUM)
            dist.all_reduce(metric_denominator, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_perceptual_numerator, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_perceptual_denominator, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_distill_numerator, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_distill_denominator, op=dist.ReduceOp.SUM)

        if self.accelerator.is_main_process:
            if float(metric_denominator.item()) > 0:
                val_mse_score = float((metric_numerator / metric_denominator).item())
                step_metrics["val_mse_score"] = val_mse_score
            else:
                logger.warning(
                    "Validation eval produced no valid batches/elements; skip metric logging for this eval step."
                )

            if float(val_loss_perceptual_denominator.item()) > 0:
                step_metrics["val_loss_perceptual"] = float(
                    (val_loss_perceptual_numerator / val_loss_perceptual_denominator).item()
                )
            else:
                logger.warning("Validation eval produced no valid batches for `val_loss_perceptual`.")

            if float(val_loss_distill_denominator.item()) > 0:
                step_metrics["val_loss_distill"] = float(
                    (val_loss_distill_numerator / val_loss_distill_denominator).item()
                )
            else:
                logger.warning("Validation eval produced no valid batches for `val_loss_distill`.")

        if dist.is_initialized():
            dist.barrier()  # ensure all processes are synchronized
        return step_metrics

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla):
        """execute single training step"""
        # Guard against mode leakage from eval_action_model().
        self.model.train()
        base_model = self.accelerator.unwrap_model(self.model)
        policy_backend = getattr(base_model, "policy_backend", None)
        if policy_backend is not None and hasattr(policy_backend, "set_flow_train_step"):
            policy_backend.set_flow_train_step(self.completed_steps)
        with self.accelerator.accumulate(self.model):
            # VLA task forward propagation
            output_dict = self.model.forward(batch_vla)
            if "total_loss" not in output_dict:
                raise KeyError("Model forward must return `total_loss`.")
            total_loss = output_dict["total_loss"]

            # VLA backward propagation
            self.accelerator.backward(total_loss)

            # Clip gradients only on the accumulation boundary right before the real optimizer update.
            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            # optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.accelerator.sync_gradients:
            self.lr_scheduler.step()

        metrics = {"train_loss": float(total_loss.detach().item())}
        # Optional component losses with stable logging keys.
        for log_key, raw_keys in TRAIN_COMPONENT_METRIC_ALIASES.items():
            for raw_key in raw_keys:
                if raw_key in output_dict and torch.is_tensor(output_dict[raw_key]):
                    metrics[log_key] = float(output_dict[raw_key].detach().item())
                    break
        return self._normalize_metric_aliases(metrics)

    def _finalize_training(self):
        """training end processing"""
        # save final model
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")


        # close W&B
        if self.accelerator.is_main_process and self.use_wandb:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    accelerator = build_accelerator(cfg)
    logger.info("VLA Training :: Warming Up")

    #  Wrap config to enable access tracking
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    # create output directory and save config
    setup_directories(cfg=cfg)
    # build model
    vla = build_framework(cfg)
    vla = apply_training_freeze_policy(vla, cfg)
    # prepare data
    vla_train_dataloader, vla_val_dataloader = prepare_data(cfg=cfg, accelerator=accelerator)

    # set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # create trainer
    # Run VLA Training
    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        vla_val_dataloader=vla_val_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/config/training/starvla_train_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # if cfg.is_debug:
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
