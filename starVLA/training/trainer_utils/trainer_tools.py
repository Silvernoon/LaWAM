"""
metrics.py

Utility classes defining a Metrics container and multiple Trackers to enable model/stage-specific logging to various
endpoints (e.g., JSONL local logs, Weights & Biases).
"""

from typing import Tuple
from numbers import Number
import re
import json
import numpy as np
import torch

from accelerate.logging import get_logger
from starVLA.model.framework.base_framework import log_full_checkpoint_summary
from starVLA.model.framework.base_framework import validate_full_checkpoint_state_dict

logger = get_logger(__name__)


# === Define Tracker Interface ===
#

# utils/cli_parser.py


def normalize_dotlist_args(args):
    """
    Convert ['--x.y', 'val'] and ['--flag'] → ['x.y=val', 'flag=true']
    """
    normalized = []
    skip = False
    for i in range(len(args)):
        if skip:
            skip = False
            continue

        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                normalized.append(key)
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                normalized.append(f"{key}={args[i + 1]}")
                skip = True
            else:
                normalized.append(f"{key}=true")
        else:
            pass  # skip orphaned values
    return normalized


def build_param_lr_groups(model, cfg):
    """
    build multiple param groups based on cfg.trainer.learning_rate.
    support specifying different learning rates for different modules, the rest use base.

    Supported configs:
      1) Legacy single-module syntax:
         learning_rate:
           base: 1e-4
           qwen_vl_interface: 1e-5
      2) Named group syntax (one group can contain multiple modules):
         learning_rate:
           base: 1e-4
           pretrained:
             lr: 1e-5
             modules: [policy_backend.vlm, policy_backend.lam.decoder]
      3) Per-group warmup (optional, named group only):
         learning_rate:
           base: 1e-4
           vlm:
             lr: 2e-5
             num_warmup_steps: 10000
             modules: [policy_backend.vlm]

    Args:
        vla: nn.Module model object
        cfg: config object, requires cfg.trainer.learning_rate dictionary

    Returns:
        List[Dict]: param_groups that can be used to build optimizer with torch.optim
    """

    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 1e-4)  # default base learning rate

    used_params = set()
    param_groups = []

    for group_name, group_spec in lr_cfg.items():
        if group_name == "base":
            continue

        group_warmup = None  # per-group warmup; None means fall back to base

        # Backward-compatible form: module_path: lr
        if isinstance(group_spec, Number):
            group_lr = float(group_spec)
            module_paths = [str(group_name)]
        else:
            # New form: group_name: {lr: <float>, modules: [path1, path2, ...], num_warmup_steps: <int> (optional)}
            if not hasattr(group_spec, "get"):
                logger.warning(
                    f"learning_rate group `{group_name}` has invalid spec `{group_spec}`; skip custom lr group"
                )
                continue
            group_lr = group_spec.get("lr", None)
            raw_modules = group_spec.get("modules", None)
            if group_lr is None or raw_modules is None:
                logger.warning(
                    f"learning_rate group `{group_name}` requires both `lr` and `modules`; skip custom lr group"
                )
                continue
            try:
                group_lr = float(group_lr)
            except (TypeError, ValueError):
                logger.warning(
                    f"learning_rate group `{group_name}` has non-numeric lr `{group_lr}`; skip custom lr group"
                )
                continue
            raw_warmup = group_spec.get("num_warmup_steps", None)
            if raw_warmup is not None:
                try:
                    group_warmup = int(raw_warmup)
                except (TypeError, ValueError):
                    logger.warning(
                        f"learning_rate group `{group_name}` has non-integer num_warmup_steps `{raw_warmup}`; ignore"
                    )
            if isinstance(raw_modules, str):
                module_paths = [raw_modules]
            else:
                try:
                    module_paths = [str(path) for path in raw_modules]
                except TypeError:
                    logger.warning(
                        f"learning_rate group `{group_name}` has non-iterable modules `{raw_modules}`; skip custom lr group"
                    )
                    continue

        if len(module_paths) == 0:
            logger.warning(f"learning_rate group `{group_name}` has empty modules; skip custom lr group")
            continue

        params = []
        for module_path in module_paths:
            # try to find the module under model by module_path (support nested paths)
            module = model
            try:
                for attr in str(module_path).split("."):
                    module = getattr(module, attr)
                for p in module.parameters():
                    pid = id(p)
                    if not p.requires_grad or pid in used_params:
                        continue
                    params.append(p)
                    used_params.add(pid)
            except AttributeError:
                logger.warning(f"module path `{module_path}` not found in model; skip this module in group `{group_name}`")

        # only add param group if there are trainable parameters
        if params:
            group_dict = {"params": params, "lr": group_lr, "name": str(group_name)}
            if group_warmup is not None:
                group_dict["num_warmup_steps"] = group_warmup
            param_groups.append(group_dict)

    # assign base learning rate to remaining trainable parameters
    other_params = []
    for p in model.parameters():
        pid = id(p)
        if not p.requires_grad or pid in used_params:
            continue
        other_params.append(p)
        used_params.add(pid)
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "base"})

    if not param_groups:
        raise ValueError("No trainable parameters found when building optimizer parameter groups.")

    return param_groups


import math
from functools import partial
from torch.optim.lr_scheduler import LambdaLR


def _linear_warmup_then_linear_decay_lambda(
    current_step, *, num_warmup_steps, num_training_steps
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def _cosine_with_min_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles, min_lr_rate
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    factor = factor * (1 - min_lr_rate) + min_lr_rate
    return max(0, factor)


def _cosine_warmup_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def _constant_with_warmup_lambda(current_step, *, num_warmup_steps):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return 1.0


def build_per_group_scheduler(optimizer, cfg):
    """Build a LambdaLR scheduler that supports per-group num_warmup_steps.

    Each optimizer param group may carry an optional ``num_warmup_steps`` key
    (set by ``build_param_lr_groups``).  Groups without this key fall back to
    ``cfg.trainer.num_warmup_steps``.

    Returns:
        torch.optim.lr_scheduler.LambdaLR
    """
    base_warmup = int(cfg.trainer.num_warmup_steps)
    num_training_steps = int(cfg.trainer.max_train_steps)
    scheduler_type = str(cfg.trainer.lr_scheduler_type).lower().replace("-", "_")
    raw_kwargs = getattr(cfg.trainer, "scheduler_specific_kwargs", None) or {}
    scheduler_kwargs = dict(raw_kwargs) if not isinstance(raw_kwargs, dict) else raw_kwargs

    lr_lambdas = []
    for group in optimizer.param_groups:
        warmup = group.get("num_warmup_steps", base_warmup)
        group_name = group.get("name", "?")

        if scheduler_type in ("cosine_with_min_lr",):
            min_lr = scheduler_kwargs.get("min_lr", None)
            min_lr_rate = scheduler_kwargs.get("min_lr_rate", None)
            if min_lr is not None and min_lr_rate is None:
                min_lr_rate = float(min_lr) / float(optimizer.defaults["lr"])
            elif min_lr_rate is None:
                min_lr_rate = 0.0
            num_cycles = float(scheduler_kwargs.get("num_cycles", 0.5))
            lr_lambdas.append(partial(
                _cosine_with_min_lr_lambda,
                num_warmup_steps=warmup,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
                min_lr_rate=min_lr_rate,
            ))
        elif scheduler_type in ("cosine", "cosine_with_warmup"):
            num_cycles = float(scheduler_kwargs.get("num_cycles", 0.5))
            lr_lambdas.append(partial(
                _cosine_warmup_lambda,
                num_warmup_steps=warmup,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
            ))
        elif scheduler_type in ("linear",):
            lr_lambdas.append(partial(
                _linear_warmup_then_linear_decay_lambda,
                num_warmup_steps=warmup,
                num_training_steps=num_training_steps,
            ))
        elif scheduler_type in ("constant_with_warmup",):
            lr_lambdas.append(partial(
                _constant_with_warmup_lambda,
                num_warmup_steps=warmup,
            ))
        else:
            raise ValueError(
                f"Per-group warmup is not supported for scheduler type `{scheduler_type}`. "
                f"Supported types: cosine_with_min_lr, cosine, linear, constant_with_warmup."
            )

        logger.info(
            "Scheduler group `%s`: lr=%.2e, warmup_steps=%d (base=%d)",
            group_name, group["lr"], warmup, base_warmup,
        )

    return LambdaLR(optimizer, lr_lambdas)


def apply_training_freeze_policy(model, cfg):
    trainer_cfg = getattr(cfg, "trainer", None)
    freeze_cfg = getattr(trainer_cfg, "freeze", None) if trainer_cfg is not None else None

    freeze_fn = getattr(model, "apply_training_freeze_policy", None)
    if callable(freeze_fn):
        freeze_fn(freeze_cfg)
    return model


import torch.distributed as dist


def only_main_process(func):
    """
    decorator: only run in main process (rank=0)
    """

    def wrapper(*args, **kwargs):
        if dist.is_initialized() and dist.get_rank() != 0:
            return None  # non-main process does not execute
        return func(*args, **kwargs)

    return wrapper


from torchvision.ops import box_iou
from PIL import Image


def resize_images(images, target_size=(224, 224)):
    """
    recursively resize all images in the nested list.

    :param images: nested list of images or single image.
    :param target_size: target size (width, height) after resizing.
    :return: resized images list, keeping the original nested structure.
    """
    if isinstance(images, Image.Image):  # if it is a single PIL image
        return images.resize(target_size)
    elif isinstance(images, list):  # if it is a list, recursively process each element
        return [resize_images(img, target_size) for img in images]
    else:
        raise ValueError("Unsupported image type or structure.")




class TrainerUtils:
    @staticmethod
    def freeze_backbones(model, freeze_modules=""):
        """
        directly freeze the specified submodules based on the relative module path list (patterns), no longer recursively find all submodule names:
          - patterns: read from config.trainer.freeze_modules, separated by commas to get the "relative path" list
            for example "qwen_vl_interface, action_model.net",
            it means to freeze model.qwen_vl_interface and model.action_model.net.

        Args:
            model: nn.Module model object
            freeze_modules: relative module path list (patterns)

        Returns:
            model: nn.Module model object
        return:
          - model:
        """
        frozen = []
        print("#"*30)
        print(freeze_modules)
        if freeze_modules and type(freeze_modules) == str:
            # split and remove whitespace
            patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()] if freeze_modules else []

            for path in patterns:
                # split the "relative path" by dots, for example "action_model.net" → ["action_model", "net"]
                attrs = path.split(".")
                module = model
                try:
                    for attr in attrs:
                        module = getattr(module, attr)
                    # if the module is successfully get, freeze it and its all submodule parameters
                    for param in module.parameters():
                        param.requires_grad = False
                    frozen.append(path)
                except AttributeError:
                    # if the attribute does not exist, skip and print warning
                    print(f"⚠️ module path does not exist, cannot freeze: {path}")
                    continue

        # accelerator.wait_for_everyone()  # synchronize when distributed training
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print(f"🔒 Frozen modules with re pattern: {frozen}")
        return model

    @staticmethod
    def print_trainable_parameters(model):
        """
        print the total number of parameters and trainable parameters of the model
        :param model: PyTorch model instance
        """
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        print("📊 model parameter statistics:")
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
        )
        return num_params, num_trainable_params

    @staticmethod
    def load_pretrained_backbones(model, checkpoint_path=None, reload_modules=None):
        """
        load checkpoint:
        - if reload_modules is set, load by path part
        - otherwise → load the entire model parameters (overwrite model)

        return:
            replace, loaded_modules: list of module paths that successfully loaded parameters; if global load, then ["<full_model>"]
        """
        if not checkpoint_path:
            return []
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print(f"📦 loading checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"❌ loading checkpoint failed: {e}")

        loaded_modules = []

        if reload_modules:  # partial load
            module_paths = [p.strip() for p in reload_modules.split(",") if p.strip()]
            for path in module_paths:
                reload_modules = path.split(".")
                module = model
                try:
                    for module_name in reload_modules:  # find the module to modify level by level
                        module = getattr(module, module_name)
                    prefix = path + "."
                    sub_state_dict = {k[len(prefix) :]: v for k, v in checkpoint.items() if k.startswith(prefix)}
                    if sub_state_dict:
                        module.load_state_dict(sub_state_dict, strict=True)
                        if (not dist.is_initialized()) or dist.get_rank() == 0:
                            print(f"✅ parameters loaded to module '{path}'")
                        loaded_modules.append(path)
                    else:
                        print(f"⚠️ parameters not found in checkpoint '{path}'")
                except AttributeError:
                    print(f"❌ cannot find module path: {path}")
        else:  # full load
            try:
                model.load_state_dict(checkpoint, strict=False)
                if (not dist.is_initialized()) or dist.get_rank() == 0:
                    print("✅ loaded <full_model> model parameters")
                loaded_modules = ["<full_model>"]
            except Exception as e:
                raise RuntimeError(f"❌ loading full model failed: {e}")
        return model

    @staticmethod
    def load_finetune_init_weights(
        model,
        checkpoint_path: str,
        load_pretrained_policy_flow: bool = True,
        load_pretrained_lam: bool = True,
    ):
        """Initialize a model from a finetune checkpoint with explicit module selection."""
        def _info(msg, *args):
            try:
                logger.info(msg, *args)
            except RuntimeError:
                formatted = msg % args if args else msg
                print(formatted)

        def _warn(msg, *args):
            try:
                logger.warning(msg, *args)
            except RuntimeError:
                formatted = msg % args if args else msg
                print(f"⚠️ {formatted}")

        if not checkpoint_path:
            raise ValueError("`checkpoint_path` must be a non-empty path for finetune initialization.")

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print(f"📦 relaxed finetune init checkpoint: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"❌ loading checkpoint failed: {e}") from e

        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                f"❌ expected checkpoint state_dict to be a dict, got `{type(checkpoint).__name__}` from `{checkpoint_path}`"
            )

        checkpoint_keys = set(checkpoint.keys())
        model_state_dict = model.state_dict()
        framework_cfg = getattr(getattr(model, "config", None), "framework", None)
        framework_name = str(getattr(framework_cfg, "name", "")).lower()

        prefix_counts = validate_full_checkpoint_state_dict(
            checkpoint_keys=checkpoint_keys,
            framework_name=framework_name,
            checkpoint_path=checkpoint_path,
        )

        filtered_checkpoint = {}
        unexpected_keys = []
        incompatible_shapes = []
        skipped_policy_flow_keys = []
        skipped_lam_keys = []
        for key, value in checkpoint.items():
            if (not load_pretrained_policy_flow) and (
                key == "policy_backend.flow"
                or key.startswith("policy_backend.flow.")
                or key == "policy_action_head"
                or key.startswith("policy_action_head.")
            ):
                skipped_policy_flow_keys.append(key)
                continue
            if (not load_pretrained_lam) and (
                key == "policy_backend.lam" or key.startswith("policy_backend.lam.")
            ):
                skipped_lam_keys.append(key)
                continue

            if key not in model_state_dict:
                unexpected_keys.append(key)
                continue

            model_value = model_state_dict[key]
            if hasattr(value, "shape") and hasattr(model_value, "shape"):
                if tuple(value.shape) != tuple(model_value.shape):
                    incompatible_shapes.append((key, tuple(value.shape), tuple(model_value.shape)))
                    continue

            filtered_checkpoint[key] = value

        try:
            incompatible_keys = {key for key, _, _ in incompatible_shapes}
            load_result = model.load_state_dict(filtered_checkpoint, strict=False)
        except RuntimeError as e:
            raise RuntimeError(f"❌ relaxed finetune init failed: {e}") from e

        missing_keys = sorted(load_result.missing_keys)
        intentionally_skipped_keys = []
        if skipped_policy_flow_keys:
            skipped_policy_flow_key_set = set(skipped_policy_flow_keys)
            intentionally_skipped_keys = sorted(skipped_policy_flow_key_set.intersection(missing_keys))
            missing_keys = sorted(set(missing_keys) - skipped_policy_flow_key_set)
            _info(
                "Relaxed finetune init skipped `policy_backend.flow` weights for `%s`: count=%d sample=%s",
                checkpoint_path,
                len(skipped_policy_flow_keys),
                skipped_policy_flow_keys[:10],
            )
        if skipped_lam_keys:
            skipped_lam_key_set = set(skipped_lam_keys)
            skipped_lam_missing_keys = sorted(skipped_lam_key_set.intersection(missing_keys))
            intentionally_skipped_keys.extend(skipped_lam_missing_keys)
            missing_keys = sorted(set(missing_keys) - skipped_lam_key_set)
            _info(
                "Relaxed finetune init skipped `policy_backend.lam` weights for `%s`: count=%d sample=%s",
                checkpoint_path,
                len(skipped_lam_keys),
                skipped_lam_keys[:10],
            )
        if unexpected_keys:
            _warn(
                "Relaxed finetune init ignored unexpected checkpoint keys for `%s`: count=%d sample=%s",
                checkpoint_path,
                len(unexpected_keys),
                sorted(unexpected_keys)[:10],
            )
        if incompatible_shapes:
            sample = [
                f"{key}: ckpt{ckpt_shape} != model{model_shape}"
                for key, ckpt_shape, model_shape in incompatible_shapes[:10]
            ]
            _warn(
                "Relaxed finetune init ignored shape-mismatched checkpoint keys for `%s`: count=%d sample=%s",
                checkpoint_path,
                len(incompatible_shapes),
                sample,
            )
        if intentionally_skipped_keys:
            _info(
                "Relaxed finetune init preserved target-initialized module weights for `%s`: count=%d sample=%s",
                checkpoint_path,
                len(intentionally_skipped_keys),
                intentionally_skipped_keys[:10],
            )
        if missing_keys:
            random_init_only = sorted(set(missing_keys) - incompatible_keys)
            _warn(
                "Relaxed finetune init left model keys at existing initialization for `%s`: count=%d sample=%s",
                checkpoint_path,
                len(missing_keys),
                missing_keys[:10],
            )
            if random_init_only:
                _warn(
                    "Relaxed finetune init random-init-only keys for `%s`: count=%d sample=%s",
                    checkpoint_path,
                    len(random_init_only),
                    random_init_only[:10],
                )
        if load_result.unexpected_keys:
            _warn(
                "Relaxed finetune init reported unexpected load_state_dict keys for `%s`: count=%d sample=%s",
                checkpoint_path,
                len(load_result.unexpected_keys),
                load_result.unexpected_keys[:10],
            )

        log_full_checkpoint_summary(
            framework_name=framework_name,
            checkpoint_path=checkpoint_path,
            checkpoint_keys=checkpoint_keys,
            prefix_counts=prefix_counts,
        )
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print("✅ relaxed full-model checkpoint loaded")
        return model

    @staticmethod
    def print_freeze_status(model):
        """
        print the freezing status of each parameter in the model
        :param model: PyTorch model instance
        """
        for name, param in model.named_parameters():
            status = "Frozen" if not param.requires_grad else "Trainable"
            print(f"{name:60s}  |  {status}")

    @staticmethod
    def setup_distributed_training(accelerator, *components):
        """
        use Accelerator to prepare distributed training components
        :param accelerator: Accelerate instance
        :param components: any number of components (such as model, optimizer, dataloader, etc.)
        :return: prepared distributed components (in the same order as input)
        """

        # use accelerator.prepare method to wrap components
        prepared_components = accelerator.prepare(*components)
        return prepared_components

    @staticmethod
    def euclidean_distance(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        return np.linalg.norm(predicted - ground_truth)

    @staticmethod
    def _set_epoch_on_dataset_tree(dataset, epoch_counter, visited=None):
        """Propagate epoch updates through wrapped datasets when sampling depends on dataset-local RNG."""
        if dataset is None:
            return False

        if visited is None:
            visited = set()
        dataset_id = id(dataset)
        if dataset_id in visited:
            return False
        visited.add(dataset_id)

        updated = False
        if callable(getattr(dataset, "set_epoch", None)):
            dataset.set_epoch(epoch_counter)
            updated = True

        child_dataset = getattr(dataset, "dataset", None)
        if child_dataset is not None and child_dataset is not dataset:
            updated = TrainerUtils._set_epoch_on_dataset_tree(child_dataset, epoch_counter, visited) or updated

        child_datasets = getattr(dataset, "datasets", None)
        if child_datasets is not None:
            for child in child_datasets:
                updated = TrainerUtils._set_epoch_on_dataset_tree(child, epoch_counter, visited) or updated

        return updated

    @staticmethod
    def _reset_dataloader(dataloader, epoch_counter):
        """safe reset dataloader iterator"""
        # 1. update epoch counter
        epoch_counter += 1

        # 2. set new epoch (distributed core)
        if hasattr(dataloader, "sampler") and callable(getattr(dataloader.sampler, "set_epoch", None)):
            dataloader.sampler.set_epoch(epoch_counter)
        if hasattr(dataloader, "dataset"):
            TrainerUtils._set_epoch_on_dataset_tree(dataloader.dataset, epoch_counter)

        # 3. create new iterator
        return iter(dataloader), epoch_counter

    @staticmethod
    def compute_grad_angle_with_stats(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> Tuple[float, float]:
        """
        compute the cosine angle between two groups of gradient vectors (degrees), and calculate the average angle and variance.
        grads_a, grads_v: gradient Tensor list corresponding to the same parameter list interface_params
        return:
            mean_angle_deg: average angle (degrees)
            angle_variance: angle variance
        """
        angle_degs = []

        # compute the cosine angle between each gradient block grads_a[0].shape = 1280, 3, 14, 14
        # grads_1 = grads_a[0][0]  # [3, 14, 14]
        # grads_2 = grads_v[0][0]
        # grads_a = grads_1.view(-1, 3)  # reshape to [196, 3]
        # grads_v = grads_2.view(-1, 3)

        # lang linear
        # reshape to 14*14, 3
        # layer
        grads_action = grads_a[0]  # [2048, 11008]
        grads_action = grads_action[
            :32, :7
        ]  # only take the first 7 elements, avoid cosim failure in high-dimensional space
        grads_vl = grads_v[0]  # [2048, 11008]
        grads_vl = grads_vl[
            :32, :7
        ]  # only take the first 32 elements, 7 dimensions, avoid cosim failure in high-dimensional space
        for g_a, g_v in zip(grads_action, grads_vl):
            dot = torch.sum(g_a * g_v)
            norm_a_sq = torch.sum(g_a * g_a)
            norm_v_sq = torch.sum(g_v * g_v)

            # avoid division by zero
            norm_a = torch.sqrt(norm_a_sq + 1e-16)
            norm_v = torch.sqrt(norm_v_sq + 1e-16)

            cos_sim = (dot / (norm_a * norm_v)).clamp(-1.0, 1.0)
            angle_rad = torch.acos(cos_sim)
            angle_deg = angle_rad * (180.0 / torch.pi)

            angle_degs.append(angle_deg.item())

        # compute the average angle and variance
        angle_degs_tensor = torch.tensor(angle_degs)
        mean_angle_deg = torch.mean(angle_degs_tensor).item()
        angle_variance = torch.sqrt(torch.var(angle_degs_tensor)).item()
        # accelerator.wait_for_everyone()
        return mean_angle_deg, angle_variance

    @staticmethod
    def pcgrad_project(grads_a: list[torch.Tensor], grads_v: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        apply PCGrad projection to the second group of gradients grads_v, suppress negative transfer between grads_a and grads_v
        if the dot product of two groups of gradients < 0, then:
            grads_v <- grads_v - (dot / ||grads_a||^2) * grads_a
        return the new grads_v list
        """
        # first compute dot and ||grads_a||^2
        dot, norm_a_sq = 0.0, 0.0
        for g_a, g_v in zip(grads_a, grads_v):
            dot += torch.sum(g_a * g_v)
            norm_a_sq += torch.sum(g_a * g_a)

        if dot < 0:
            coeff = dot / (norm_a_sq + 1e-6)
            # projection
            grads_v = [g_v - coeff * g_a for g_a, g_v in zip(grads_a, grads_v)]

        return grads_v

    @staticmethod
    def eval_qwenpi(qwenpi, dataloader, num_batches=20):
        """
        evaluate QwenQFormerDiT model, compute IoU and action distance.

        Args:
            qwenpi: QwenQFormerDiT model instance.
            dataloader: data loader.
            num_batches: number of batches to evaluate.

        Returns:
            dict: contains IoU and action distance evaluation results.
        """
        iou_scores = []
        action_distances = []
        count = 0

        dataset_iter = iter(dataloader)
        while count < num_batches:
            try:
                batch_samples = next(dataset_iter)
                count += 1
            except StopIteration:
                break

            # extract data
            images = [example["image"] for example in batch_samples]
            instructions = [example["lang"] for example in batch_samples]
            actions = [example["action"] for example in batch_samples]
            solutions = [example["solution"] for example in batch_samples]

            # model prediction
            predicted_solutions, normalized_actions = qwenpi.predict_action_withCoT(
                images=images, instructions=instructions
            )

            # extract and convert predicted results
            parsed_solutions = []
            for solution in predicted_solutions:
                parsed_solution = TrainerUtils.extract_json_from_string(solution)
                parsed_solutions.append(parsed_solution)

            # compute IoU
            for pred_dict, gt_dict in zip(parsed_solutions, solutions):
                pred_pick_bbox = torch.tensor(pred_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_pick_bbox = torch.tensor(gt_dict["pick"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                pred_place_bbox = torch.tensor(pred_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)
                gt_place_bbox = torch.tensor(gt_dict["place"]["bbox_2d"], dtype=torch.float32).unsqueeze(0)

                pick_iou = box_iou(pred_pick_bbox, gt_pick_bbox).item()
                place_iou = box_iou(pred_place_bbox, gt_place_bbox).item()

                iou_scores.append({"pick_iou": pick_iou, "place_iou": place_iou})

            # compute action distance
            actions = np.array(actions)  # convert to numpy array
            num_pots = np.prod(actions.shape)  # B*len*dim
            action_distance = TrainerUtils.euclidean_distance(normalized_actions, actions)
            average_action_distance = action_distance / num_pots
            action_distances.append(average_action_distance)

        # summarize results
        avg_action_distance = np.mean(action_distances)
        return {"iou_scores": iou_scores, "average_action_distance": avg_action_distance}

    @staticmethod
    def extract_json_from_string(input_string):
        """
        extract valid JSON part from string and convert to dictionary.

        Args:
            input_string (str): string containing extra characters.

        Returns:
            dict: dictionary extracted and parsed.
        """
        json_match = re.search(r"{.*}", input_string, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON decode failed: {e}")
                return None
        else:
            print("No valid JSON part found")
            return None

    def _get_latest_checkpoint(self, checkpoint_dir):
        """Find the latest checkpoint in the directory based on step number."""
        if not os.path.exists(checkpoint_dir):
            self.accelerator.print(f"No checkpoint directory found at {checkpoint_dir}")
            return None, 0

        checkpoints = [
            f for f in os.listdir(checkpoint_dir) 
            if re.match(r"steps_(\d+)_pytorch_model\.pt$", f)
            and os.path.isfile(os.path.join(checkpoint_dir, f))
        ]

        if not checkpoints:
            self.accelerator.print(f"No checkpoints found in {checkpoint_dir}")
            return None, 0

        try:
            checkpoints_with_steps = [
                (ckpt, int(re.search(r"steps_(\d+)_pytorch_model\.pt", ckpt).group(1)))
                for ckpt in checkpoints
            ]
        except AttributeError as e:
            self.accelerator.print(f"Error parsing checkpoint filenames: {e}")
            return None, 0

        checkpoints_with_steps.sort(key=lambda x: x[1])
        latest_checkpoint, completed_steps = checkpoints_with_steps[-1]

        latest_checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        self.accelerator.print(f"Latest checkpoint found: {latest_checkpoint_path}")
        return latest_checkpoint_path, completed_steps

import os


def is_main_process():
    rank = int(os.environ.get("RANK", 0))  # if RANK is not set, default to 0
    return rank == 0
