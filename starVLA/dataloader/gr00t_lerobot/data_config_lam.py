from __future__ import annotations

import copy
from typing import Sequence

from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.video import VideoTransform


def _normalize_state_keys(
    base_state_keys: Sequence[str],
    requested_state_keys: Sequence[str] | None,
) -> list[str]:
    if requested_state_keys is None:
        state_keys = list(base_state_keys)
    else:
        state_keys = list(requested_state_keys)

    if not state_keys:
        raise ValueError("No state keys provided for LAM state normalization.")

    for key in state_keys:
        if not key.startswith("state."):
            raise ValueError(f"Invalid state key '{key}'. Expected key prefix 'state.'.")

    base_set = set(base_state_keys)
    unknown_keys = [key for key in state_keys if key not in base_set]
    if unknown_keys:
        raise ValueError(
            f"Requested state keys are not present in base config: {unknown_keys}. "
            f"Available keys: {list(base_state_keys)}"
        )

    return state_keys


def filter_lam_video_keys(
    base_video_keys: Sequence[str],
    preferred_video_key: str | None = None,
    include_wrist_views: bool = False,
) -> list[str]:
    """Select the camera streams used by LAM, preserving configured order."""
    filtered_video_keys = [
        key for key in base_video_keys if include_wrist_views or "wrist" not in key.lower()
    ]
    if not filtered_video_keys:
        raise ValueError(
            "No compatible video keys available for LAM. "
            f"include_wrist_views={include_wrist_views}, received keys={list(base_video_keys)}"
        )

    if preferred_video_key is not None:
        if preferred_video_key in filtered_video_keys:
            # Keep all compatible keys, but place preferred key first.
            return [preferred_video_key] + [
                key for key in filtered_video_keys if key != preferred_video_key
            ]
        if preferred_video_key in base_video_keys:
            raise ValueError(
                f"Preferred video key '{preferred_video_key}' is excluded by the configured view policy. "
                f"Available keys: {filtered_video_keys}"
            )

    return filtered_video_keys


def build_lam_state_normalize_transform(
    robot_type: str,
    state_keys: Sequence[str] | None = None,
) -> ComposedModalityTransform:
    """
    Build a LAM-specific transform that only keeps state preprocessing.

    The transform is derived from the original robot config in ROBOT_TYPE_CONFIG_MAP,
    while stripping out video/action/concat transforms and preserving the original
    state preprocessing chain for the requested keys.
    """
    if robot_type not in ROBOT_TYPE_CONFIG_MAP:
        raise ValueError(
            f"Unknown robot_type '{robot_type}'. "
            f"Known robot types: {list(ROBOT_TYPE_CONFIG_MAP.keys())}"
        )

    base_cfg = ROBOT_TYPE_CONFIG_MAP[robot_type]
    base_modality_cfg = base_cfg.modality_config()
    if "state" not in base_modality_cfg:
        raise ValueError(f"Robot type '{robot_type}' has no state modality config.")

    filtered_state_keys = _normalize_state_keys(
        base_state_keys=base_modality_cfg["state"].modality_keys,
        requested_state_keys=state_keys,
    )

    base_transform = base_cfg.transform()
    if not isinstance(base_transform, ComposedModalityTransform):
        raise TypeError(
            "Base data config transform must be ComposedModalityTransform, "
            f"got {type(base_transform)} for robot_type '{robot_type}'."
        )

    state_transforms = []

    for transform in base_transform.transforms:
        apply_to = getattr(transform, "apply_to", None)
        if not apply_to:
            continue

        filtered_apply_to = [key for key in apply_to if key in filtered_state_keys]
        if not filtered_apply_to:
            continue

        transform_copy = copy.deepcopy(transform)
        transform_copy.apply_to = filtered_apply_to

        if isinstance(transform_copy, StateActionToTensor):
            transform_copy.input_dtypes = {
                key: dtype
                for key, dtype in transform_copy.input_dtypes.items()
                if key in filtered_state_keys
            }
            transform_copy.output_dtypes = {
                key: dtype
                for key, dtype in transform_copy.output_dtypes.items()
                if key in filtered_state_keys
            }
        elif isinstance(transform_copy, StateActionTransform):
            transform_copy.normalization_modes = {
                key: mode
                for key, mode in transform_copy.normalization_modes.items()
                if key in filtered_state_keys
            }
            transform_copy.target_rotations = {
                key: rotation
                for key, rotation in transform_copy.target_rotations.items()
                if key in filtered_state_keys
            }
            transform_copy.invert_normalized_keys = [
                key for key in transform_copy.invert_normalized_keys if key in filtered_state_keys
            ]

        state_transforms.append(transform_copy)

    if not state_transforms:
        raise ValueError(
            f"No state transforms found in base config for robot_type '{robot_type}'."
        )

    return ComposedModalityTransform(transforms=state_transforms)


def build_lam_dataset_transform(
    robot_type: str,
    image_hw: tuple[int, int],
    state_keys: Sequence[str] | None = None,
    *,
    video_keys: Sequence[str] | None = None,
    include_wrist_views: bool = False,
) -> ComposedModalityTransform:
    """
    Build the LAM training transform:
    - deterministic video preprocessing from the robot data_config
    - state tensor/normalization from the existing LAM helper

    `video_keys` is optional so callers can build a transform that matches the
    currently sampled view subset without assuming all configured views are present.
    """
    if robot_type not in ROBOT_TYPE_CONFIG_MAP:
        raise ValueError(
            f"Unknown robot_type '{robot_type}'. "
            f"Known robot types: {list(ROBOT_TYPE_CONFIG_MAP.keys())}"
        )

    base_cfg = ROBOT_TYPE_CONFIG_MAP[robot_type]
    base_modality_cfg = base_cfg.modality_config()
    if "video" not in base_modality_cfg:
        raise ValueError(f"Robot type '{robot_type}' has no video modality config.")

    base_video_keys = filter_lam_video_keys(
        base_modality_cfg["video"].modality_keys,
        include_wrist_views=include_wrist_views,
    )
    if video_keys is None:
        filtered_video_keys = list(base_video_keys)
    else:
        filtered_video_keys = list(video_keys)
        if not filtered_video_keys:
            raise ValueError("No video keys provided for LAM dataset transform.")
        unknown_video_keys = [key for key in filtered_video_keys if key not in base_video_keys]
        if unknown_video_keys:
            raise ValueError(
                f"Requested video keys are not present in the configured LAM view set: {unknown_video_keys}. "
                f"Available keys: {base_video_keys}"
            )

    base_transform = base_cfg.transform(image_hw=image_hw)
    if not isinstance(base_transform, ComposedModalityTransform):
        raise TypeError(
            "Base data config transform must be ComposedModalityTransform, "
            f"got {type(base_transform)} for robot_type '{robot_type}'."
        )

    video_transforms = []
    for transform in base_transform.transforms:
        if not isinstance(transform, VideoTransform):
            continue
        transform_copy = copy.deepcopy(transform)
        transform_copy.apply_to = [key for key in transform_copy.apply_to if key in filtered_video_keys]
        if transform_copy.apply_to:
            video_transforms.append(transform_copy)

    if not video_transforms:
        raise ValueError(
            f"No video transforms found for robot_type '{robot_type}' and video_keys={filtered_video_keys}."
        )

    state_transform = build_lam_state_normalize_transform(
        robot_type=robot_type,
        state_keys=state_keys,
    )

    return ComposedModalityTransform(
        transforms=[
            *video_transforms,
            *copy.deepcopy(state_transform.transforms),
        ]
    )
