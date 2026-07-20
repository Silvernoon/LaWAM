# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
In this file, we define 3 types of datasets:
1. LeRobotSingleDataset: a single dataset for a given embodiment tag
2. LeRobotMixtureDataset: a mixture of datasets for a given list of embodiment tags
3. CachedLeRobotSingleDataset: a single dataset for a given embodiment tag,
                                with caching for the video frames

See `scripts/load_dataset.py` for examples on how to use these datasets.
"""
import copy
import os
import fcntl
import hashlib
import json, torch
import multiprocessing as mp
import shutil
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.distributed as dist

from starVLA.dataloader.gr00t_lerobot.video import (
    _resize_frames_preserve_aspect,
    get_all_frames,
    get_frames_by_timestamps,
    get_frames_by_timestamps_single_seek,
)

from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EMBODIMENT_TAG_MAPPING, EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
)
from starVLA.dataloader.gr00t_lerobot.transform import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.video import VideoTransform


from datasets import Dataset as HFDataset


# LeRobot v3.0 dataset file names
LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats_gr00t.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.parquet"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes/*/*.parquet"
EPSILON = 5e-4
VIDEO_FRAME_CACHE_ROOT = Path("meta/cache/video_frame_cache")
VIDEO_FRAME_CACHE_MANIFEST_FILENAME = "manifest.json"
VIDEO_FRAME_CACHE_SCHEMA_VERSION = 2
DEFAULT_VIDEO_CACHE_BUILD_WORKERS = 8
DATASET_STAT_FIELDS = ("mean", "std", "min", "max", "q01", "q99")

LE_ROBOT_STANDARD_STATS_FILENAME = "meta/stats.json"


def _ensure_required_quantiles(statistics: dict[str, Any]) -> dict[str, Any]:
    """Normalize official LeRobot v3 stats to the fields used by GR00T transforms."""
    normalized = copy.deepcopy(statistics)
    for original_key in ("observation.state", "action"):
        field_stats = normalized.get(original_key)
        if not isinstance(field_stats, dict):
            continue
        if "min" in field_stats:
            field_stats.setdefault("q01", field_stats["min"])
        if "max" in field_stats:
            field_stats.setdefault("q99", field_stats["max"])
    return normalized


def _build_standard_lerobot_modality_metadata(
    info_meta: dict[str, Any],
    modality_configs: dict[str, Any],
    *,
    video_key_overrides: dict[str, str] | None = None,
) -> LeRobotModalityMetadata:
    """Build GR00T modality aliases from an official LeRobot v3 info.json."""
    features = info_meta.get("features", {})
    if not isinstance(features, dict):
        raise TypeError("Official LeRobot v3 `meta/info.json` must contain a `features` mapping.")

    metadata: dict[str, Any] = {"state": {}, "action": {}, "video": {}, "annotation": {}}
    original_vector_keys = {"state": "observation.state", "action": "action"}
    for modality, original_key in original_vector_keys.items():
        modality_config = modality_configs.get(modality)
        configured_keys = list(getattr(modality_config, "modality_keys", []))
        if not configured_keys:
            continue

        feature = features.get(original_key)
        if not isinstance(feature, dict):
            raise KeyError(
                f"Official LeRobot v3 dataset is missing `{original_key}` in info.json features."
            )
        shape = feature.get("shape", [])
        if len(shape) != 1:
            raise ValueError(
                f"Official LeRobot v3 `{original_key}` must be a flat vector, got shape={shape}."
            )
        feature_dim = int(shape[0])
        feature_names = feature.get("names")
        if not isinstance(feature_names, list) or len(feature_names) != feature_dim:
            feature_names = [f"dim_{index}" for index in range(feature_dim)]

        for configured_key in configured_keys:
            subkey = configured_key.split(".", 1)[1]
            if subkey not in feature_names:
                raise KeyError(
                    f"Configured key `{configured_key}` is absent from `{original_key}` names: "
                    f"{feature_names}."
                )
            feature_index = feature_names.index(subkey)
            metadata[modality][subkey] = {
                "start": feature_index,
                "end": feature_index + 1,
                "absolute": True,
                "dtype": str(feature.get("dtype", "float32")),
                "original_key": original_key,
            }

    video_features = [
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    video_config = modality_configs.get("video")
    configured_video_keys = list(getattr(video_config, "modality_keys", []))
    overrides = dict(video_key_overrides or {})
    assigned_original_keys: set[str] = set()
    for index, configured_key in enumerate(configured_video_keys):
        subkey = configured_key.split(".", 1)[1]
        original_key = overrides.get(configured_key, overrides.get(subkey))
        if original_key is None:
            direct_candidate = f"observation.images.{subkey}"
            if direct_candidate in video_features:
                original_key = direct_candidate
            elif index < len(video_features):
                original_key = video_features[index]
        if original_key not in video_features:
            raise KeyError(
                f"Cannot map configured video key `{configured_key}` to official LeRobot v3 features "
                f"{video_features}. Set `datasets.vla_data.standard_video_key_map` explicitly."
            )
        if original_key in assigned_original_keys:
            raise ValueError(
                f"Official LeRobot v3 video feature `{original_key}` was mapped more than once."
            )
        assigned_original_keys.add(original_key)
        metadata["video"][subkey] = {"original_key": original_key}

    language_config = modality_configs.get("language")
    for configured_key in list(getattr(language_config, "modality_keys", [])):
        subkey = configured_key.split(".", 1)[1]
        metadata["annotation"][subkey] = {"original_key": "task_index"}
    if not metadata["annotation"]:
        metadata["annotation"] = None

    return LeRobotModalityMetadata.model_validate(metadata)



def _is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def _resolve_video_cache_build_workers() -> int:
    default_workers = min(os.cpu_count() or 1, DEFAULT_VIDEO_CACHE_BUILD_WORKERS)
    raw_value = os.environ.get("LAM_VIDEO_CACHE_BUILD_WORKERS")
    if raw_value is None:
        return max(1, int(default_workers))

    try:
        resolved = int(raw_value)
    except (TypeError, ValueError):
        return max(1, int(default_workers))
    return max(1, resolved)


def _metadata_section_dim(modality_metadata: dict[str, Any], key_order: Sequence[str] | None = None) -> int:
    total_dim = 0
    keys = list(key_order) if key_order is not None else list(modality_metadata.keys())
    for key in keys:
        meta = modality_metadata.get(key)
        if meta is None:
            return -1
        shape = getattr(meta, "shape", None)
        if shape is None:
            return -1
        total_dim += int(np.prod(tuple(shape)))
    return total_dim


def _split_flat_statistics_by_metadata(
    section_stats: dict[str, Any],
    modality_metadata: dict[str, Any],
    key_order: Sequence[str] | None = None,
) -> dict[str, dict[str, list[float]]] | None:
    if not isinstance(section_stats, dict) or not modality_metadata:
        return None

    ordered_keys = list(key_order) if key_order is not None else list(modality_metadata.keys())
    total_dim = _metadata_section_dim(modality_metadata, ordered_keys)
    if total_dim < 0:
        return None

    for field in DATASET_STAT_FIELDS:
        values = section_stats.get(field)
        if values is None or len(values) != total_dim:
            return None

    split_stats: dict[str, dict[str, list[float]]] = {}
    offset = 0
    for key in ordered_keys:
        meta = modality_metadata[key]
        dim = int(np.prod(tuple(meta.shape)))
        split_stats[key] = {}
        for field in DATASET_STAT_FIELDS:
            values = list(section_stats[field])
            split_stats[key][field] = values[offset : offset + dim]
        offset += dim
    return split_stats


def _modality_key_order_from_configs(modality_configs: dict[str, Any]) -> dict[str, list[str]]:
    key_order: dict[str, list[str]] = {}
    for section in ("state", "action"):
        config = modality_configs.get(section)
        modality_keys = getattr(config, "modality_keys", None)
        if not modality_keys:
            continue
        key_order[section] = [
            str(key).split(".", 1)[1]
            for key in modality_keys
            if str(key).startswith(f"{section}.")
        ]
    return key_order


def _apply_dataset_statistics_override(
    metadata: DatasetMetadata,
    statistics_override: dict[str, Any] | None,
    *,
    tag: str,
    dataset_name: str,
    modality_key_order: dict[str, list[str]] | None = None,
) -> DatasetMetadata:
    if not statistics_override:
        return metadata

    tag_stats = statistics_override.get(str(tag))
    if not isinstance(tag_stats, dict):
        return metadata

    metadata_dict = metadata.model_dump(mode="json")
    replaced_sections: list[str] = []
    for section in ("state", "action"):
        section_stats = tag_stats.get(section)
        modality_metadata = getattr(metadata.modalities, section, {})
        split_stats = _split_flat_statistics_by_metadata(
            section_stats,
            modality_metadata,
            key_order=(modality_key_order or {}).get(section),
        )
        if split_stats is None:
            continue
        metadata_dict["statistics"][section] = split_stats
        replaced_sections.append(section)

    if not replaced_sections:
        print(
            f"Pretrained dataset statistics for tag={tag} are incompatible with "
            f"dataset={dataset_name}; using current dataset statistics."
        )
        return metadata

    print(
        f"Using pretrained dataset statistics for dataset={dataset_name} "
        f"tag={tag} sections={replaced_sections}."
    )
    return DatasetMetadata.model_validate(metadata_dict)


def _normalize_cached_video_frames_static(
    frames: np.ndarray,
    expected_length: int,
    video_key: str,
    trajectory_id: int,
    video_path: str | Path,
    dataset_name: str,
) -> np.ndarray:
    video_path = Path(video_path)
    if frames.ndim != 4:
        raise ValueError(f"Expected 4D array for {video_key}, got shape={frames.shape}")
    if frames.shape[3] != 3:
        raise ValueError(f"Expected 3 channels for {video_key}, got shape={frames.shape}")
    if frames.shape[0] == 0:
        raise ValueError(
            f"Video has no frames for dataset={dataset_name} trajectory={trajectory_id} path={video_path}"
        )
    if frames.shape[0] > expected_length:
        frames = frames[:expected_length]
    elif frames.shape[0] < expected_length:
        pad_count = expected_length - frames.shape[0]
        pad = np.repeat(frames[-1:], pad_count, axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    return np.asarray(frames, dtype=np.uint8)


def _decode_video_for_cache_task(
    video_path: str,
    trajectory_id: int,
    trajectory_index: int,
    expected_length: int,
    video_key: str,
    dataset_name: str,
    video_backend: str,
    video_backend_kwargs: dict[str, Any],
    resize_short_side: int | None,
    video_timestamps: list[float] | None = None,
) -> dict[str, Any]:
    if video_timestamps is not None:
        frames = get_frames_by_timestamps_single_seek(
            video_path,
            video_timestamps,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
        )
        frames = _resize_frames_preserve_aspect(frames, resize_short_side)
    else:
        frames = get_all_frames(
            video_path,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
            resize_size=resize_short_side,
        )
    normalized = _normalize_cached_video_frames_static(
        frames=frames,
        expected_length=int(expected_length),
        video_key=video_key,
        trajectory_id=int(trajectory_id),
        video_path=video_path,
        dataset_name=dataset_name,
    )
    return {
        "trajectory_id": int(trajectory_id),
        "trajectory_index": int(trajectory_index),
        "expected_length": int(expected_length),
        "frames": normalized,
        "frame_shape": tuple(int(x) for x in normalized.shape[1:]),
    }



def _resolve_embodiment_id_from_tag(tag: str) -> int:
    if not isinstance(tag, str):
        raise TypeError(f"`embodiment_tag` must be str, got {type(tag)}.")
    if tag not in EMBODIMENT_TAG_MAPPING:
        raise ValueError(
            f"Unknown embodiment tag `{tag}`. "
            f"Available tags: {sorted(EMBODIMENT_TAG_MAPPING.keys())}."
        )
    mapped_value = EMBODIMENT_TAG_MAPPING[tag]
    if isinstance(mapped_value, bool) or not isinstance(mapped_value, int):
        raise TypeError(
            f"`EMBODIMENT_TAG_MAPPING[{tag}]` must be int, got {type(mapped_value)}."
        )
    return int(mapped_value)


class _FileLock:
    def __init__(self, lock_path: Path) -> None:
        self.lock_path = Path(lock_path)
        self._handle = None

    def __enter__(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.lock_path, "a+")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


def calculate_dataset_statistics(parquet_paths: list[Path]) -> dict:
    """Calculate the dataset statistics of all columns for a list of parquet files."""
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    # parquet_paths = parquet_paths[:3]
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)
    
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    for le_modality in tqdm(all_low_dim_data.columns, desc="Processing modalities"):
        print(le_modality)
        if "task_info" in le_modality:
            continue
        print(f"Computing statistics for {le_modality}...")
        try:
            np_data = np.vstack(
                [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
            )
        except Exception as e:
            print(f"Warning: Failed to process modality {le_modality} due to error: {e}")
            continue  

        dataset_statistics[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
    return dataset_statistics


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class LeRobotSingleDataset(Dataset):
    """
    Base dataset class for LeRobot that supports sharding.
    """
    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        mode: str = "all",
        _val_tail_ratio: float = 0.001,
        video_backend: str = "pyav",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        data_cfg = None,
        action_hz: float | None = None,
        dataset_statistics_override: dict[str, Any] | None = None,
        _force_recompute_stats: bool = False,
        **kwargs,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (Path | str): The path to the dataset.
            modality_configs (dict[str, ModalityConfig]): The configuration for each modality. The keys are the modality names, and the values are the modality configurations.
                See `ModalityConfig` for more details.
            video_backend (str): Backend for video reading.
            video_backend_kwargs (dict): Keyword arguments for the video backend when initializing the video reader.
            transforms (ComposedModalityTransform): The transforms to apply to the dataset.
            embodiment_tag (EmbodimentTag): Overload the embodiment tag for the dataset. e.g. define it as "new_embodiment"
        """
        # first check if the path directory exists
        self.data_cfg = data_cfg
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
        # Internal switch: recompute stats at runtime and overwrite cached file.
        # Priority: explicit init arg > data_cfg.force_recompute_stats > default(False)
        self._force_recompute_stats = bool(_force_recompute_stats)
        if hasattr(self.data_cfg, "get"):
            self._force_recompute_stats = bool(
                self.data_cfg.get("force_recompute_stats", self._force_recompute_stats)
            )
        self.modality_configs = modality_configs
        self._action_hz_hint = float(action_hz) if action_hz is not None else None
        self.video_backend = video_backend
        self.video_backend_kwargs = dict(video_backend_kwargs) if video_backend_kwargs is not None else {}
        # Never trust manually provided fixed_fps: always derive per-dataset fps from info.json metadata.
        if "fixed_fps" in self.video_backend_kwargs:
            self.video_backend_kwargs.pop("fixed_fps", None)
            print(
                "Ignore manual video_backend_kwargs['fixed_fps']; "
                "will use fps from dataset info.json automatically."
            )
        self.transforms = (
            transforms if transforms is not None else ComposedModalityTransform(transforms=[])
        )

        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name
        normalized_mode = str(mode).lower()
        if normalized_mode not in {"train", "val", "test", "all"}:
            raise ValueError(f"Unsupported dataset mode `{mode}`. Expected one of ['train', 'val', 'test'].")
        self._split_mode = "val" if normalized_mode == "test" else normalized_mode
        self._val_tail_ratio = float(_val_tail_ratio)
        if isinstance(embodiment_tag, EmbodimentTag):
            self.tag = embodiment_tag.value
        else:
            self.tag = embodiment_tag
        self.embodiment_id = _resolve_embodiment_id_from_tag(self.tag)

        self._metadata = _apply_dataset_statistics_override(
            self._get_metadata(EmbodimentTag(self.tag)),
            dataset_statistics_override,
            tag=str(self.tag),
            dataset_name=self._dataset_name,
            modality_key_order=_modality_key_order_from_configs(self.modality_configs),
        )

        # LeRobot-specific config
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        self._data_path_pattern = self._get_data_path_pattern()
        self._video_path_pattern = self._get_video_path_pattern()
        self._chunk_size = self._get_chunk_size()
        self._task_index_to_task: dict[int, str] = {}
        self._tasks = self._get_tasks()
        # self._episodes = self._get_episode_info() # TODO why we need this func
        self.curr_traj_data = None
        self.curr_traj_id = None
        self._curr_episode_id: int | None = None
        self._curr_row_idx: int | None = None
        self._curr_from_index: int = 0
        self._curr_to_index: int = 0
        self._curr_length: int = 0
        self._curr_episode_row: dict | None = None

        self._load_episodes_hf()
        self._load_hf_dataset()

        full_trajectory_ids, full_trajectory_lengths = self._get_trajectories()
        self._all_trajectory_ids = full_trajectory_ids
        self._all_trajectory_lengths = full_trajectory_lengths
        self._build_mode_split_from_trajectories()
        self._build_active_step_indexing()
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self._action_hz = self._resolve_action_hz()
        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)

        print(
            f"Initialized dataset {self.dataset_name} with {embodiment_tag} "
            f"(mode={self._split_mode}, val_tail_ratio={self._val_tail_ratio:.4f}, "
            f"active_trajectories={len(self._trajectory_ids)}, active_steps={self._subset_total_steps})"
        )


        # Check if the dataset is valid
        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths

    @property
    def modality_keys(self) -> dict:
        """The modality keys for the dataset. The keys are the modality names, and the values are the keys for each modality.

        Example: {
            "video": ["video.image_side_0", "video.image_side_1"],
            "state": ["state.eef_position", "state.eef_rotation"],
            "action": ["action.eef_position", "action.eef_rotation"],
            "language": ["language.human.task"],
            "timestamp": ["timestamp"],
            "reward": ["reward"],
        }
        """
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset. The keys are the modality.key, and the values are the delta indices for each modality.key."""
        return self._delta_indices

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def action_hz(self) -> float:
        """Per-dataset control frequency (Hz) for action timestamps."""
        return float(self._action_hz)

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Returns:
            dict: The metadata for the dataset.
        """

        # Get used keys from modality_configs (if available) to filter metadata
        used_keys = {}
        if hasattr(self, 'modality_configs') and self.modality_configs:
            for modality, config in self.modality_configs.items():
                if hasattr(config, 'modality_keys'):
                    # Extract subkey from full key (e.g., "state.eef_position" -> "eef_position")
                    used_keys[modality] = set()
                    for full_key in config.modality_keys:
                        parts = full_key.split(".", 1)
                        if len(parts) == 2:
                            used_keys[modality].add(parts[1])
                        else:
                            used_keys[modality].add(full_key)

        # 1. Modality metadata
        simplified_modality_meta: dict[str, dict] = {}
        le_modality_meta = self._get_lerobot_modality_meta()
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            
            # If modality_configs is provided but this modality is not in it, skip entirely
            if used_keys and modality not in used_keys:
                continue
                
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(
                le_modality_meta, modality
            )
            for subkey in le_state_action_meta:
                # Filter: only include keys that are used in modality_configs
                if modality in used_keys and subkey not in used_keys[modality]:
                    continue
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [
                        le_state_action_meta[subkey].end - le_state_action_meta[subkey].start
                    ],
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert (
            le_info_path.exists()
        ), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        info_features = le_info.get("features", {})
        for new_key in le_modality_meta.video:
            # Filter: only include video keys that are used in modality_configs
            if "video" in used_keys and new_key not in used_keys["video"]:
                continue
            original_key = le_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = info_features.get(original_key)
            if le_video_meta is None:
                # Fallback: use first available video feature from info (e.g. dataset has image/image2 but info lists one)
                for k, v in info_features.items():
                    if v.get("dtype") == "video":
                        le_video_meta = v
                        break
            if le_video_meta is None:
                raise KeyError(
                    f"Video key '{original_key}' not in info.json features {list(info_features.keys())}. "
                    "Ensure meta/info.json features match modality.json video keys."
                )
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            # NOTE(FH): different lerobot dataset versions have different keys for the number of channels and fps
            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
                fps = le_video_meta["video_info"]["video.fps"]
            except (ValueError, KeyError):
                channels = le_video_meta.get("info", {}).get("video.channels", 3)
                fps = le_video_meta.get("info", {}).get("video.fps", le_video_meta.get("video_info", {}).get("video.fps", 30))
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }


        # 2. Dataset statistics
        def is_main():
            return (not dist.is_initialized()) or dist.get_rank() == 0
        
        stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        standard_stats_path = self.dataset_path / LE_ROBOT_STANDARD_STATS_FILENAME
        precomputed_stats_path = stats_path
        if not stats_path.exists() and standard_stats_path.exists():
            precomputed_stats_path = standard_stats_path
        tmp_path = stats_path.with_suffix(".tmp")

        # ---------- all rank try to read ----------
        if self._force_recompute_stats:
            le_statistics = None
            if is_main():
                print(
                    f"[RANK 0] Force recompute enabled, rebuilding dataset statistics for "
                    f"{self.dataset_name} and overwriting {stats_path}"
                )
        else:
            if precomputed_stats_path.exists():
                try:
                    with open(precomputed_stats_path, "r") as f:
                        le_statistics = _ensure_required_quantiles(json.load(f))
                    required_original_keys = {
                        meta.original_key
                        for modality in ("state", "action")
                        for meta in getattr(le_modality_meta, modality).values()
                    }
                    for original_key in required_original_keys:
                        DatasetStatisticalValues.model_validate(le_statistics[original_key])
                except Exception as e:
                    print(
                        f"[RANK {os.environ.get('RANK', 'NA')}] "
                        f"Failed to load dataset statistics ({e}), rebuilding..."
                    )
                    le_statistics = None
            else:
                le_statistics = None
        
        # ---------- rank0 build ----------
        if le_statistics is None and is_main():
            print(f"[RANK 0] Calculating dataset statistics for {self.dataset_name}")
        
            parquet_files = list(self.dataset_path.glob(LE_ROBOT_DATA_FILENAME))
            parquet_files_filtered = [
                pf for pf in parquet_files if "episode_033675.parquet" not in pf.name
            ]
        
            le_statistics = _ensure_required_quantiles(
                calculate_dataset_statistics(parquet_files_filtered)
            )
        
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w") as f:
                json.dump(le_statistics, f, indent=4)
            os.replace(tmp_path, stats_path)
        
            print(f"[RANK 0] Dataset statistics cached to {stats_path}")
        
        # ---------- sync ----------
        if dist.is_initialized():
            dist.barrier()
        
        # ---------- all rank read again ----------
        if le_statistics is None:
            with open(stats_path, "r") as f:
                le_statistics = json.load(f)

        dataset_statistics = {}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = le_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                le_modality = state_action_meta.original_key
                for stat_name in DATASET_STAT_FIELDS:
                    indices = np.arange(
                        state_action_meta.start,
                        state_action_meta.end,
                    )
                    stat = np.array(le_statistics[le_modality][stat_name])
                    dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

        return metadata

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        if not hasattr(self, "episodes_hf") or self.episodes_hf is None:
            raise RuntimeError("LeRobot v3.0 requires _load_episodes_hf() to be called before _get_trajectories()")
        trajectory_ids = []
        trajectory_lengths = []
        # Always build mapping: episode_index -> row index
        self._episode_index_to_row: dict[int, int] = {}
        self._trajectory_id_to_index: dict[int, int] = {}
        for i in range(len(self.episodes_hf)):
            row = self.episodes_hf[i]
            ep_idx = int(row["episode_index"])
            self._episode_index_to_row[ep_idx] = i
            self._trajectory_id_to_index[ep_idx] = len(trajectory_ids)
            trajectory_ids.append(ep_idx)  # Use episode_index column value (semantic ID)
            trajectory_lengths.append(int(row["length"]))
        print(f"[INFO] Built episode_index->row mapping ({len(self._episode_index_to_row)} episodes)")

        # Detect video keys from episode columns (e.g. videos/observation.images.image/from_timestamp)
        self._detected_video_keys = []
        for col in self.episodes_hf.column_names:
            if col.startswith("videos/") and col.endswith("/from_timestamp"):
                vid_key = col.replace("videos/", "").replace("/from_timestamp", "")
                self._detected_video_keys.append(vid_key)
        return np.array(trajectory_ids), np.array(trajectory_lengths)

    def _build_mode_split_from_trajectories(self) -> None:
        """Build deterministic train/val split using trajectory order and tail ratio."""
        total_traj = int(len(self._all_trajectory_ids))
        if total_traj == 0:
            self._active_episode_indices = np.array([], dtype=np.int64)
            self._active_traj_ids = np.array([], dtype=np.int64)
            self._trajectory_ids = self._active_traj_ids
            self._trajectory_lengths = np.array([], dtype=np.int64)
            self._trajectory_id_to_index_active = {}
            return

        n_val_base = int(np.floor(total_traj * self._val_tail_ratio))
        if total_traj >= 2:
            n_val = max(1, n_val_base)
            n_val = min(n_val, total_traj - 1)
        else:
            # Keep single-trajectory dataset in train set.
            n_val = 0

        split_at = total_traj - n_val
        if self._split_mode == "train":
            active_slice = slice(0, split_at)
        elif self._split_mode == "val":
            active_slice = slice(split_at, total_traj)
        elif self._split_mode == "all":
            active_slice = slice(0, total_traj)
        else:
            raise ValueError(f"Invalid split mode: {self._split_mode}")

        active_ids = self._all_trajectory_ids[active_slice].astype(np.int64, copy=False)
        active_lengths = self._all_trajectory_lengths[active_slice].astype(np.int64, copy=False)

        self._active_episode_indices = active_ids
        self._active_traj_ids = active_ids
        self._trajectory_ids = active_ids
        self._trajectory_lengths = active_lengths
        self._trajectory_id_to_index_active = {
            int(traj_id): idx for idx, traj_id in enumerate(self._trajectory_ids.tolist())
        }

    def _build_active_step_indexing(self) -> None:
        """Build local-step indexing arrays for fast local->absolute index mapping."""
        if len(self._active_episode_indices) == 0:
            self._subset_from_indices = np.array([], dtype=np.int64)
            self._subset_lengths = np.array([], dtype=np.int64)
            self._subset_cum_lengths = np.array([], dtype=np.int64)
            self._subset_total_steps = 0
            return

        from_indices = []
        lengths = []
        for ep_idx in self._active_episode_indices.tolist():
            row_idx = self._episode_index_to_row[int(ep_idx)]
            row = self.episodes_hf[row_idx]
            from_indices.append(int(row["dataset_from_index"]))
            lengths.append(int(row["length"]))

        self._subset_from_indices = np.asarray(from_indices, dtype=np.int64)
        self._subset_lengths = np.asarray(lengths, dtype=np.int64)
        self._subset_cum_lengths = np.cumsum(self._subset_lengths, dtype=np.int64)
        self._subset_total_steps = int(self._subset_cum_lengths[-1]) if len(self._subset_cum_lengths) > 0 else 0

    def _local_to_abs_index(self, local_idx: int) -> int:
        """Map mode-local step index to absolute hf_dataset index."""
        if self._subset_total_steps <= 0:
            raise IndexError(
                f"Dataset split `{self._split_mode}` has no available steps for dataset `{self.dataset_name}`."
            )
        idx = int(local_idx)
        if idx < 0:
            idx += self._subset_total_steps
        if idx < 0 or idx >= self._subset_total_steps:
            raise IndexError(
                f"Index {local_idx} out of range for dataset `{self.dataset_name}` split `{self._split_mode}` "
                f"with length {self._subset_total_steps}."
            )

        traj_local_idx = int(np.searchsorted(self._subset_cum_lengths, idx, side="right"))
        prev_cum = 0 if traj_local_idx == 0 else int(self._subset_cum_lengths[traj_local_idx - 1])
        offset_in_traj = idx - prev_cum
        return int(self._subset_from_indices[traj_local_idx] + offset_in_traj)

    def abs_index_to_episode_step(self, abs_idx: int) -> tuple[int, int]:
        """Convert a mode-local frame index to (trajectory_id, base_index).
        
        This follows the official LeRobot v3 approach: use hf_dataset absolute index
        and map it to episode-relative index using episode metadata.
        
        Args:
            abs_idx: Mode-local index into active subset (0 <= abs_idx < len(self))
            
        Returns:
            tuple[int, int]: (trajectory_id, base_index) where base_index is 
                the step's position within the trajectory.
        """
        abs_idx = self._local_to_abs_index(int(abs_idx))
        item = self.hf_dataset[abs_idx]
        trajectory_id = int(item["episode_index"])
        try:
            row_idx = self._episode_index_to_row[trajectory_id]
        except KeyError as exc:
            mapped_episode_ids = list(self._episode_index_to_row.keys())
            mapped_min = min(mapped_episode_ids) if mapped_episode_ids else None
            mapped_max = max(mapped_episode_ids) if mapped_episode_ids else None
            raise KeyError(
                "Episode index mapping mismatch detected. "
                f"dataset_name={self.dataset_name}, dataset_path={self.dataset_path}, "
                f"abs_idx={abs_idx}, episode_index_from_data={trajectory_id}, "
                f"hf_dataset_len={len(self.hf_dataset)}, episodes_hf_len={len(self.episodes_hf)}, "
                f"mapped_episode_count={len(mapped_episode_ids)}, "
                f"mapped_episode_min={mapped_min}, mapped_episode_max={mapped_max}, "
                f"mapped_episode_samples={mapped_episode_ids[:10]}. "
                "This usually means `data/*.parquet` and `meta/episodes/*.parquet` are inconsistent "
                "or loaded from different dataset versions."
            ) from exc
        from_index = int(self.episodes_hf[row_idx]["dataset_from_index"])
        return trajectory_id, abs_idx - from_index

    def _get_position_and_gripper_values(self, data: pd.DataFrame) -> tuple[list, list]:
        """Get position and gripper values based on available columns in the dataset."""
        # Get action keys from modality_keys
        action_keys = self.modality_keys.get('action', [])
        
        # Extract position data
        delta_position_values = None
        position_candidates = ['delta_eef_position']
        coordinate_candidates = ['x', 'y', 'z']
        
        # First try combined position fields
        for pos_key in position_candidates:
            full_key = f"action.{pos_key}"
            if full_key in action_keys:
                try:
                    # Get the lerobot key for this modality
                    le_action_cfg = self.lerobot_modality_meta.action
                    subkey = pos_key
                    if subkey in le_action_cfg:
                        le_key = le_action_cfg[subkey].original_key or subkey
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[subkey].start, le_action_cfg[subkey].end)
                            filtered_data = data_array[:, le_indices]
                            delta_position_values = filtered_data.tolist()
                            break
                except Exception:
                    continue
        
        # If combined fields not found, try individual x,y,z coordinates
        if delta_position_values is None:
            x_data, y_data, z_data = None, None, None
            for coord in coordinate_candidates:
                full_key = f"action.{coord}"
                if full_key in action_keys:
                    try:
                        le_action_cfg = self.lerobot_modality_meta.action
                        if coord in le_action_cfg:
                            le_key = le_action_cfg[coord].original_key or coord
                            if le_key in data.columns:
                                data_array = np.stack(data[le_key])
                                le_indices = np.arange(le_action_cfg[coord].start, le_action_cfg[coord].end)
                                coord_data = data_array[:, le_indices].flatten()
                                if coord == 'x':
                                    x_data = coord_data
                                elif coord == 'y':
                                    y_data = coord_data
                                elif coord == 'z':
                                    z_data = coord_data
                    except Exception:
                        continue
            
            if x_data is not None and y_data is not None and z_data is not None:
                delta_position_values = np.column_stack((x_data, y_data, z_data)).tolist()
        
        if delta_position_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if 'action.delta_eef_position' in data.columns:
                delta_position_values = data['action.delta_eef_position'].to_numpy().tolist()
            elif all(col in data.columns for col in ['action.x', 'action.y', 'action.z']):
                x_vals = data['action.x'].to_numpy()
                y_vals = data['action.y'].to_numpy() 
                z_vals = data['action.z'].to_numpy()
                delta_position_values = np.column_stack((x_vals, y_vals, z_vals)).tolist()
            else:
                raise ValueError(f"No suitable position columns found. Available columns: {data.columns.tolist()}")
        
        # Extract gripper data
        gripper_values = None
        gripper_candidates = ['gripper_close', 'gripper']
        
        for grip_key in gripper_candidates:
            full_key = f"action.{grip_key}"
            if full_key in action_keys:
                try:
                    le_action_cfg = self.lerobot_modality_meta.action
                    if grip_key in le_action_cfg:
                        le_key = le_action_cfg[grip_key].original_key or grip_key
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[grip_key].start, le_action_cfg[grip_key].end)
                            gripper_data = data_array[:, le_indices].flatten()
                            gripper_values = gripper_data.tolist()
                            break
                except Exception:
                    continue
        
        if gripper_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if 'action.gripper_close' in data.columns:
                gripper_values = data['action.gripper_close'].to_numpy().tolist()
            elif 'action.gripper' in data.columns:
                gripper_values = data['action.gripper'].to_numpy().tolist()
            else:
                raise ValueError(f"No suitable gripper columns found. Available columns: {data.columns.tolist()}")
        
        return delta_position_values, gripper_values

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.
        The keys are the modality names, and the values are the keys for each modality.
        See property `modality_keys` for the expected format.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _resolve_action_hz(self) -> float:
        """Resolve per-dataset action/control frequency from hint or dataset metadata."""
        if self._action_hz_hint is not None:
            if self._action_hz_hint <= 0:
                raise ValueError(
                    f"Invalid action_hz hint for dataset={self.dataset_name}: {self._action_hz_hint}"
                )
            return float(self._action_hz_hint)

        info_meta = self.lerobot_info_meta if isinstance(self.lerobot_info_meta, dict) else {}
        try:
            dataset_fps = float(info_meta.get("fps"))
            if dataset_fps > 0:
                return dataset_fps
        except (TypeError, ValueError):
            pass

        features = info_meta.get("features", {}) if isinstance(info_meta.get("features", {}), dict) else {}
        action_feature = features.get("action", {}) if isinstance(features.get("action", {}), dict) else {}
        try:
            action_fps = float(action_feature.get("fps"))
            if action_fps > 0:
                return action_fps
        except (TypeError, ValueError):
            pass

        fps_candidates: list[float] = []
        for key in self.modality_keys.get("video", []):
            meta_key = key.replace("video.", "", 1) if key.startswith("video.") else key
            if meta_key in self.metadata.modalities.video:
                fps = float(self.metadata.modalities.video[meta_key].fps)
                if fps > 0:
                    fps_candidates.append(fps)

        if not fps_candidates:
            for video_meta in self.metadata.modalities.video.values():
                fps = float(video_meta.fps)
                if fps > 0:
                    fps_candidates.append(fps)

        if not fps_candidates:
            fallback = 10.0
            print(
                f"[LeRobotDataset] dataset={self.dataset_name} cannot resolve action_hz from metadata; "
                f"fallback to {fallback} Hz."
            )
            return fallback

        if len(fps_candidates) > 1:
            unique_fps = sorted({round(v, 6) for v in fps_candidates})
            if len(unique_fps) > 1:
                print(
                    f"[LeRobotDataset] dataset={self.dataset_name} has multiple video fps={unique_fps}; "
                    f"use first={fps_candidates[0]} as action_hz."
                )
        return float(fps_candidates[0])

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Load GR00T modality metadata or derive it from official LeRobot v3 metadata."""
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        if modality_meta_path.exists():
            with open(modality_meta_path, "r") as f:
                return LeRobotModalityMetadata.model_validate(json.load(f))

        raw_overrides = {}
        if self.data_cfg is not None:
            if hasattr(self.data_cfg, "get"):
                raw_overrides = self.data_cfg.get("standard_video_key_map", {}) or {}
            else:
                raw_overrides = getattr(self.data_cfg, "standard_video_key_map", {}) or {}
        video_key_overrides = (
            {str(key): str(value) for key, value in raw_overrides.items()}
            if hasattr(raw_overrides, "items")
            else {}
        )
        return _build_standard_lerobot_modality_metadata(
            self._get_lerobot_info_meta(),
            self.modality_configs,
            video_key_overrides=video_key_overrides,
        )

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        """Get the video path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _load_episodes_hf(self) -> None:
        """Official LeRobot v3.0: load episode metadata as HuggingFace Dataset."""
        episodes_dir = self.dataset_path / "meta" / "episodes"
        paths = sorted(episodes_dir.glob("*/*.parquet"))
        # if not paths:
        #     raise FileNotFoundError(f"No episode parquet files in {episodes_dir}")
        
        # # Sort by numeric chunk and file indices
        # import re
        # def _parse_path_indices(p: Path) -> tuple[int, int]:
        #     chunk_dir = p.parent.name
        #     file_name = p.stem
        #     chunk_match = re.search(r'(\d+)', chunk_dir)
        #     chunk_idx = int(chunk_match.group(1)) if chunk_match else 0
        #     file_match = re.search(r'(\d+)', file_name)
        #     file_idx = int(file_match.group(1)) if file_match else 0
        #     return (chunk_idx, file_idx)
        
        # paths = sorted(paths, key=_parse_path_indices)
        
        self.episodes_hf = HFDataset.from_parquet([str(p) for p in paths])
        # Drop stats/ columns like official load_episodes
        cols = [k for k in self.episodes_hf.column_names if not k.startswith("stats/")]
        self.episodes_hf = self.episodes_hf.select_columns(cols)
        # print(f"[INFO] Loaded {len(self.episodes_hf)} episodes (HF Dataset)")

    def _load_hf_dataset(self) -> None:
        """Official LeRobot v3.0: load all data parquet files as single HuggingFace Dataset (global index)."""
        data_dir = self.dataset_path / "data"
        paths = list(data_dir.glob("*/*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No data parquet files in {data_dir}")
        
        # Sort by numeric chunk and file indices to match episodes metadata order
        # Handles both "chunk-000/file-000.parquet" and "0/0000.parquet" formats
        def _parse_path_indices(p: Path) -> tuple[int, int]:
            """Extract (chunk_index, file_index) from path for numeric sorting."""
            import re
            chunk_dir = p.parent.name  # e.g., "chunk-000" or "0"
            file_name = p.stem  # e.g., "file-000" or "0000"
            
            # Extract numbers from chunk dir name
            chunk_match = re.search(r'(\d+)', chunk_dir)
            chunk_idx = int(chunk_match.group(1)) if chunk_match else 0
            
            # Extract numbers from file name
            file_match = re.search(r'(\d+)', file_name)
            file_idx = int(file_match.group(1)) if file_match else 0
            
            return (chunk_idx, file_idx)
        
        # Sort paths by (chunk_index, file_index) numerically
        paths = sorted(paths, key=_parse_path_indices)
        
        self.hf_dataset = HFDataset.from_parquet([str(p) for p in paths])
        print(f"[INFO] Loaded HF dataset with {len(self.hf_dataset)} frames")

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
        df = pd.read_parquet(tasks_path)
        if "task_index" not in df.columns:
            index_name = df.index.name if df.index.name is not None else "index"
            df = df.reset_index().rename(columns={index_name: "task_index"})

        if "task" not in df.columns:
            # Some datasets (e.g. LIBERO exports) store task text in the row index.
            if not isinstance(df.index, pd.RangeIndex):
                df = df.copy()
                df["task"] = df.index.astype(str)
            if "task" not in df.columns:
                for fallback_col in ("task_name", "instruction", "text"):
                    if fallback_col in df.columns:
                        df = df.rename(columns={fallback_col: "task"})
                        break

        if "task_index" not in df.columns or "task" not in df.columns:
            raise KeyError(
                f"Invalid tasks schema in {tasks_path}, expected columns ['task_index', 'task'], got {list(df.columns)}"
            )

        df = df[["task_index", "task"]].copy()
        df["task_index"] = df["task_index"].astype(int)
        df["task"] = df["task"].fillna("").astype(str)
        df = df.drop_duplicates(subset="task_index", keep="first")
        self._task_index_to_task = {
            int(task_index): task
            for task_index, task in zip(df["task_index"], df["task"])
        }
        return df

    def _check_integrity(self):
        """Use the config to check if the keys are valid and detect silent data corruption."""
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        for modality_config in self.modality_configs.values():
            for key in modality_config.modality_keys:
                if key == "lapa_action" or key == "dream_actions":
                    continue  # no need for any metadata for lapa actions because it comes normalized
                # Check if the key is valid
                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(
                        ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}"
                    )

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        self.transforms.set_metadata(metadata)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Get the total number of data points in the dataset.

        Returns:
            int: the total number of data points in the dataset.
        """
        if self.hf_dataset is None:
            raise RuntimeError("hf_dataset is not loaded. Ensure _load_hf_dataset() ran successfully.")
        return self._subset_total_steps

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"


    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        if self.hf_dataset is None:
            raise RuntimeError("hf_dataset is not loaded. Ensure _load_hf_dataset() ran successfully.")
        index = self._local_to_abs_index(int(index))
        item = self.hf_dataset[index]
        trajectory_id = int(item["episode_index"])
        # Calculate base_index within episode using dataset_from_index
        row_idx = self._episode_index_to_row[trajectory_id]
        ep = self.episodes_hf[row_idx]
        from_index = int(ep["dataset_from_index"])
        base_index = index - from_index
        data = self.get_step_data(trajectory_id, base_index)
        
        # Process all video keys dynamically
        images = []
        videos = []
        for video_key in self.modality_keys["video"]:
            view_frames = data[video_key]
            if view_frames.ndim != 4:
                raise ValueError(
                    f"Expected video array shape [T, H, W, C] for key={video_key}, got {view_frames.shape}"
                )

            images.append(view_frames[0])
            videos.append(list(view_frames))
        
        # Get language and action data
        language = data[self.modality_keys["language"][0]][0]
        action = []
        for action_key in self.modality_keys["action"]:
            action.append(data[action_key])
        action = np.concatenate(action, axis=1)
        
        return dict(
            action=action,
            image=images,
            video=videos,
            language=language,
            embodiment_id=int(self.embodiment_id),
            action_hz=float(self.action_hz),
        )

    def get_step_data(
        self,
        trajectory_id: int,
        base_index: int,
        modality_keys_override: dict[str, list[str]] | None = None,
    ) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            base_index (int): The base step index in the trajectory.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data: dict = {}
        self._set_curr_episode(trajectory_id)
        modality_keys = {modality: list(keys) for modality, keys in self.modality_keys.items()}
        if modality_keys_override:
            for modality, keys in modality_keys_override.items():
                if modality in modality_keys:
                    modality_keys[modality] = list(keys)
        get_data = self.get_data_by_modality
        for modality, keys in modality_keys.items():
            for key in keys:
                data[key] = get_data(trajectory_id, modality, key, base_index)
        return data

    
    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory from lerobot v3 (official: slice of HF Dataset)."""
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        # Always use mapping to find correct row
        row_idx = self._episode_index_to_row[trajectory_id]
        ep = self.episodes_hf[row_idx]
        from_index = int(ep["dataset_from_index"])
        to_index = int(ep["dataset_to_index"])
        
        # Validate indices are within hf_dataset bounds
        hf_len = len(self.hf_dataset)
        if from_index < 0 or from_index >= hf_len or to_index > hf_len:
            raise IndexError(
                f"Episode {trajectory_id} has invalid indices: "
                f"dataset_from_index={from_index}, dataset_to_index={to_index}, "
                f"but hf_dataset has only {hf_len} frames. "
                f"This may indicate data/episode metadata mismatch or file sorting issue."
            )
        
        slice_ds = self.hf_dataset.select(range(from_index, to_index))
        episode_data = slice_ds.to_pandas()
        # Timestamps in data are relative to episode; add video from_timestamp for frame lookup
        if getattr(self, "_detected_video_keys", None):
            first_video_key = self._detected_video_keys[0]
            from_ts_col = f"videos/{first_video_key}/from_timestamp"
            if from_ts_col in ep:
                from_ts = float(ep[from_ts_col])
                if "timestamp" in episode_data.columns:
                    ts = episode_data["timestamp"]
                    # HF may store as list/array per row; flatten to 1d
                    if hasattr(ts.iloc[0], "__len__") and not isinstance(ts.iloc[0], (str, bytes)):
                        ts = np.array([float(x[0]) if len(x) else float(x) for x in ts])
                    else:
                        ts = ts.to_numpy().astype(np.float64)
                    episode_data["timestamp"] = ts + from_ts
        self.curr_traj_id = trajectory_id
        self.curr_traj_data = episode_data
        return episode_data


    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        try:
            return self._trajectory_id_to_index_active[trajectory_id]
        except KeyError as exc:
            raise ValueError(f"Error finding trajectory index for {trajectory_id}") from exc

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        row_idx = self._episode_index_to_row[ep_index]
        return int(self.episodes_hf[row_idx]["data/chunk_index"])

    def _set_curr_episode(self, trajectory_id: int) -> None:
        """Cache episode metadata to avoid repeated lookups on hot paths."""
        if self._curr_episode_id == trajectory_id:
            return
        row_idx = self._episode_index_to_row[trajectory_id]
        ep = self.episodes_hf[row_idx]
        self._curr_episode_id = trajectory_id
        self._curr_row_idx = row_idx
        self._curr_from_index = int(ep["dataset_from_index"])
        self._curr_to_index = int(ep["dataset_to_index"])
        self._curr_length = int(ep["length"])
        self._curr_episode_row = ep

    def _fetch_hf_column(
        self,
        column: str,
        abs_indices: list[int] | np.ndarray,
    ):
        abs_indices_list = self._to_int_index_list(abs_indices)
        if not abs_indices_list:
            return []
        return self.hf_dataset[abs_indices_list][column]

    @staticmethod
    def _to_int_index_list(abs_indices: list[int] | np.ndarray) -> list[int]:
        if isinstance(abs_indices, np.ndarray):
            if abs_indices.size == 0:
                return []
            return abs_indices.astype(np.int64, copy=False).reshape(-1).tolist()
        if len(abs_indices) == 0:
            return []
        return [int(i) for i in abs_indices]

    @staticmethod
    def _to_float64_timestamps(ts_vals) -> np.ndarray:
        ts_arr = np.asarray(ts_vals)
        if ts_arr.ndim == 1 and np.issubdtype(ts_arr.dtype, np.number):
            return ts_arr.astype(np.float64, copy=False)
        if ts_arr.ndim == 2 and ts_arr.shape[1] == 1 and np.issubdtype(ts_arr.dtype, np.number):
            return ts_arr[:, 0].astype(np.float64, copy=False)

        if not isinstance(ts_vals, (list, tuple, np.ndarray)):
            ts_vals = [ts_vals]
        out = np.empty(len(ts_vals), dtype=np.float64)
        for i, value in enumerate(ts_vals):
            if isinstance(value, (list, tuple, np.ndarray)):
                if len(value) == 0:
                    out[i] = 0.0
                    continue
                value = value[0]
            out[i] = float(value)
        return out

    @staticmethod
    def _slice_state_action_rows(
        column_vals,
        start: int,
        end: int,
        expected_dim: int,
    ) -> np.ndarray:
        if len(column_vals) == 0:
            return np.empty((0, expected_dim), dtype=np.float32)

        vals_arr = np.asarray(column_vals)
        if vals_arr.dtype != object:
            if vals_arr.ndim == 2:
                return vals_arr[:, start:end].astype(np.float32, copy=False)
            if vals_arr.ndim == 1:
                return vals_arr.reshape(-1, 1)[:, start:end].astype(np.float32, copy=False)

        rows = []
        for val in column_vals:
            arr = np.asarray(val)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            rows.append(arr[start:end])
        return np.stack(rows, axis=0).astype(np.float32, copy=False)


    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        """Retrieve the data from the dataset and pad it if necessary.
        Args:
            array (np.ndarray): The array to retrieve the data from.
            step_indices (np.ndarray): The step indices to retrieve the data for.
            max_length (int): The maximum length of the data.
            padding_strategy (str): The padding strategy, either "first" or "last".
        """
        # Get the padding indices
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        # Retrieve the data with the non-padding indices
        # If there exists some padding, Given T step_indices, the shape of the retrieved data will be (T', ...) where T' < T
        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"
        # This is the shape of the output, (T, ...)
        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        # Pad the data
        output = np.zeros(expected_shape)
        # Assign the non-padded data
        output[~padding_positions] = raw_data
        # If there exists some padding, pad the data
        if padding_positions.any():
            if padding_strategy == "first_last":
                # Use first / last step data to pad
                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":
                # Use zero padding
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        # Always use mapping to find correct row
        row_idx = self._episode_index_to_row[trajectory_id]
        ep = self.episodes_hf[row_idx]
        vid_chunk_index = int(ep[f"videos/{original_key}/chunk_index"])
        vid_file_index = int(ep[f"videos/{original_key}/file_index"])
        video_filename = self.video_path_pattern.format(
            video_key=original_key,
            chunk_index=vid_chunk_index,
            file_index=vid_file_index,
        )
        return self.dataset_path / video_filename

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        self._set_curr_episode(trajectory_id)
        delta = np.asarray(self.delta_indices[key], dtype=np.int64).reshape(-1)
        if delta.size == 0:
            raise ValueError(f"Empty delta_indices for video key={key} in dataset={self.dataset_name}.")
        # Wrist stream is consumed as a static image: decode only the chunk initial frame.
        # This keeps wrist decoding independent from `num_frames`.
        if "wrist" in key.lower():
            delta = delta[:1]
        step_indices = delta + int(base_index)
        step_indices = np.clip(step_indices, 0, self._curr_length - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        key = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, key)
        abs_indices = self._curr_from_index + step_indices
        if "timestamp" not in self.hf_dataset.column_names:
            raise KeyError(
                f"No 'timestamp' column found in HF dataset for {self.dataset_name}. "
                f"Available columns: {self.hf_dataset.column_names}"
            )
        ts_vals = self._fetch_hf_column("timestamp", abs_indices)
        rel_ts = self._to_float64_timestamps(ts_vals)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        from_ts = 0.0
        ep = self._curr_episode_row if self._curr_episode_row is not None else self.episodes_hf[self._curr_row_idx]
        from_ts_col = f"videos/{original_key}/from_timestamp"
        if from_ts_col in ep:
            from_ts = float(ep[from_ts_col])
        elif getattr(self, "_detected_video_keys", None):
            fallback_key = self._detected_video_keys[0]
            fallback_col = f"videos/{fallback_key}/from_timestamp"
            if fallback_col in ep:
                from_ts = float(ep[fallback_col])
        video_timestamp = rel_ts + from_ts
        # Build per-call kwargs so each dataset/video key uses its own fps parsed from info.json.
        backend_kwargs = dict(self.video_backend_kwargs)

        fps = None
        if key in self.metadata.modalities.video:
            fps = self.metadata.modalities.video[key].fps
        elif original_key in self.metadata.modalities.video:
            fps = self.metadata.modalities.video[original_key].fps
        if fps is not None:
            backend_kwargs["fixed_fps"] = float(fps)

        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend,
            video_backend_kwargs=backend_kwargs,
        )

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        self._set_curr_episode(trajectory_id)
        step_indices = np.asarray(self.delta_indices[key], dtype=np.int64) + int(base_index)
        max_length = self._curr_length
        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        key = key.replace(modality + ".", "")
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[key].original_key
        if le_key is None:
            le_key = key
        le_start = le_state_or_action_cfg[key].start
        le_end = le_state_or_action_cfg[key].end
        feature_dim = le_end - le_start
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[key]
        padding_strategy = "first_last" if state_or_action_cfg.absolute else "zero"

        if padding_strategy == "first_last":
            clamped_step = np.clip(step_indices, 0, max_length - 1)
            target_abs = self._curr_from_index + clamped_step
            unique_abs, inverse = np.unique(target_abs, return_inverse=True)
            unique_vals = self._fetch_hf_column(le_key, unique_abs)
            unique_rows = self._slice_state_action_rows(unique_vals, le_start, le_end, feature_dim)
            return unique_rows[inverse]

        valid_mask = np.logical_and(step_indices >= 0, step_indices < max_length)
        if not np.any(valid_mask):
            return np.zeros((len(step_indices), feature_dim), dtype=np.float32)

        valid_abs = self._curr_from_index + step_indices[valid_mask]
        unique_abs, inverse = np.unique(valid_abs, return_inverse=True)
        unique_vals = self._fetch_hf_column(le_key, unique_abs)
        unique_rows = self._slice_state_action_rows(unique_vals, le_start, le_end, feature_dim)
        output = np.zeros((len(step_indices), unique_rows.shape[-1]), dtype=unique_rows.dtype)
        output[valid_mask] = unique_rows[inverse]
        return output

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            base_index (int): The base index of the trajectory.

        Returns:
            list[str]: The annotation data for the trajectory and step indices. If no matching data is found, return empty strings.
        """
        self._set_curr_episode(trajectory_id)
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the maximum length of the trajectory
        max_length = self._curr_length
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations
        task_indices: list[int | None] = []
        assert key.startswith(
            "annotation."
        ), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert (
            subkey in annotation_meta
        ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
        abs_indices = self._curr_from_index + step_indices
        abs_indices_list = [int(i) for i in np.asarray(abs_indices).tolist()]
        task_vals = self._fetch_hf_column(
            original_key,
            abs_indices_list,
        )
        if not isinstance(task_vals, (list, tuple, np.ndarray)):
            task_vals = [task_vals]
        for value in task_vals:
            if isinstance(value, (list, tuple, np.ndarray)):
                if len(value) == 0:
                    task_indices.append(None)
                    continue
                value = value[0]
            try:
                task_indices.append(int(value))
            except Exception:
                task_indices.append(None)

        return [self._task_index_to_task.get(i, "") if i is not None else "" for i in task_indices]

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ):
        """Get the data corresponding to the modality for a trajectory by a base index.
        This method will call the corresponding helper method based on the modality.
        See the helper methods for more details.
        NOTE: For the language modality, the data is padded with empty strings if no matching data is found.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.
        """
        if modality == "video":
            return self.get_video(
                trajectory_id,
                key,
                base_index,
            )
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(
                trajectory_id,
                modality,
                key,
                base_index,
            )
        elif modality == "language":
            return self.get_language(
                trajectory_id,
                key,
                base_index,
            )
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def build_dataset_statistics(self) -> dict:
        """
        Build dataset statistics for the current dataset keyed by embodiment tag.
        Only includes statistics for keys that are actually used in the dataset.
        The exported statistics order must match the original concat order.
        """
        statistics_data = {}

        # Get used modality keys
        used_action_keys, used_state_keys = get_used_modality_keys(self.modality_keys)
        action_normalization_modes = _collect_action_normalization_modes_from_transforms(self.transforms)

        # Organize statistics by tag
        tag = self.tag
        tag_stats = {}

        # Process action statistics (only for used keys)
        if hasattr(self.metadata.statistics, "action") and self.metadata.statistics.action:
            action_stats = self.metadata.statistics.action

            filtered_action_stats = {}
            for key in used_action_keys:
                if key in action_stats:
                    filtered_action_stats[key] = action_stats[key]

            if filtered_action_stats:
                combined_action_stats = combine_modality_stats(filtered_action_stats)
                mask = generate_action_mask_for_used_keys(
                    self.metadata.modalities.action,
                    filtered_action_stats.keys(),
                    action_normalization_modes=action_normalization_modes,
                )
                combined_action_stats["mask"] = mask
                tag_stats["action"] = combined_action_stats

        # Process state statistics (only for used keys)
        if hasattr(self.metadata.statistics, "state") and self.metadata.statistics.state:
            state_stats = self.metadata.statistics.state

            filtered_state_stats = {}
            for key in used_state_keys:
                if key in state_stats:
                    filtered_state_stats[key] = state_stats[key]

            if filtered_state_stats:
                combined_state_stats = combine_modality_stats(filtered_state_stats)
                tag_stats["state"] = combined_state_stats

        tag_stats["num_transitions"] = len(self)
        tag_stats["num_trajectories"] = len(self.trajectory_ids)
        statistics_data[tag] = tag_stats
        return statistics_data

    def _save_dataset_statistics_(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save dataset statistics to specified path in the required format.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        statistics_data = self.build_dataset_statistics()
        used_action_keys, used_state_keys = get_used_modality_keys(self.modality_keys)

        if format.lower() == "json":
            if not str(save_path).endswith(".json"):
                save_path = save_path.with_suffix(".json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")

        print(f"Single dataset statistics saved to: {save_path}")
        print(f"Used action keys: {list(used_action_keys)}")
        print(f"Used state keys: {list(used_state_keys)}")


class CachedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, img_resize: bool = True, *args, **kwargs):
        """
        This class stores decoded video frames in a disk-backed memmap cache so
        repeated training runs can reuse them without re-decoding source videos.

        Args:
            img_resize (bool): If True, resize frames by preserving aspect ratio and
                scaling the short side to 256. If False, keep the original resolution.
        """
        if not isinstance(img_resize, bool):
            raise TypeError(f"`img_resize` must be bool, got {type(img_resize)}.")
        self.img_resize = bool(img_resize)
        self._cache_resize_short_side = 256 if self.img_resize else None
        self._cache_schema_version = VIDEO_FRAME_CACHE_SCHEMA_VERSION
        self._video_cache_root = Path()
        self._video_cache_dir = Path()
        self._video_cache_manifest: dict[str, Any] = {}
        self._video_memmaps: dict[str, np.memmap] = {}
        self._cache_start_indices_all = np.array([], dtype=np.int64)
        self._cache_traj_id_to_index_all: dict[int, int] = {}

        super().__init__(*args, **kwargs)
        self._prepare_video_disk_cache()
        self._load_video_cache_manifest()

    def _prepare_video_disk_cache(self) -> None:
        video_keys = list(self.modality_keys.get("video", []))
        self._video_cache_root = self.dataset_path / VIDEO_FRAME_CACHE_ROOT
        self._cache_traj_id_to_index_all = {
            int(traj_id): idx for idx, traj_id in enumerate(self._all_trajectory_ids.tolist())
        }
        self._cache_start_indices_all = (
            np.cumsum(self._all_trajectory_lengths, dtype=np.int64) - self._all_trajectory_lengths
            if len(self._all_trajectory_lengths) > 0
            else np.array([], dtype=np.int64)
        )
        if not video_keys:
            return

        cache_inputs = self._collect_video_cache_inputs(video_keys)
        self._video_cache_dir = self._video_cache_root / cache_inputs["fingerprint"]

        if self._cache_dir_ready(self._video_cache_dir):
            return

        if dist.is_initialized():
            if _is_main_process():
                self._build_video_disk_cache_if_needed(cache_inputs)
            dist.barrier()
        else:
            self._build_video_disk_cache_if_needed(cache_inputs)

        if not self._cache_dir_ready(self._video_cache_dir):
            raise RuntimeError(f"Video cache is not ready for dataset={self.dataset_name}: {self._video_cache_dir}")

    def _collect_video_cache_inputs(self, video_keys: list[str]) -> dict[str, Any]:
        source_videos: dict[str, list[dict[str, Any]]] = {}
        source_signature_payload: list[dict[str, Any]] = []
        for video_key in video_keys:
            cache_key = video_key.replace("video.", "", 1)
            key_entries: list[dict[str, Any]] = []
            for trajectory_id, trajectory_length in zip(self._all_trajectory_ids, self._all_trajectory_lengths):
                del trajectory_length
                video_path = self.get_video_path(int(trajectory_id), cache_key)
                stat = video_path.stat()
                try:
                    relative_path = video_path.relative_to(self.dataset_path).as_posix()
                except ValueError:
                    relative_path = video_path.as_posix()
                item = {
                    "trajectory_id": int(trajectory_id),
                    "relative_path": relative_path,
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "from_timestamp": float(self._get_video_episode_from_timestamp(int(trajectory_id), video_key)),
                }
                key_entries.append(item)
                source_signature_payload.append({"video_key": cache_key, **item})
            source_videos[cache_key] = key_entries

        fingerprint_payload = {
            "schema_version": self._cache_schema_version,
            "dataset_name": self.dataset_name,
            "dataset_path": str(self.dataset_path.resolve()),
            "img_resize": bool(self.img_resize),
            "resize_short_side": self._cache_resize_short_side,
            "video_keys": sorted(video_keys),
            "trajectory_ids_all": [int(x) for x in self._all_trajectory_ids.tolist()],
            "trajectory_lengths_all": [int(x) for x in self._all_trajectory_lengths.tolist()],
            "source_videos": source_signature_payload,
        }
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        return {
            "fingerprint": fingerprint,
            "video_keys": list(video_keys),
            "source_videos": source_videos,
        }

    def _get_video_episode_from_timestamp(self, trajectory_id: int, video_key: str) -> float:
        if not hasattr(self, "_episode_index_to_row") or not hasattr(self, "episodes_hf"):
            return 0.0
        cache_key = video_key.replace("video.", "", 1)
        if int(trajectory_id) not in self._episode_index_to_row:
            return 0.0
        row_idx = self._episode_index_to_row[int(trajectory_id)]
        ep = self.episodes_hf[row_idx]
        if not hasattr(self, "lerobot_modality_meta") or cache_key not in self.lerobot_modality_meta.video:
            return 0.0
        original_key = self.lerobot_modality_meta.video[cache_key].original_key
        if original_key is None:
            original_key = cache_key
        from_ts_col = f"videos/{original_key}/from_timestamp"
        if from_ts_col in ep:
            return float(ep[from_ts_col])
        if getattr(self, "_detected_video_keys", None):
            fallback_key = self._detected_video_keys[0]
            fallback_col = f"videos/{fallback_key}/from_timestamp"
            if fallback_col in ep:
                return float(ep[fallback_col])
        return 0.0

    def _get_video_cache_episode_timestamps(self, trajectory_id: int, video_key: str) -> list[float]:
        row_idx = self._episode_index_to_row[int(trajectory_id)]
        ep = self.episodes_hf[row_idx]
        from_index = int(ep["dataset_from_index"])
        expected_length = int(ep["length"])
        abs_indices = np.arange(from_index, from_index + expected_length, dtype=np.int64)
        rel_ts = self._to_float64_timestamps(self._fetch_hf_column("timestamp", abs_indices))
        from_ts = self._get_video_episode_from_timestamp(int(trajectory_id), video_key)
        return (rel_ts + from_ts).astype(np.float64, copy=False).tolist()

    def _cache_dir_ready(self, cache_dir: Path) -> bool:
        manifest_path = cache_dir / VIDEO_FRAME_CACHE_MANIFEST_FILENAME
        if not manifest_path.exists():
            return False
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            return False
        entries = manifest.get("entries", {})
        if not isinstance(entries, dict):
            return False
        for entry in entries.values():
            filename = entry.get("filename")
            if not filename or not (cache_dir / filename).exists():
                return False
        return True

    def _build_video_disk_cache_if_needed(self, cache_inputs: dict[str, Any]) -> None:
        if self._cache_dir_ready(self._video_cache_dir):
            return

        self._video_cache_root.mkdir(parents=True, exist_ok=True)
        lock_path = self._video_cache_root / ".build.lock"
        with _FileLock(lock_path):
            if self._cache_dir_ready(self._video_cache_dir):
                return

            tmp_dir = self._video_cache_root / f"{cache_inputs['fingerprint']}.tmp.{os.getpid()}"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            try:
                manifest = self._build_video_disk_cache_into_dir(tmp_dir, cache_inputs)
                with open(tmp_dir / VIDEO_FRAME_CACHE_MANIFEST_FILENAME, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
                os.replace(tmp_dir, self._video_cache_dir)
            finally:
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)

    def _build_video_disk_cache_into_dir(self, cache_dir: Path, cache_inputs: dict[str, Any]) -> dict[str, Any]:
        entries: dict[str, dict[str, Any]] = {}
        total_bytes = 0
        total_frames_expected = int(np.sum(self._all_trajectory_lengths))
        for video_key in cache_inputs["video_keys"]:
            cache_key = video_key.replace("video.", "", 1)
            entry_filename = f"{cache_key}.frames.uint8.memmap"
            entry_path = cache_dir / entry_filename
            tasks = [
                {
                    "video_path": self.get_video_path(int(trajectory_id), cache_key).as_posix(),
                    "trajectory_id": int(trajectory_id),
                    "trajectory_index": trajectory_index,
                    "expected_length": int(trajectory_length),
                    "video_key": video_key,
                    "dataset_name": self.dataset_name,
                    "video_backend": self.video_backend,
                    "video_backend_kwargs": dict(self.video_backend_kwargs),
                    "resize_short_side": self._cache_resize_short_side,
                    "video_timestamps": self._get_video_cache_episode_timestamps(int(trajectory_id), video_key),
                }
                for trajectory_index, (trajectory_id, trajectory_length) in enumerate(
                    zip(self._all_trajectory_ids, self._all_trajectory_lengths)
                )
            ]
            memmap = None
            shape: tuple[int, ...] | None = None

            if tasks:
                worker_count = min(_resolve_video_cache_build_workers(), len(tasks))
                print(
                    f"[LeRobotDataset][video_cache] dataset={self.dataset_name} cache_key={cache_key} "
                    f"workers={worker_count} total_trajectories={len(tasks)}"
                )
                with tqdm(total=len(tasks), desc=f"Building disk cache {cache_key}") as progress:
                    first_result = _decode_video_for_cache_task(**tasks[0])
                    shape = (total_frames_expected, *first_result["frame_shape"])
                    memmap = np.memmap(entry_path, dtype=np.uint8, mode="w+", shape=shape)
                    self._write_cache_result_to_memmap(
                        memmap=memmap,
                        result=first_result,
                        dataset_name=self.dataset_name,
                        video_key=video_key,
                        expected_hwc=tuple(first_result["frame_shape"]),
                    )
                    progress.update(1)

                    remaining_tasks = tasks[1:]
                    if worker_count <= 1 or not remaining_tasks:
                        for task in remaining_tasks:
                            result = _decode_video_for_cache_task(**task)
                            self._write_cache_result_to_memmap(
                                memmap=memmap,
                                result=result,
                                dataset_name=self.dataset_name,
                                video_key=video_key,
                                expected_hwc=tuple(first_result["frame_shape"]),
                            )
                            progress.update(1)
                    else:
                        max_pending = max(worker_count * 2, 1)
                        mp_context = mp.get_context("spawn")
                        task_iter = iter(remaining_tasks)
                        pending: dict[Any, dict[str, Any]] = {}

                        def submit_next(executor: ProcessPoolExecutor) -> bool:
                            try:
                                task = next(task_iter)
                            except StopIteration:
                                return False
                            future = executor.submit(_decode_video_for_cache_task, **task)
                            pending[future] = task
                            return True

                        with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_context) as executor:
                            for _ in range(min(max_pending, len(remaining_tasks))):
                                if not submit_next(executor):
                                    break

                            while pending:
                                done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                                for future in done:
                                    pending.pop(future, None)
                                    result = future.result()
                                    self._write_cache_result_to_memmap(
                                        memmap=memmap,
                                        result=result,
                                        dataset_name=self.dataset_name,
                                        video_key=video_key,
                                        expected_hwc=tuple(first_result["frame_shape"]),
                                    )
                                    progress.update(1)
                                    submit_next(executor)

            if shape is None:
                if self._cache_resize_short_side is not None:
                    shape = (0, int(self._cache_resize_short_side), int(self._cache_resize_short_side), 3)
                else:
                    shape = (0, 0, 0, 3)
                memmap = np.memmap(entry_path, dtype=np.uint8, mode="w+", shape=shape)
            assert memmap is not None and shape is not None
            memmap.flush()
            del memmap

            total_bytes += int(np.prod(shape, dtype=np.int64)) * np.dtype(np.uint8).itemsize
            entries[cache_key] = {
                "filename": entry_filename,
                "dtype": np.dtype(np.uint8).name,
                "shape": list(shape),
                "resolution": [int(shape[2]), int(shape[1])] if len(shape) == 4 else [0, 0],
                "channels": int(shape[3]) if len(shape) == 4 else 0,
                "total_frames": int(shape[0]) if len(shape) > 0 else 0,
            }

        total_mib = total_bytes / (1024 * 1024) if total_bytes else 0.0
        print(
            f"[LeRobotDataset][video_cache] dataset={self.dataset_name} cache_dir={self._video_cache_dir} "
            f"video_keys={len(entries)} trajectories={len(self._all_trajectory_ids)} "
            f"frames={total_frames_expected * max(1, len(entries))} approx_disk_mib={total_mib:.2f}"
        )
        return {
            "schema_version": self._cache_schema_version,
            "fingerprint": cache_inputs["fingerprint"],
            "dataset_name": self.dataset_name,
            "dataset_path": str(self.dataset_path.resolve()),
            "img_resize": bool(self.img_resize),
            "resize_short_side": self._cache_resize_short_side,
            "video_keys": list(cache_inputs["video_keys"]),
            "trajectory_ids_all": [int(x) for x in self._all_trajectory_ids.tolist()],
            "trajectory_lengths_all": [int(x) for x in self._all_trajectory_lengths.tolist()],
            "start_indices_all": [int(x) for x in self._cache_start_indices_all.tolist()],
            "entries": entries,
            "source_videos": cache_inputs["source_videos"],
        }

    def _write_cache_result_to_memmap(
        self,
        memmap: np.memmap,
        result: dict[str, Any],
        dataset_name: str,
        video_key: str,
        expected_hwc: tuple[int, ...],
    ) -> None:
        frames = np.asarray(result["frames"], dtype=np.uint8)
        frame_hwc = tuple(int(x) for x in frames.shape[1:])
        if frame_hwc != expected_hwc:
            raise ValueError(
                f"Frame shape mismatch while building video cache for dataset={dataset_name} "
                f"video_key={video_key} trajectory_id={int(result['trajectory_id'])}: "
                f"got shape={frame_hwc}, expected shape={expected_hwc}"
            )

        expected_length = int(result["expected_length"])
        if int(frames.shape[0]) != expected_length:
            raise ValueError(
                f"Frame length mismatch while building video cache for dataset={dataset_name} "
                f"video_key={video_key} trajectory_id={int(result['trajectory_id'])}: "
                f"got length={int(frames.shape[0])}, expected length={expected_length}"
            )

        trajectory_index = int(result["trajectory_index"])
        start = int(self._cache_start_indices_all[trajectory_index])
        end = start + expected_length
        memmap[start:end] = frames

    def _normalize_cached_video_frames(
        self,
        frames: np.ndarray,
        expected_length: int,
        video_key: str,
        trajectory_id: int,
        video_path: Path,
    ) -> np.ndarray:
        return _normalize_cached_video_frames_static(
            frames=frames,
            expected_length=expected_length,
            video_key=video_key,
            trajectory_id=trajectory_id,
            video_path=video_path,
            dataset_name=self.dataset_name,
        )

    def _load_video_cache_manifest(self) -> None:
        if not self.modality_keys.get("video", []):
            return
        manifest_path = self._video_cache_dir / VIDEO_FRAME_CACHE_MANIFEST_FILENAME
        with open(manifest_path, "r", encoding="utf-8") as f:
            self._video_cache_manifest = json.load(f)

        self._cache_start_indices_all = np.asarray(self._video_cache_manifest["start_indices_all"], dtype=np.int64)
        self._cache_traj_id_to_index_all = {
            int(traj_id): idx
            for idx, traj_id in enumerate(self._video_cache_manifest["trajectory_ids_all"])
        }

    def _ensure_video_memmap_open(self, key: str) -> np.memmap:
        cache_key = key.replace("video.", "", 1)
        if cache_key in self._video_memmaps:
            return self._video_memmaps[cache_key]
        entry = self._video_cache_manifest["entries"][cache_key]
        memmap = np.memmap(
            self._video_cache_dir / entry["filename"],
            dtype=np.dtype(entry["dtype"]),
            mode="r",
            shape=tuple(int(x) for x in entry["shape"]),
        )
        self._video_memmaps[cache_key] = memmap
        return memmap

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        delta = np.asarray(self.delta_indices[key], dtype=np.int64).reshape(-1)
        if delta.size == 0:
            raise ValueError(f"Empty delta_indices for video key={key} in dataset={self.dataset_name}.")
        # Wrist stream is consumed as a static image: decode only the chunk initial frame.
        # This keeps wrist decoding independent from `num_frames`.
        if "wrist" in key.lower():
            delta = delta[:1]
        step_indices = delta + int(base_index)
        trajectory_index = self._cache_traj_id_to_index_all[int(trajectory_id)]
        max_length = int(self._video_cache_manifest["trajectory_lengths_all"][trajectory_index])
        step_indices = np.clip(step_indices, 0, max_length - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        absolute_indices = self._cache_start_indices_all[trajectory_index] + step_indices
        frames = self._ensure_video_memmap_open(key)[absolute_indices]
        return np.asarray(frames)

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        if self.img_resize and self._video_cache_manifest:
            for key, entry in self._video_cache_manifest.get("entries", {}).items():
                if key in metadata.modalities.video:
                    metadata.modalities.video[key].resolution = list(entry["resolution"])
        super().set_transforms_metadata(metadata)


def safe_hash(input_tuple):
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class MixtureSpecElement(BaseModel):
    dataset_path: list[Path] | Path = Field(..., description="The path to the dataset.")
    dataset_weight: float = Field(..., description="The weight of the dataset in the mixture.")
    distribute_weights: bool = Field(
        default=False,
        description="Whether to distribute the weights of the dataset across all the paths. If True, the weights will be evenly distributed across all the paths.",
    )


# Helper functions for dataset statistics

def combine_modality_stats(modality_stats: dict) -> dict:
    """
    Combine statistics from all sub-keys under a modality.
    
    Args:
        modality_stats (dict): Statistics for a modality, containing multiple sub-keys.
                               Each sub-key contains DatasetStatisticalValues object.
        
    Returns:
        dict: Combined statistics
    """
    combined_stats = {
        "mean": [],
        "std": [],
        "max": [],
        "min": [],
        "q01": [],
        "q99": []
    }
    
    # Combine statistics in sub-key order
    for subkey in modality_stats.keys():
        subkey_stats = modality_stats[subkey]  # This is a DatasetStatisticalValues object
        
        # Convert DatasetStatisticalValues to dict-like access
        for stat_name in ["mean", "std", "max", "min", "q01", "q99"]:
            stat_value = getattr(subkey_stats, stat_name)
            if isinstance(stat_value, (list, tuple)):
                combined_stats[stat_name].extend(stat_value)
            else:
                # Handle NDArray case - convert to list
                if hasattr(stat_value, 'tolist'):
                    combined_stats[stat_name].extend(stat_value.tolist())
                else:
                    combined_stats[stat_name].append(float(stat_value))
    
    return combined_stats

def _collect_action_normalization_modes_from_transforms(transforms: Any) -> dict[str, str]:
    normalization_modes: dict[str, str] = {}
    if transforms is None:
        return normalization_modes

    transform_list = getattr(transforms, "transforms", None)
    if transform_list is None:
        transform_list = [transforms]

    for transform in transform_list:
        modes = getattr(transform, "normalization_modes", None)
        apply_to = getattr(transform, "apply_to", None)
        if not isinstance(modes, dict) or not isinstance(apply_to, list):
            continue
        for key in apply_to:
            if not isinstance(key, str) or not key.startswith("action."):
                continue
            mode = modes.get(key, None)
            if mode is not None:
                normalization_modes[key.replace("action.", "", 1)] = str(mode)
    return normalization_modes


def _collect_action_normalization_modes_from_datasets(datasets: Sequence[Any]) -> dict[str, str]:
    normalization_modes: dict[str, str] = {}
    for dataset in datasets:
        for key, mode in _collect_action_normalization_modes_from_transforms(getattr(dataset, "transforms", None)).items():
            normalization_modes.setdefault(key, mode)
    return normalization_modes


def generate_action_mask_for_used_keys(
    action_modalities: dict,
    used_action_keys_ordered,
    action_normalization_modes: dict[str, str] | None = None,
) -> list[bool]:
    """
    Generate mask based on action modalities, but only for used keys.
    Only action dimensions normalized as `binary` are marked False.
    
    Args:
        action_modalities (dict): Configuration information for action modalities.
        used_action_keys_ordered: Iterable of actually used action keys in the correct order.
        action_normalization_modes: Mapping from action subkey to normalization mode.
        
    Returns:
        list[bool]: List of mask values
    """
    mask = []
    normalization_modes = dict(action_normalization_modes or {})
    
    # Generate mask in the same order as the statistics were combined
    for subkey in used_action_keys_ordered:
        if subkey in action_modalities:
            subkey_config = action_modalities[subkey]
            
            # Get dimension count from shape
            if hasattr(subkey_config, 'shape') and len(subkey_config.shape) > 0:
                dim_count = subkey_config.shape[0]
            else:
                dim_count = 1
            
            normalization_mode = str(normalization_modes.get(subkey, "min_max")).lower()
            is_binary = normalization_mode == "binary"
            
            # Generate mask value for each dimension
            for _ in range(dim_count):
                mask.append(not is_binary)
    
    return mask

def get_used_modality_keys(modality_keys: dict) -> tuple[list, list]:
    """Extract used action and state keys from modality configuration."""
    used_action_keys = []
    used_state_keys = []
    
    # Extract action keys (remove "action." prefix)
    for action_key in modality_keys.get("action", []):
        if action_key.startswith("action."):
            clean_key = action_key.replace("action.", "")
            used_action_keys.append(clean_key)
    
    # Extract state keys (remove "state." prefix)  
    for state_key in modality_keys.get("state", []):
        if state_key.startswith("state."):
            clean_key = state_key.replace("state.", "")
            used_state_keys.append(clean_key)
    
    return used_action_keys, used_state_keys

class LeRobotMixtureDataset(Dataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    def __init__(
        self,
        data_mixture: Sequence[tuple[LeRobotSingleDataset, float]],
        mode: str,
        balance_dataset_weights: bool = True,
        seed: int = 42,
        random_single_non_wrist_view: bool = False,
        metadata_config: dict = {
            "percentile_mixing_method": "min_max",
        },
        **kwargs,
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[LeRobotSingleDataset, float]]): Datasets and their corresponding weights.
            mode (str): If "train", __getitem__ will return different samples every epoch; if "val" or "test", __getitem__ will return the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be multiplied by the total trajectory length of each dataset.
            seed (int): Random seed for sampling.
        """
        datasets: list[LeRobotSingleDataset] = []
        dataset_sampling_weights: list[float] = []
        for dataset, weight in data_mixture:
            # Check if dataset is valid and has data
            if len(dataset) == 0:
                print(f"Warning: Skipping empty dataset {dataset.dataset_name}")
                continue
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)
        
        if len(datasets) == 0:
            raise ValueError("No valid datasets found in the mixture. All datasets are empty.")
        
        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.seed = seed
        self.mode = mode
        self.data_cfg = kwargs["data_cfg"] if "data_cfg" in kwargs else None

        self.random_single_non_wrist_view = random_single_non_wrist_view
        self._subset_transform_cache: dict[tuple[int, tuple[str, ...]], ComposedModalityTransform] = {}

        # Set properties for sampling

        # 1. Dataset lengths
        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])
        print(f"Dataset lengths: {self._dataset_lengths}")

        # 2. Dataset sampling weights
        self._raw_dataset_sampling_weights = np.array(dataset_sampling_weights, dtype=np.float64)
        self._dataset_sampling_weights = self._raw_dataset_sampling_weights.copy()
        
        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths
        
        # Check for zero or negative weights before normalization
        if np.any(self._dataset_sampling_weights <= 0):
            print(f"Warning: Found zero or negative sampling weights: {self._dataset_sampling_weights}")
            # Set minimum weight to prevent division issues
            self._dataset_sampling_weights = np.maximum(self._dataset_sampling_weights, 1e-8)
        
        # Normalize weights
        weights_sum = self._dataset_sampling_weights.sum()
        if weights_sum == 0 or np.isnan(weights_sum):
            print(f"Error: Invalid weights sum: {weights_sum}")
            # Fallback to equal weights
            self._dataset_sampling_weights = np.ones(len(self.datasets)) / len(self.datasets)
            print(f"Fallback to equal weights")
            self._effective_dataset_sampling_weights = self._dataset_sampling_weights.copy()
        else:
            self._effective_dataset_sampling_weights = self._dataset_sampling_weights.copy()
            self._dataset_sampling_weights /= weights_sum

        # 3. Primary dataset indices
        self._primary_dataset_indices = np.array(dataset_sampling_weights) == 1.0
        if not np.any(self._primary_dataset_indices):
            print(f"Warning: No dataset with weight 1.0 found. Original weights: {dataset_sampling_weights}")
            # Fallback: use the dataset(s) with maximum weight as primary
            max_weight = max(dataset_sampling_weights)
            self._primary_dataset_indices = np.array(dataset_sampling_weights) == max_weight
            print(f"Using datasets with maximum weight {max_weight} as primary: {self._primary_dataset_indices}")
            
        if not np.any(self._primary_dataset_indices):
            # This should never happen, but just in case
            print("Error: Still no primary dataset found. Using first dataset as primary.")
            self._primary_dataset_indices = np.zeros(len(self.datasets), dtype=bool)
            self._primary_dataset_indices[0] = True

        self._log_dataset_mixture_table()

        # Set the epoch and sample the first epoch
        self.set_epoch(0)

        self.update_metadata(metadata_config)

    @property
    def dataset_lengths(self) -> np.ndarray:
        """The lengths of each dataset."""
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The sampling weights for each dataset."""
        return self._dataset_sampling_weights

    @property
    def primary_dataset_indices(self) -> np.ndarray:
        """The indices of the primary datasets."""
        return self._primary_dataset_indices

    def __str__(self) -> str:
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
            }
            dataset_descriptions.append(dataset_description)
        return json.dumps({"Mixture dataset": dataset_descriptions}, indent=2)

    def _log_dataset_mixture_table(self) -> None:
        if not _is_main_process():
            return

        headers = ["Idx", "Dataset", "Ratio(%)"]
        rows: list[list[str]] = []

        for idx, dataset in enumerate(self.datasets):
            dataset_name = getattr(dataset, "dataset_name", str(dataset))
            rows.append(
                [
                    str(idx),
                    str(dataset_name),
                    f"{float(self.dataset_sampling_weights[idx]) * 100.0:.2f}",
                ]
            )

        col_widths = [
            max(len(headers[col_idx]), *(len(row[col_idx]) for row in rows)) for col_idx in range(len(headers))
        ]

        def _format_row(values: list[str]) -> str:
            return " | ".join(value.ljust(col_widths[col_idx]) for col_idx, value in enumerate(values))

        sep = "-+-".join("-" * width for width in col_widths)
        print(
            f"[RANK 0] Dataset mixture ratios "
            f"(balance_dataset_weights={self.balance_dataset_weights}):"
        )
        print(_format_row(headers))
        print(sep)
        for row in rows:
            print(_format_row(row))

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch
        # self.sampled_steps = self.sample_epoch()

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        """Sample a single step from the dataset.
        
        Uses uniform sampling across all steps via hf_dataset absolute index,
        following the official LeRobot v3 approach.
        """
        rng = self._create_rng(index)

        # Sample dataset based on dataset_sampling_weights
        dataset_index = rng.choice(len(self.datasets), p=self.dataset_sampling_weights)
        dataset = self.datasets[dataset_index]

        # Sample local index uniformly from active split, then map to (trajectory_id, base_index)
        local_idx = rng.integers(0, len(dataset))
        trajectory_id, base_index = dataset.abs_index_to_episode_step(local_idx)
        return dataset, trajectory_id, base_index

    def _create_rng(self, index: int, salt: str = "") -> np.random.Generator:
        if self.mode not in {"train", "all"}:
            if salt:
                seed = safe_hash((index, salt))
            else:
                seed = int(index)
        else:
            seed_tuple = (self.epoch, index, self.seed) if not salt else (self.epoch, index, self.seed, salt)
            seed = safe_hash(seed_tuple)
        return np.random.default_rng(seed)

    def _get_retry_index(self, original_index: int, attempt: int, reason: str) -> int:
        if len(self) <= 1:
            return 0
        rng = self._create_rng(original_index, salt=f"retry:{reason}:{attempt}")
        return int(rng.integers(0, len(self)))

    def _select_video_keys_for_sample(self, dataset: LeRobotSingleDataset, index: int) -> list[str]:
        all_video_keys = list(dataset.modality_keys.get("video", []))
        if not self.random_single_non_wrist_view or len(all_video_keys) <= 1:
            return all_video_keys

        non_wrist_keys = [k for k in all_video_keys if "wrist" not in k.lower()]
        if len(non_wrist_keys) <= 1:
            return all_video_keys

        rng = self._create_rng(index, salt="view_select")
        chosen_non_wrist = non_wrist_keys[int(rng.integers(0, len(non_wrist_keys)))]
        selected_keys = {chosen_non_wrist}
        for key in all_video_keys:
            if "wrist" in key.lower():
                selected_keys.add(key)
        # Preserve original key order from modality config.
        return [k for k in all_video_keys if k in selected_keys]

    def _sample_candidate(
        self,
        index: int,
        max_video_retries: int,
    ) -> tuple[LeRobotSingleDataset, int, int, list[str]]:
        current_index = int(index)
        for video_retry in range(max_video_retries):
            dataset, trajectory_id, step = self.sample_step(current_index)
            selected_video_keys = self._select_video_keys_for_sample(dataset, current_index)
            if not selected_video_keys:
                raise ValueError(f"Dataset {dataset.dataset_name} has no video keys configured.")
            key = selected_video_keys[0].replace("video.", "")
            video_path = dataset.get_video_path(trajectory_id, key)
            if os.path.exists(video_path):
                return dataset, trajectory_id, step, selected_video_keys
            current_index = self._get_retry_index(index, video_retry, reason="missing_video")
        raise FileNotFoundError(f"Failed to find valid video path after {max_video_retries} attempts for index {index}.")

    def _get_transforms_for_selected_video_keys(
        self,
        dataset: LeRobotSingleDataset,
        selected_video_keys: list[str],
    ) -> ComposedModalityTransform:
        base_video_keys = list(dataset.modality_keys.get("video", []))
        if tuple(selected_video_keys) == tuple(base_video_keys):
            return dataset.transforms

        cache_key = (id(dataset), tuple(selected_video_keys))
        cached = self._subset_transform_cache.get(cache_key)
        if cached is not None:
            return cached

        transforms = copy.deepcopy(dataset.transforms)
        for transform in transforms.transforms:
            if isinstance(transform, VideoTransform):
                transform.apply_to = [key for key in transform.apply_to if key in selected_video_keys]

        transforms.set_metadata(dataset.metadata)
        if getattr(dataset.transforms, "training", True):
            transforms.train()
        else:
            transforms.eval()

        self._subset_transform_cache[cache_key] = transforms
        return transforms

    def _build_output_sample(
        self,
        dataset: LeRobotSingleDataset,
        data: dict,
        selected_video_keys: list[str],
    ) -> dict:
        return self._build_latent_world_output_sample(dataset, data, selected_video_keys)

    @staticmethod
    def _ensure_sequence_2d_tensor(value: torch.Tensor) -> torch.Tensor:
        tensor = value.to(dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.contiguous()

    def _build_latent_world_output_sample(
        self,
        dataset: LeRobotSingleDataset,
        data: dict,
        selected_video_keys: list[str],
    ) -> dict:
        primary_videos: list[torch.Tensor] = []
        wrist_frames: list[torch.Tensor] = []
        for video_key in selected_video_keys:
            view_frames = data[video_key]
            view_frames = view_frames.contiguous()
            if "wrist" not in video_key.lower():
                primary_videos.append(view_frames)
            else:
                wrist_frames.append(view_frames[0].contiguous())

        if not primary_videos:
            raise ValueError("LatentWorld raw sample could not resolve any primary video views.")

        primary_video_tensor = torch.stack(primary_videos, dim=0).contiguous()
        reference_primary_video = primary_video_tensor[0]

        height = int(reference_primary_video.shape[-2])
        width = int(reference_primary_video.shape[-1])
        if wrist_frames:
            wrist_images = torch.stack(wrist_frames, dim=0)
        else:
            wrist_images = torch.empty((0, 3, height, width), dtype=reference_primary_video.dtype)

        language = data[dataset.modality_keys["language"][0]][0]
        if "action" not in data or "state" not in data:
            raise KeyError(
                f"Missing required transformed keys for LatentWorld sample: available keys={sorted(data.keys())}."
            )

        return {
            "primary_videos": primary_video_tensor,
            "wrist_images": wrist_images,
            "lang": str(language),
            "state": self._ensure_sequence_2d_tensor(data["state"]),
            "action": self._ensure_sequence_2d_tensor(data["action"]),
            "embodiment_id": int(dataset.embodiment_id),
            "action_hz": float(dataset.action_hz),
        }

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single trajectory and start index.

        Args:
            index (int): The index of the trajectory to get.

        Returns:
            dict: The data for the trajectory and start index.
        """
        max_retries = 10
        last_exception = None
        current_index = int(index)
        
        for attempt in range(max_retries):
            try:
                dataset, trajectory_id, step, selected_video_keys = self._sample_candidate(
                    current_index,
                    max_video_retries=max_retries,
                )
                raw_data = dataset.get_step_data(
                    trajectory_id,
                    step,
                    modality_keys_override={"video": selected_video_keys},
                )
                transforms = self._get_transforms_for_selected_video_keys(dataset, selected_video_keys)
                data = transforms(raw_data)
                return self._build_output_sample(dataset, data, selected_video_keys)

                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Log the error but continue trying
                    print(f"Attempt {attempt + 1}/{max_retries} failed for index {current_index}: {e}")
                    print(f"Retrying with new sample...")
                    current_index = self._get_retry_index(index, attempt, reason="exception")
                else:
                    # All retries exhausted
                    print(f"All {max_retries} attempts failed for index {current_index}")
                    print(f"Last error: {last_exception}")
                    raise last_exception

    def __len__(self) -> int:
        """Get the length of a single epoch in the mixture.

        Returns:
            int: The length of a single epoch in the mixture.
        """
        # Check for potential issues
        if len(self.datasets) == 0:
            return 0
            
        # Check if any dataset lengths are 0 or NaN
        if np.any(self.dataset_lengths == 0) or np.any(np.isnan(self.dataset_lengths)):
            print(f"Warning: Found zero or NaN dataset lengths: {self.dataset_lengths}")
            # Filter out zero/NaN length datasets
            valid_indices = (self.dataset_lengths > 0) & (~np.isnan(self.dataset_lengths))
            if not np.any(valid_indices):
                print("Error: All datasets have zero or NaN length")
                return 0
        else:
            valid_indices = np.ones(len(self.datasets), dtype=bool)
        
        # Check if any sampling weights are 0 or NaN
        if np.any(self.dataset_sampling_weights == 0) or np.any(np.isnan(self.dataset_sampling_weights)):
            print(f"Warning: Found zero or NaN sampling weights: {self.dataset_sampling_weights}")
            # Use only valid weights
            valid_weights = (self.dataset_sampling_weights > 0) & (~np.isnan(self.dataset_sampling_weights))
            valid_indices = valid_indices & valid_weights
            if not np.any(valid_indices):
                print("Error: All sampling weights are zero or NaN")
                return 0
        
        # Check primary dataset indices
        primary_and_valid = self.primary_dataset_indices & valid_indices
        if not np.any(primary_and_valid):
            print(f"Warning: No valid primary datasets found. Primary indices: {self.primary_dataset_indices}, Valid indices: {valid_indices}")
            # Fallback: use the largest valid dataset
            if np.any(valid_indices):
                max_length = self.dataset_lengths[valid_indices].max()
                print(f"Fallback: Using maximum dataset length: {max_length}")
                return int(max_length)
            else:
                return 0
        
        # Calculate the ratio and get max
        ratios = (self.dataset_lengths / self.dataset_sampling_weights)[primary_and_valid]
        
        # Check for NaN or inf in ratios
        if np.any(np.isnan(ratios)) or np.any(np.isinf(ratios)):
            print(f"Warning: Found NaN or inf in ratios: {ratios}")
            print(f"Dataset lengths: {self.dataset_lengths[primary_and_valid]}")
            print(f"Sampling weights: {self.dataset_sampling_weights[primary_and_valid]}")
            # Filter out invalid ratios
            valid_ratios = ratios[~np.isnan(ratios) & ~np.isinf(ratios)]
            if len(valid_ratios) == 0:
                print("Error: All ratios are NaN or inf")
                return 0
            max_ratio = valid_ratios.max()
        else:
            max_ratio = ratios.max()
        
        result = int(max_ratio)
        if result == 0:
            print(f"Warning: Dataset mixture length is 0")
        return result

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
        dataset_sampling_weights: list[float] | np.ndarray,
        percentile_mixing_method: str = "weighted_average",
    ) -> dict[str, dict[str, list[float]]]:
        """
        Computes overall statistics from per-task statistics using dataset sample weights.

        Args:
            per_task_stats: List of per-task statistics.
            Example format of one element in the per-task statistics list:
                {
                    "state.gripper": {
                        "min": [...],
                        "max": [...],
                        "mean": [...],
                        "std": [...],
                        "q01": [...],
                        "q99": [...],
                    },
                    ...
                }
            dataset_sampling_weights: List of sample weights for each task.
            percentile_mixing_method: The method to mix the percentiles, either "weighted_average" or "weighted_std".

        Returns:
            A dict of overall statistics per modality.
        """
        # Normalize the sample weights to sum to 1
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        # Initialize overall statistics dict
        overall_stats: dict[str, dict[str, list[float]]] = {}

        # Get the list of modality keys
        modality_keys = per_task_stats[0].keys()

        for modality in modality_keys:
            # Number of dimensions (assuming consistent across tasks)
            num_dims = len(per_task_stats[0][modality]["mean"])

            # Initialize accumulators for means and variances
            weighted_means = np.zeros(num_dims)
            weighted_squares = np.zeros(num_dims)

            # Collect min, max, q01, q99 from all tasks
            min_list = []
            max_list = []
            q01_list = []
            q99_list = []

            for task_idx, task_stats in enumerate(per_task_stats):
                w_i = normalized_weights[task_idx]
                stats = task_stats[modality]
                means = np.array(stats["mean"])
                stds = np.array(stats["std"])

                # Update weighted sums for mean and variance
                weighted_means += w_i * means
                weighted_squares += w_i * (stds**2 + means**2)

                # Collect min, max, q01, q99
                min_list.append(stats["min"])
                max_list.append(stats["max"])
                q01_list.append(stats["q01"])
                q99_list.append(stats["q99"])

            # Compute overall mean
            overall_mean = weighted_means.tolist()

            # Compute overall variance and std deviation
            overall_variance = weighted_squares - weighted_means**2
            overall_std = np.sqrt(overall_variance).tolist()

            # Compute overall min and max per dimension
            overall_min = np.min(np.array(min_list), axis=0).tolist()
            overall_max = np.max(np.array(max_list), axis=0).tolist()

            # Compute overall q01 and q99 per dimension
            # Use weighted average of per-task quantiles
            q01_array = np.array(q01_list)
            q99_array = np.array(q99_list)
            if percentile_mixing_method == "weighted_average":
                weighted_q01 = np.average(q01_array, axis=0, weights=normalized_weights).tolist()
                weighted_q99 = np.average(q99_array, axis=0, weights=normalized_weights).tolist()
                # std_q01 = np.std(q01_array, axis=0).tolist()
                # std_q99 = np.std(q99_array, axis=0).tolist()
                # print(modality)
                # print(f"{std_q01=}, {std_q99=}")
                # print(f"{weighted_q01=}, {weighted_q99=}")
            elif percentile_mixing_method == "min_max":
                weighted_q01 = np.min(q01_array, axis=0).tolist()
                weighted_q99 = np.max(q99_array, axis=0).tolist()
            else:
                raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")

            # Store the overall statistics for the modality
            overall_stats[modality] = {
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

        return overall_stats

    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
        used_keys: dict[str, set[str]] | None = None,
    ) -> DatasetMetadata:
        """Merge multiple metadata into one.
        
        Args:
            metadatas: List of DatasetMetadata objects to merge.
            dataset_sampling_weights: Weights for each dataset when computing statistics.
            percentile_mixing_method: Method for mixing percentiles ("weighted_average" or "min_max").
            used_keys: Optional dict mapping modality names to sets of keys actually used in training.
                       If provided, only these keys will be included in the merged metadata.
                       Example: {"video": {"primary_view"}, "state": {"eef_position", "gripper"}}
        """
        # Convert to dicts
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]
        # Create a new metadata dict
        merged_metadata = {}

        # Check all metadata have the same embodiment tag
        assert all(
            metadata.embodiment_tag == metadatas[0].embodiment_tag for metadata in metadatas
        ), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        # Merge the dataset statistics
        dataset_statistics = {}
        dataset_statistics["state"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        merged_metadata["statistics"] = dataset_statistics

        # Merge the modality configs
        modality_configs = defaultdict(set)
        for metadata in metadata_dicts:
            for modality, configs in metadata["modalities"].items():
                modality_configs[modality].add(json.dumps(configs))
        merged_metadata["modalities"] = {}
        for modality, configs in modality_configs.items():
            # Get the set of used keys for this modality (if filtering is enabled)
            modality_used_keys = used_keys.get(modality) if used_keys else None
            
            if modality == "video":
                # Union all video configs - different datasets may have different views
                # Video resolution is overridden by img_resize anyway, and fps is never used
                merged_video = {}
                for config_json in configs:
                    config_dict = json.loads(config_json)
                    for video_key, video_meta in config_dict.items():
                        # Filter by used_keys if provided
                        if modality_used_keys is not None and video_key not in modality_used_keys:
                            continue
                        if video_key not in merged_video:
                            merged_video[video_key] = video_meta
                merged_metadata["modalities"][modality] = merged_video
            else:
                # For state/action, require strict consistency
                assert (
                    len(configs) == 1
                ), f"Multiple modality configs for modality {modality}: {list(configs)}"
                merged_metadata["modalities"][modality] = json.loads(configs.pop())

        return DatasetMetadata.model_validate(merged_metadata)

    def update_metadata(self, metadata_config: dict, cached_statistics_path: Path | str | None = None) -> None:
        """
        Merge multiple metadatas into one and set the transforms with the merged metadata.

        Args:
            metadata_config (dict): Configuration for the metadata.
                "percentile_mixing_method": The method to mix the percentiles, either "weighted_average" or "min_max".
                    weighted_average: Use the weighted average of the percentiles using the weight used in sampling the datasets.
                    min_max: Use the min of the 1st percentile and max of the 99th percentile.
        """
        # If cached path is provided, try to load and apply
        if cached_statistics_path is not None:
            try:
                cached_stats = self.load_merged_statistics(cached_statistics_path)
                self.apply_cached_statistics(cached_stats)
                return
            except (FileNotFoundError, KeyError, ValidationError) as e:
                print(f"Failed to load cached statistics: {e}")
                print("Falling back to computing statistics from scratch...")

        self.tag = EmbodimentTag.NEW_EMBODIMENT.value
        self.merged_metadata: dict[str, DatasetMetadata] = {}
        
        # Collect all used keys from all datasets' modality_configs
        all_used_keys: dict[str, set[str]] = {}
        for dataset in self.datasets:
            if hasattr(dataset, 'modality_configs') and dataset.modality_configs:
                for modality, config in dataset.modality_configs.items():
                    if hasattr(config, 'modality_keys'):
                        if modality not in all_used_keys:
                            all_used_keys[modality] = set()
                        for full_key in config.modality_keys:
                            # Extract subkey from full key (e.g., "state.eef_position" -> "eef_position")
                            parts = full_key.split(".", 1)
                            if len(parts) == 2:
                                all_used_keys[modality].add(parts[1])
                            else:
                                all_used_keys[modality].add(full_key)
        
        # Group metadata and weights by tag
        # Fix: Each tag should only use weights from datasets belonging to that tag
        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        all_weights_by_tag: dict[str, list[float]] = {}
        for idx, dataset in enumerate(self.datasets):
            if dataset.tag not in all_metadatas:
                all_metadatas[dataset.tag] = []
                all_weights_by_tag[dataset.tag] = []
            all_metadatas[dataset.tag].append(dataset.metadata)
            all_weights_by_tag[dataset.tag].append(self.dataset_sampling_weights[idx])
        
        for tag, metadatas in all_metadatas.items():
            # Use only the weights corresponding to datasets in this tag
            tag_weights = all_weights_by_tag[tag]
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=tag_weights,
                percentile_mixing_method=metadata_config["percentile_mixing_method"],
                used_keys=all_used_keys if all_used_keys else None,
            )
        for dataset in self.datasets:
            dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])

    def build_dataset_statistics(self) -> dict:
        """
        Build merged dataset statistics keyed by embodiment tag.
        Only includes statistics for keys that are actually used in the datasets.
        The exported statistics order must match the original concat order.
        """
        statistics_data = {}

        # Collect actually used keys from all datasets
        all_used_action_keys = []
        all_used_state_keys = []
        all_action_normalization_modes: dict[str, str] = {}

        for dataset in self.datasets:
            used_action_keys, used_state_keys = get_used_modality_keys(dataset.modality_keys)
            for used_action_key in used_action_keys:
                if used_action_key not in all_used_action_keys:
                    all_used_action_keys.append(used_action_key)
            for used_state_key in used_state_keys:
                if used_state_key not in all_used_state_keys:
                    all_used_state_keys.append(used_state_key)
            for key, mode in _collect_action_normalization_modes_from_transforms(dataset.transforms).items():
                all_action_normalization_modes.setdefault(key, mode)

        # Organize statistics by tag
        for tag, merged_metadata in self.merged_metadata.items():
            tag_stats = {}

            if hasattr(merged_metadata.statistics, "action") and merged_metadata.statistics.action:
                action_stats = merged_metadata.statistics.action

                filtered_action_stats = {}
                for key in all_used_action_keys:
                    if key in action_stats:
                        filtered_action_stats[key] = action_stats[key]

                if filtered_action_stats:
                    combined_action_stats = combine_modality_stats(filtered_action_stats)

                    mask = generate_action_mask_for_used_keys(
                        merged_metadata.modalities.action,
                        filtered_action_stats.keys(),
                        action_normalization_modes=all_action_normalization_modes,
                    )
                    combined_action_stats["mask"] = mask

                    tag_stats["action"] = combined_action_stats

            if hasattr(merged_metadata.statistics, "state") and merged_metadata.statistics.state:
                state_stats = merged_metadata.statistics.state

                filtered_state_stats = {}
                for key in all_used_state_keys:
                    if key in state_stats:
                        filtered_state_stats[key] = state_stats[key]

                if filtered_state_stats:
                    combined_state_stats = combine_modality_stats(filtered_state_stats)
                    tag_stats["state"] = combined_state_stats

            tag_stats.update(self._get_dataset_counts(tag))
            statistics_data[tag] = tag_stats

        return statistics_data

    def save_dataset_statistics(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save merged dataset statistics to specified path in the required format.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        statistics_data = self.build_dataset_statistics()

        all_used_action_keys = []
        all_used_state_keys = []
        all_action_normalization_modes: dict[str, str] = {}
        for dataset in self.datasets:
            used_action_keys, used_state_keys = get_used_modality_keys(dataset.modality_keys)
            for used_action_key in used_action_keys:
                if used_action_key not in all_used_action_keys:
                    all_used_action_keys.append(used_action_key)
            for used_state_key in used_state_keys:
                if used_state_key not in all_used_state_keys:
                    all_used_state_keys.append(used_state_key)
            for key, mode in _collect_action_normalization_modes_from_transforms(dataset.transforms).items():
                all_action_normalization_modes.setdefault(key, mode)

        if format.lower() == "json":
            if not str(save_path).endswith(".json"):
                save_path = save_path.with_suffix(".json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")

        print(f"Merged dataset statistics saved to: {save_path}")
        print(f"Used action keys: {list(all_used_action_keys)}")
        print(f"Used state keys: {list(all_used_state_keys)}")


    def _combine_modality_stats(self, modality_stats: dict) -> dict:
        """Backward compatibility wrapper."""
        return combine_modality_stats(modality_stats)

    def _generate_action_mask_for_used_keys(self, action_modalities: dict, used_action_keys_ordered) -> list[bool]:
        """Backward compatibility wrapper."""
        return generate_action_mask_for_used_keys(
            action_modalities,
            used_action_keys_ordered,
            action_normalization_modes=_collect_action_normalization_modes_from_datasets(self.datasets),
        )

    def _get_dataset_counts(self, tag: str) -> dict:
        """
        Get dataset count information for specified tag.
        
        Args:
            tag (str): embodiment tag
            
        Returns:
            dict: Dictionary containing num_transitions and num_trajectories
        """
        num_transitions = 0
        num_trajectories = 0
        
        # Count dataset information belonging to this tag
        for dataset in self.datasets:
            if dataset.tag == tag:
                num_transitions += len(dataset)
                num_trajectories += len(dataset.trajectory_ids)
        
        return {
            "num_transitions": num_transitions,
            "num_trajectories": num_trajectories
        }

    @classmethod
    def load_merged_statistics(cls, load_path: Path | str) -> dict:
        """
        Load merged dataset statistics from file.
        
        Args:
            load_path (Path | str): Path to the statistics file
            
        Returns:
            dict: Dictionary containing merged statistics
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Statistics file not found: {load_path}")
        
        if load_path.suffix.lower() == '.json':
            with open(load_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif load_path.suffix.lower() == '.pkl':
            import pickle
            with open(load_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {load_path.suffix}")

    def apply_cached_statistics(self, cached_statistics: dict) -> None:
        """
        Apply cached statistics to avoid recomputation.
        
        Args:
            cached_statistics (dict): Statistics loaded from file
        """
        # Validate that cached statistics match current datasets
        if "metadata" in cached_statistics:
            cached_dataset_names = set(cached_statistics["metadata"]["dataset_names"])
            current_dataset_names = set(dataset.dataset_name for dataset in self.datasets)
            
            if cached_dataset_names != current_dataset_names:
                print("Warning: Cached statistics dataset names don't match current datasets.")
                print(f"Cached: {cached_dataset_names}")
                print(f"Current: {current_dataset_names}")
                return
        
        # Apply cached statistics
        self.merged_metadata = {}
        for tag, stats_data in cached_statistics.items():
            if tag == "metadata":  # Skip metadata field
                continue
                
            # Convert back to DatasetMetadata format
            metadata_dict = {
                "embodiment_tag": tag,
                "statistics": {
                    "action": {},
                    "state": {}
                },
                "modalities": {}
            }
            
            # Convert action statistics back
            if "action" in stats_data:
                action_data = stats_data["action"]
                # This is simplified - you may need to split back to sub-keys
                metadata_dict["statistics"]["action"] = action_data
            
            # Convert state statistics back
            if "state" in stats_data:
                state_data = stats_data["state"]
                metadata_dict["statistics"]["state"] = state_data
            
            self.merged_metadata[tag] = DatasetMetadata.model_validate(metadata_dict)
        
        # Update transforms metadata for each dataset
        for dataset in self.datasets:
            if dataset.tag in self.merged_metadata:
                dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])
        
        print(f"Applied cached statistics for {len(self.merged_metadata)} embodiment tags.")
