from __future__ import annotations

from typing import Any, Sequence

import torch

from starVLA.model.framework.latent_world.batch_utils import (
    align_feature_dim_with_mask,
    apply_shared_color_jitter_uint8,
    apply_shared_random_resized_crop_uint8,
    build_placeholder_masks,
    imagenet_normalize_video_,
    sample_or_pad_sequence_with_mask,
)
from starVLA.model.framework.latent_world.processor_utils import (
    LatentWorldProcessorSpec,
    load_latent_world_processor,
)
from starVLA.model.framework.latent_world.types import LatentWorldPolicyTrainBatch, LatentWorldPolicyTrainRawSample
from starVLA.model.framework.latent_world.vlm_adapter import build_qwenvl_messages
from starVLA.model.framework.latent_world.vlm_adapter import (
    DEFAULT_LATENT_WORLD_POLICY_COT_PROMPT,
    DEFAULT_LATENT_WORLD_TEMPORAL_COT_PROMPT,
)


class LatentWorldTrainCollator:
    def __init__(
        self,
        *,
        policy_cfg: Any,
        processor_spec: LatentWorldProcessorSpec,
        act_queries: int,
        flow_queries: int,
        enable_primary_video_aug: bool,
        enable_primary_random_resized_crop: bool,
        cot_prompt_before_wrist: str | None = None,
        cot_prompt_after_wrist: str | None = None,
    ) -> None:
        self.policy_cfg = policy_cfg
        self.processor_spec = processor_spec
        self.placeholder_token = str(processor_spec.placeholder_token)
        self.act_queries = int(act_queries)
        self.flow_queries = int(flow_queries)
        self.enable_primary_video_aug = bool(enable_primary_video_aug)
        self.stereo_num_views = int(getattr(policy_cfg, "stereo_num_views", 1))
        self.stereo_include_wrist_views = bool(getattr(policy_cfg, "stereo_include_wrist_views", False))
        if self.stereo_num_views < 1:
            raise ValueError(f"stereo_num_views must be >= 1, got {self.stereo_num_views}.")
        self.enable_primary_random_resized_crop = bool(enable_primary_random_resized_crop)
        self.cot_prompt_before_wrist = (
            str(cot_prompt_before_wrist)
            if cot_prompt_before_wrist is not None
            else DEFAULT_LATENT_WORLD_TEMPORAL_COT_PROMPT
        )
        self.cot_prompt_after_wrist = (
            str(cot_prompt_after_wrist)
            if cot_prompt_after_wrist is not None
            else DEFAULT_LATENT_WORLD_POLICY_COT_PROMPT
        )
        self.training = True
        self._processor: Any | None = None
        self._placeholder_token_id: int | None = None

    def train(self) -> "LatentWorldTrainCollator":
        self.training = True
        return self

    def eval(self) -> "LatentWorldTrainCollator":
        self.training = False
        return self

    def _ensure_processor(self) -> tuple[Any, int]:
        if self._processor is None:
            processor, _, placeholder_token_id = load_latent_world_processor(self.processor_spec)
            self._processor = processor
            self._placeholder_token_id = placeholder_token_id
        assert self._processor is not None
        assert self._placeholder_token_id is not None
        return self._processor, self._placeholder_token_id

    @staticmethod
    def _to_wrist_view_list(wrist_videos: torch.Tensor) -> list[torch.Tensor]:
        if int(wrist_videos.shape[0]) == 0:
            return []
        if wrist_videos.ndim != 5 or int(wrist_videos.shape[1]) < 1:
            raise ValueError(
                "Expected wrist_videos with shape [V,T,C,H,W] and T>=1, "
                f"got {tuple(wrist_videos.shape)}."
            )
        return [video[0].contiguous() for video in wrist_videos]

    @staticmethod
    def _to_primary_view_list(primary_videos: torch.Tensor) -> list[torch.Tensor]:
        if primary_videos.ndim != 5 or int(primary_videos.shape[1]) < 1:
            raise ValueError(
                "Expected primary_videos with shape [V, T, C, H, W] and T>=1, "
                f"got {tuple(primary_videos.shape)}."
            )
        return [video[0].contiguous() for video in primary_videos]

    @staticmethod
    def _augment_primary_and_wrist_views(
        *,
        primary_videos: torch.Tensor,
        wrist_videos: torch.Tensor,
        enable_random_resized_crop: bool,
        enable_color_jitter: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if primary_videos.ndim != 5 or wrist_videos.ndim != 5:
            raise ValueError(
                "Expected primary_videos and wrist_videos as [V,T,C,H,W], got "
                f"{tuple(primary_videos.shape)} and {tuple(wrist_videos.shape)}."
            )
        primary_flat_count = int(primary_videos.shape[0] * primary_videos.shape[1])
        flat_primary = primary_videos.reshape(primary_flat_count, *primary_videos.shape[-3:])
        flat_wrist_count = int(wrist_videos.shape[0] * wrist_videos.shape[1])
        stacked_views = [flat_primary]
        if flat_wrist_count > 0:
            stacked_views.append(wrist_videos.reshape(flat_wrist_count, *wrist_videos.shape[-3:]))
        augmented_views = torch.cat(stacked_views, dim=0)
        if enable_random_resized_crop:
            augmented_views = apply_shared_random_resized_crop_uint8(augmented_views)
        if enable_color_jitter:
            augmented_views = apply_shared_color_jitter_uint8(augmented_views)
        processed_primary = augmented_views[:primary_flat_count].reshape_as(primary_videos).contiguous()
        processed_wrist = augmented_views[primary_flat_count:].reshape_as(wrist_videos).contiguous()
        return processed_primary, processed_wrist

    def _select_stereo_videos(
        self,
        primary_videos: torch.Tensor,
        wrist_videos: torch.Tensor,
    ) -> torch.Tensor:
        views = list(primary_videos.unbind(0))
        if self.stereo_include_wrist_views:
            views.extend(wrist_videos.unbind(0))
        if len(views) < self.stereo_num_views:
            raise ValueError(
                f"LaWAM stereo input requires {self.stereo_num_views} views, but only {len(views)} are available "
                f"(primary={primary_videos.shape[0]}, wrist={wrist_videos.shape[0]}, "
                f"include_wrist={self.stereo_include_wrist_views})."
            )
        target_frames = int(primary_videos.shape[1])
        aligned_views: list[torch.Tensor] = []
        for view in views[: self.stereo_num_views]:
            frame_count = int(view.shape[0])
            if frame_count < 1:
                raise ValueError("LaWAM stereo view contains no frames.")
            if frame_count < target_frames:
                padding = view[-1:].expand(target_frames - frame_count, *view.shape[1:])
                view = torch.cat([view, padding], dim=0)
            elif frame_count > target_frames:
                indices = torch.linspace(0, frame_count - 1, target_frames).round().long()
                view = view.index_select(0, indices)
            aligned_views.append(view.contiguous())
        return torch.stack(aligned_views, dim=0).contiguous()

    def __call__(self, features: Sequence[LatentWorldPolicyTrainRawSample]) -> LatentWorldPolicyTrainBatch:
        if len(features) == 0:
            raise ValueError("LatentWorldTrainCollator received an empty feature list.")

        processor, placeholder_token_id = self._ensure_processor()

        image_views_batch: list[list[Any]] = []
        wrist_image_views_batch: list[list[Any]] = []
        stereo_video_uint8_seqs: list[torch.Tensor] = []
        instructions: list[str] = []
        actions_list: list[torch.Tensor] = []
        camera_rays_list: list[torch.Tensor] = []
        camera_poses_list: list[torch.Tensor] = []
        camera_rays_present: bool | None = None
        camera_poses_present: bool | None = None
        states_list: list[torch.Tensor] = []
        embodiment_ids: list[int] = []
        action_hz_list: list[float] = []

        for sample in features:
            primary_videos = sample["primary_videos"].contiguous()
            wrist_videos = sample["wrist_videos"].contiguous()
            state_tensor = sample["state"].to(dtype=torch.float32).contiguous()
            action_tensor = sample["action"].to(dtype=torch.float32).contiguous()
            sample_has_rays = "camera_rays" in sample
            sample_has_poses = "camera_poses" in sample
            if camera_rays_present is None:
                camera_rays_present = sample_has_rays
            elif camera_rays_present != sample_has_rays:
                raise ValueError("camera_rays must be present for every sample or none of the samples.")
            if camera_poses_present is None:
                camera_poses_present = sample_has_poses
            elif camera_poses_present != sample_has_poses:
                raise ValueError("camera_poses must be present for every sample or none of the samples.")
            if sample_has_rays:
                camera_rays_list.append(torch.as_tensor(sample["camera_rays"], dtype=torch.float32).contiguous())
            if sample_has_poses:
                camera_poses_list.append(torch.as_tensor(sample["camera_poses"], dtype=torch.float32).contiguous())

            instruction = str(sample["lang"])
            embodiment_id = int(sample["embodiment_id"])
            action_hz = float(sample["action_hz"])

            processed_primary_videos = primary_videos
            processed_wrist_videos = wrist_videos
            if self.training and (self.enable_primary_random_resized_crop or self.enable_primary_video_aug):
                processed_primary_videos, processed_wrist_videos = self._augment_primary_and_wrist_views(
                    primary_videos=primary_videos,
                    wrist_videos=wrist_videos,
                    enable_random_resized_crop=self.enable_primary_random_resized_crop,
                    enable_color_jitter=self.enable_primary_video_aug,
                )

            image_views_batch.append(self._to_primary_view_list(processed_primary_videos))
            wrist_image_views_batch.append(self._to_wrist_view_list(processed_wrist_videos))
            stereo_video_uint8_seqs.append(
                self._select_stereo_videos(processed_primary_videos, processed_wrist_videos)
            )
            instructions.append(instruction)
            actions_list.append(action_tensor)
            states_list.append(state_tensor)
            embodiment_ids.append(embodiment_id)
            action_hz_list.append(action_hz)

        qwen_messages = build_qwenvl_messages(
            images=image_views_batch,
            wrist_images=wrist_image_views_batch,
            instructions=instructions,
            placeholder_token=self.placeholder_token,
            act_queries=self.act_queries,
            flow_queries=self.flow_queries,
            cot_prompt_before_wrist=self.cot_prompt_before_wrist,
            cot_prompt_after_wrist=self.cot_prompt_after_wrist,
        )
        qwen_inputs = processor.apply_chat_template(
            qwen_messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = qwen_inputs["input_ids"]
        act_mask, flow_mask = build_placeholder_masks(
            input_ids,
            act_queries=self.act_queries,
            flow_queries=self.flow_queries,
            placeholder_id=placeholder_token_id,
        )

        action_dim = int(self.policy_cfg.flow_cfg.action_dim)
        window_size = int(self.policy_cfg.action_horizon)
        state_dim = int(self.policy_cfg.flow_cfg.state_dim)

        action_tensors = []
        action_masks = []
        state_tensors = []
        state_masks = []
        for idx, state_tensor in enumerate(states_list):
            action_tensor = actions_list[idx]
            action_tensor, action_dim_mask = align_feature_dim_with_mask(action_tensor, action_dim)
            action_tensor, action_time_mask = sample_or_pad_sequence_with_mask(action_tensor, window_size)
            action_tensors.append(action_tensor)
            action_masks.append(action_time_mask.unsqueeze(1) & action_dim_mask.unsqueeze(0))

            aligned_state, state_dim_mask = align_feature_dim_with_mask(state_tensor, state_dim)
            state_tensors.append(aligned_state[-1].contiguous())
            state_masks.append(state_dim_mask.contiguous())

        stereo_video_uint8_batch = torch.stack(stereo_video_uint8_seqs, dim=0)
        stereo_video_batch = stereo_video_uint8_batch.to(dtype=torch.float32).div_(255.0)
        batch_size, num_views, num_frames = stereo_video_batch.shape[:3]
        flat_stereo_video_batch = stereo_video_batch.reshape(
            batch_size * num_views, num_frames, *stereo_video_batch.shape[3:]
        )
        imagenet_normalize_video_(flat_stereo_video_batch)
        if self.stereo_num_views == 1:
            stereo_video_batch = stereo_video_batch[:, 0]

        batch: LatentWorldPolicyTrainBatch = {
            "pixel_values": qwen_inputs["pixel_values"],
            "input_ids": input_ids,
            "attention_mask": qwen_inputs["attention_mask"],
            "act_placeholder_mask": act_mask,
            "flow_placeholder_mask": flow_mask,
            "stereo_video": stereo_video_batch,
            "state": torch.stack(state_tensors, dim=0),
            "state_mask": torch.stack(state_masks, dim=0),
            "embodiment_id": torch.tensor(embodiment_ids, dtype=torch.long),
            "action_hz": torch.tensor(action_hz_list, dtype=torch.float32),
            "image_grid_thw": qwen_inputs.get("image_grid_thw"),
            "actions": torch.stack(action_tensors, dim=0),
            "actions_mask": torch.stack(action_masks, dim=0),
        }
        if not torch.is_tensor(batch["image_grid_thw"]):
            batch["image_grid_thw"] = None
        if camera_rays_present:
            if self.training and self.enable_primary_random_resized_crop:
                raise ValueError(
                    "Camera rays cannot be used with random resized crop unless the geometry is "
                    "transformed with the same operation."
                )
            batch["camera_rays"] = torch.stack(camera_rays_list, dim=0)
        if camera_poses_present:
            batch["camera_poses"] = torch.stack(camera_poses_list, dim=0)
        return batch
