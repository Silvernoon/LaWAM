from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenDA3GeometryProvider(nn.Module):
    """Estimate per-frame multi-view camera geometry with frozen Depth Anything 3."""

    def __init__(
        self,
        *,
        model_id: Optional[str] = None,
        model: Optional[nn.Module] = None,
        input_size: int = 252,
        grid_hw_per_view: Tuple[int, int],
        use_ray_pose: bool = False,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.grid_height = int(grid_hw_per_view[0])
        self.grid_width = int(grid_hw_per_view[1])
        self.use_ray_pose = bool(use_ray_pose)

        if self.input_size < 14 or self.input_size % 14 != 0:
            raise ValueError(f"DA3 input_size must be a positive multiple of 14, got {self.input_size}.")
        if self.grid_height < 1 or self.grid_width < 1:
            raise ValueError(f"DA3 output grid must be positive, got {grid_hw_per_view}.")
        if model is None:
            if not model_id:
                raise ValueError("FrozenDA3GeometryProvider requires either model_id or model.")
            model = self._load_model(str(model_id), cache_dir=cache_dir)

        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _load_model(model_id: str, *, cache_dir: Optional[str | Path]) -> nn.Module:
        try:
            from depth_anything_3.cfg import create_object
            from omegaconf import OmegaConf
            from safetensors.torch import load_model
        except ImportError as exc:
            raise ImportError(
                "Depth Anything 3 core dependencies are required. Install the official package and `addict`; "
                "the geometry provider does not require DA3's visualization/export dependencies."
            ) from exc

        model_path = Path(model_id).expanduser()
        if not model_path.is_dir():
            from huggingface_hub import snapshot_download

            model_path = Path(
                snapshot_download(
                    repo_id=model_id,
                    cache_dir=None if cache_dir is None else str(cache_dir),
                    allow_patterns=["config.json", "model.safetensors"],
                )
            )
        config_path = model_path / "config.json"
        weights_path = model_path / "model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                f"DA3 model directory must contain config.json and model.safetensors, got {model_path}."
            )

        with config_path.open("r", encoding="utf-8") as stream:
            model_config = json.load(stream)
        architecture_config = model_config.get("config")
        if not isinstance(architecture_config, dict):
            raise ValueError(f"DA3 config.json does not contain an architecture `config`: {config_path}.")
        model = create_object(OmegaConf.create(architecture_config))
        checkpoint_wrapper = nn.Module()
        checkpoint_wrapper.add_module("model", model)
        load_model(checkpoint_wrapper, str(weights_path), strict=True, device="cpu")
        return model

    def train(self, mode: bool = True) -> "FrozenDA3GeometryProvider":
        del mode
        super().train(False)
        return self

    @staticmethod
    def _extract_prediction(output: Any, key: str) -> torch.Tensor:
        if isinstance(output, dict):
            value = output.get(key)
        else:
            value = getattr(output, key, None)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"Depth Anything 3 output does not contain tensor `{key}`.")
        return value

    @staticmethod
    def _homogeneous_extrinsics(extrinsics: torch.Tensor) -> torch.Tensor:
        if extrinsics.ndim != 4 or tuple(extrinsics.shape[-2:]) not in ((3, 4), (4, 4)):
            raise ValueError(f"DA3 extrinsics must be [B,V,3,4] or [B,V,4,4], got {tuple(extrinsics.shape)}.")
        if tuple(extrinsics.shape[-2:]) == (4, 4):
            return extrinsics
        bottom = torch.zeros(*extrinsics.shape[:-2], 1, 4, device=extrinsics.device, dtype=extrinsics.dtype)
        bottom[..., 0, 3] = 1.0
        return torch.cat((extrinsics, bottom), dim=-2)

    @staticmethod
    def _rotation_to_euler_xyz(rotation: torch.Tensor) -> torch.Tensor:
        sy = torch.sqrt(rotation[..., 0, 0].square() + rotation[..., 1, 0].square())
        singular = sy < 1e-6
        roll_regular = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
        pitch = torch.atan2(-rotation[..., 2, 0], sy)
        yaw_regular = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
        roll_singular = torch.atan2(-rotation[..., 1, 2], rotation[..., 1, 1])
        yaw_singular = torch.zeros_like(yaw_regular)
        roll = torch.where(singular, roll_singular, roll_regular)
        yaw = torch.where(singular, yaw_singular, yaw_regular)
        return torch.stack((yaw, pitch, roll), dim=-1)

    def _camera_rays(self, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        batch_views = int(extrinsics.shape[0] * extrinsics.shape[1])
        rotation = extrinsics[..., :3, :3].reshape(batch_views, 3, 3)
        inverse_intrinsics = torch.linalg.inv(intrinsics.reshape(batch_views, 3, 3))

        y = (torch.arange(self.grid_height, device=extrinsics.device, dtype=extrinsics.dtype) + 0.5)
        x = (torch.arange(self.grid_width, device=extrinsics.device, dtype=extrinsics.dtype) + 0.5)
        y = y * (float(self.input_size) / float(self.grid_height))
        x = x * (float(self.input_size) / float(self.grid_width))
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        pixels = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=-1)
        pixels = pixels.reshape(1, self.grid_height * self.grid_width, 3).expand(batch_views, -1, -1)

        camera_rays = torch.matmul(inverse_intrinsics, pixels.transpose(1, 2)).transpose(1, 2)
        world_rays = torch.matmul(rotation.transpose(1, 2), camera_rays.transpose(1, 2)).transpose(1, 2)
        world_rays = F.normalize(world_rays, p=2, dim=-1, eps=1e-6)
        return world_rays.reshape(
            extrinsics.shape[0],
            extrinsics.shape[1],
            self.grid_height,
            self.grid_width,
            3,
        )

    def _camera_poses(self, extrinsics: torch.Tensor) -> torch.Tensor:
        rotation = extrinsics[..., :3, :3]
        translation = extrinsics[..., :3, 3]
        euler = self._rotation_to_euler_xyz(rotation)
        rotation_t = rotation.transpose(-1, -2)
        camera_position = -torch.matmul(rotation_t, translation.unsqueeze(-1)).squeeze(-1)
        optical_axis_camera = torch.zeros_like(translation)
        optical_axis_camera[..., 2] = 1.0
        optical_axis_world = torch.matmul(rotation_t, optical_axis_camera.unsqueeze(-1)).squeeze(-1)
        return torch.cat((euler, translation, camera_position, optical_axis_world), dim=-1)

    @torch.inference_mode()
    def forward(self, stereo_video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if stereo_video.ndim != 6:
            raise ValueError(f"DA3 geometry input must be [B,V,T,C,H,W], got {tuple(stereo_video.shape)}.")
        batch_size, num_views, num_frames, channels, height, width = map(int, stereo_video.shape)
        if channels != 3:
            raise ValueError(f"DA3 geometry input must be RGB, got C={channels}.")

        images = stereo_video.permute(0, 2, 1, 3, 4, 5).reshape(
            batch_size * num_frames * num_views,
            channels,
            height,
            width,
        )
        images = F.interpolate(
            images.to(dtype=torch.float32),
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        images = images.reshape(batch_size * num_frames, num_views, channels, self.input_size, self.input_size)
        output = self.model(
            images,
            export_feat_layers=[],
            use_ray_pose=self.use_ray_pose,
        )
        extrinsics = self._homogeneous_extrinsics(self._extract_prediction(output, "extrinsics")).float()
        intrinsics = self._extract_prediction(output, "intrinsics").float()
        expected_intrinsics = (batch_size * num_frames, num_views, 3, 3)
        if tuple(intrinsics.shape) != expected_intrinsics:
            raise ValueError(
                f"DA3 intrinsics must be {expected_intrinsics}, got {tuple(intrinsics.shape)}."
            )
        if tuple(extrinsics.shape[:2]) != (batch_size * num_frames, num_views):
            raise ValueError(
                "DA3 extrinsics batch/view mismatch: "
                f"got {tuple(extrinsics.shape[:2])}, expected {(batch_size * num_frames, num_views)}."
            )

        rays = self._camera_rays(extrinsics, intrinsics).reshape(
            batch_size,
            num_frames,
            num_views,
            self.grid_height,
            self.grid_width,
            3,
        )
        poses = self._camera_poses(extrinsics).reshape(batch_size, num_frames, num_views, 12)
        return rays.to(device=stereo_video.device), poses.to(device=stereo_video.device)
