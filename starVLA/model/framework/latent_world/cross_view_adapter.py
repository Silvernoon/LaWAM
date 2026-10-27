from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryAwareCrossViewAdapter(nn.Module):
    """Zero-initialized cross-view communication for stitched visual tokens.

    This is a policy-side, PAIWorld-inspired adapter. It does not replace the
    visual tokenizer or claim a 3D prior by itself. Camera rays and poses are
    optional unless ``require_camera_geometry`` is enabled.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        num_views: int,
        grid_hw_per_view: Tuple[int, int],
        num_heads: int = 8,
        dropout: float = 0.0,
        pose_dim: int = 12,
        require_camera_geometry: bool = False,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_views = int(num_views)
        self.grid_height = int(grid_hw_per_view[0])
        self.grid_width = int(grid_hw_per_view[1])
        self.num_heads = int(num_heads)
        self.head_dim = self.feature_dim // self.num_heads
        self.dropout = float(dropout)
        self.pose_dim = int(pose_dim)
        self.require_camera_geometry = bool(require_camera_geometry)

        if self.num_views < 2:
            raise ValueError(f"GeometryAwareCrossViewAdapter requires num_views >= 2, got {self.num_views}.")
        if self.grid_height < 1 or self.grid_width < 1:
            raise ValueError(
                "GeometryAwareCrossViewAdapter requires positive per-view grid dimensions, "
                f"got {(self.grid_height, self.grid_width)}."
            )
        if self.num_heads < 1 or self.feature_dim % self.num_heads != 0:
            raise ValueError(
                f"feature_dim={self.feature_dim} must be divisible by num_heads={self.num_heads}."
            )
        if self.pose_dim < 1:
            raise ValueError(f"pose_dim must be >= 1, got {self.pose_dim}.")

        self.view_embedding = nn.Parameter(torch.empty(1, self.num_views, 1, self.feature_dim))
        nn.init.normal_(self.view_embedding, std=0.02)

        self.ray_encoder = nn.Sequential(
            nn.Linear(3, self.feature_dim),
            nn.SiLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.pose_encoder = nn.Sequential(
            nn.Linear(self.pose_dim, self.feature_dim),
            nn.SiLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.condition_norm = nn.LayerNorm(self.feature_dim)
        self.attention_norm = nn.LayerNorm(self.feature_dim)
        self.query = nn.Linear(self.feature_dim, self.feature_dim)
        self.key = nn.Linear(self.feature_dim, self.feature_dim)
        self.value = nn.Linear(self.feature_dim, self.feature_dim)
        self.attention_output = nn.Linear(self.feature_dim, self.feature_dim, bias=False)

        hidden_dim = self.feature_dim * 4
        self.ffn_norm = nn.LayerNorm(self.feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, self.feature_dim),
        )
        self.ffn_output = nn.Linear(self.feature_dim, self.feature_dim, bias=False)

        # Preserve the pre-adapter policy exactly at initialization.
        nn.init.zeros_(self.attention_output.weight)
        nn.init.zeros_(self.ffn_output.weight)

    @property
    def tokens_per_view(self) -> int:
        return self.grid_height * self.grid_width

    def _prepare_camera_rays(
        self,
        camera_rays: torch.Tensor,
        *,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rays = camera_rays.to(device=device, dtype=dtype)
        if rays.ndim == 6:
            expected_prefix = (batch_size, num_frames, self.num_views)
            if tuple(rays.shape[:3]) != expected_prefix or int(rays.shape[-1]) != 3:
                raise ValueError(
                    "camera_rays must be [B,T,V,H,W,3], "
                    f"got {tuple(rays.shape)} for expected prefix {expected_prefix}."
                )
            ray_h, ray_w = int(rays.shape[3]), int(rays.shape[4])
            rays = rays.reshape(batch_size * num_frames * self.num_views, ray_h, ray_w, 3)
            rays = rays.permute(0, 3, 1, 2).contiguous()
            if (ray_h, ray_w) != (self.grid_height, self.grid_width):
                rays = F.interpolate(
                    rays,
                    size=(self.grid_height, self.grid_width),
                    mode="bilinear",
                    align_corners=False,
                )
            rays = rays.permute(0, 2, 3, 1).reshape(
                batch_size * num_frames,
                self.num_views * self.tokens_per_view,
                3,
            )
        elif rays.ndim == 5:
            expected_prefix = (batch_size, num_frames, self.num_views)
            if tuple(rays.shape[:3]) != expected_prefix or tuple(rays.shape[-2:]) != (self.tokens_per_view, 3):
                raise ValueError(
                    "flattened camera_rays must be [B,T,V,HW,3], "
                    f"got {tuple(rays.shape)} for HW={self.tokens_per_view}."
                )
            rays = rays.reshape(
                batch_size * num_frames,
                self.num_views * self.tokens_per_view,
                3,
            )
        else:
            raise ValueError(
                "camera_rays must be [B,T,V,H,W,3] or [B,T,V,HW,3], "
                f"got {tuple(rays.shape)}."
            )
        return F.normalize(rays, p=2, dim=-1, eps=1e-6)

    def _prepare_camera_poses(
        self,
        camera_poses: torch.Tensor,
        *,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        poses = camera_poses.to(device=device, dtype=dtype)
        expected = (batch_size, num_frames, self.num_views, self.pose_dim)
        if tuple(poses.shape) != expected:
            raise ValueError(f"camera_poses must be [B,T,V,{self.pose_dim}], got {tuple(poses.shape)}.")
        poses = poses.unsqueeze(-2).expand(-1, -1, -1, self.tokens_per_view, -1)
        return poses.reshape(
            batch_size * num_frames,
            self.num_views * self.tokens_per_view,
            self.pose_dim,
        )

    def forward(
        self,
        features: torch.Tensor,
        *,
        camera_rays: Optional[torch.Tensor] = None,
        camera_poses: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"features must be [B,T,K,D], got {tuple(features.shape)}.")
        batch_size, num_frames, num_tokens, feature_dim = map(int, features.shape)
        expected_tokens = self.num_views * self.tokens_per_view
        if num_tokens != expected_tokens or feature_dim != self.feature_dim:
            raise ValueError(
                "cross-view feature layout mismatch: "
                f"got K={num_tokens}, D={feature_dim}; expected K={expected_tokens}, D={self.feature_dim}."
            )
        if self.require_camera_geometry and (camera_rays is None or camera_poses is None):
            raise ValueError(
                "GeometryAwareCrossViewAdapter requires both camera_rays and camera_poses when "
                "require_camera_geometry=True."
            )

        tokens = features.reshape(batch_size * num_frames, num_tokens, feature_dim)
        condition = self.view_embedding.expand(batch_size * num_frames, -1, self.tokens_per_view, -1)
        condition = condition.reshape(batch_size * num_frames, num_tokens, feature_dim)

        if camera_rays is not None:
            rays = self._prepare_camera_rays(
                camera_rays,
                batch_size=batch_size,
                num_frames=num_frames,
                device=features.device,
                dtype=features.dtype,
            )
            condition = condition + self.ray_encoder(rays)
        if camera_poses is not None:
            poses = self._prepare_camera_poses(
                camera_poses,
                batch_size=batch_size,
                num_frames=num_frames,
                device=features.device,
                dtype=features.dtype,
            )
            condition = condition + self.pose_encoder(poses)

        normalized = self.attention_norm(tokens)
        query = self.query(normalized)
        key = self.key(normalized)
        condition = self.condition_norm(condition).to(dtype=query.dtype)
        query = query + condition
        key = key + condition
        value = self.value(normalized)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                batch_size * num_frames,
                num_tokens,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        query_heads = split_heads(query)
        key_heads = split_heads(key)
        value_heads = split_heads(value)
        attended = F.scaled_dot_product_attention(
            query_heads,
            key_heads,
            value_heads,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch_size * num_frames, num_tokens, feature_dim)
        tokens = tokens + self.attention_output(attended)
        tokens = tokens + self.ffn_output(self.ffn(self.ffn_norm(tokens)))
        return tokens.reshape(batch_size, num_frames, num_tokens, feature_dim)
