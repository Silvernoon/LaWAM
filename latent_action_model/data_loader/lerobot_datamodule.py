import os
from typing import Optional, Union
from functools import partial

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .video_aug import LAM_IMAGE_HW
from .collate import lam_collate
from .lerobot_dataset import LeRobotLAMDataset


def _lam_worker_init_fn(_worker_id: int) -> None:
    _n = os.environ.get("LAM_WORKER_OMP_THREADS", "1")
    os.environ["OMP_NUM_THREADS"] = _n
    os.environ["MKL_NUM_THREADS"] = _n
    os.environ["OPENBLAS_NUM_THREADS"] = _n
    os.environ["NUMEXPR_NUM_THREADS"] = _n
    os.environ["VECLIB_MAXIMUM_THREADS"] = _n
    os.environ["BLIS_NUM_THREADS"] = _n
    n = int(_n) if _n.isdigit() else 1
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(max(1, min(n, 2)))
    except RuntimeError:
        pass


class LeRobotDataModule(LightningDataModule):
    def __init__(
        self,
        data_root_dir: str,
        data_mix: str,
        num_frames: int = 5,
        video_backend: str = "torchvision_av",
        preferred_video_key: Optional[str] = None,
        state_keys: Optional[list[str]] = None,
        image_hw: tuple[int, int] = LAM_IMAGE_HW,
        num_views: int = 1,
        include_wrist_views: bool = False,
        *,
        # Physical-time sampling interval in seconds (required).
        frame_dt_sec: float,
        human_frame_dt_sec: Optional[float] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        prefetch_factor: Optional[int] = None,
        in_order: bool = False,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        debug_repeat_batch: Union[bool, int] = False,
        val_tail_ratio: float = 0.001,
        max_state_dim: int = 14,  # Maximum proprio dimension for padding
    ):
        super().__init__()
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")
        if frame_dt_sec <= 0:
            raise ValueError(f"frame_dt_sec must be > 0, got {frame_dt_sec}")
        if human_frame_dt_sec is not None and human_frame_dt_sec <= 0:
            raise ValueError(
                f"human_frame_dt_sec must be > 0 when provided, got {human_frame_dt_sec}"
            )
        if not (0.0 <= float(val_tail_ratio) < 1.0):
            raise ValueError(f"val_tail_ratio must be in [0.0, 1.0), got {val_tail_ratio}")

        self.data_root_dir = data_root_dir
        self.data_mix = data_mix
        self.num_frames = num_frames
        self.video_backend = video_backend
        self.preferred_video_key = preferred_video_key
        self.state_keys = state_keys
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.num_views = int(num_views)
        self.include_wrist_views = bool(include_wrist_views)
        self.frame_dt_sec = frame_dt_sec
        self.human_frame_dt_sec = human_frame_dt_sec
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.in_order = in_order
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.debug_repeat_batch = debug_repeat_batch
        self.val_tail_ratio = float(val_tail_ratio)
        self.max_state_dim = max_state_dim

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = LeRobotLAMDataset(
                data_root_dir=self.data_root_dir,
                data_mix=self.data_mix,
                num_frames=self.num_frames,
                mode="train",
                val_tail_ratio=self.val_tail_ratio,
                video_backend=self.video_backend,
                preferred_video_key=self.preferred_video_key,
                state_keys=self.state_keys,
                image_hw=self.image_hw,
                num_views=self.num_views,
                include_wrist_views=self.include_wrist_views,
                frame_dt_sec=self.frame_dt_sec,
                human_frame_dt_sec=self.human_frame_dt_sec,
                debug_repeat_batch=self.debug_repeat_batch,
            )
        if stage in (None, "fit", "validate"):
            if self.val_tail_ratio > 0:
                self.val_dataset = LeRobotLAMDataset(
                    data_root_dir=self.data_root_dir,
                    data_mix=self.data_mix,
                    num_frames=self.num_frames,
                    mode="val",
                    val_tail_ratio=self.val_tail_ratio,
                    video_backend=self.video_backend,
                    preferred_video_key=self.preferred_video_key,
                    state_keys=self.state_keys,
                    image_hw=self.image_hw,
                    num_views=self.num_views,
                    include_wrist_views=self.include_wrist_views,
                    frame_dt_sec=self.frame_dt_sec,
                    human_frame_dt_sec=self.human_frame_dt_sec,
                    debug_repeat_batch=False,
                )
            else:
                self.val_dataset = None

    def train_dataloader(self):
        # Use partial to pass max_state_dim to collate function
        collate_fn = partial(lam_collate, max_state_dim=self.max_state_dim)
        use_persistent_workers = self.num_workers > 0 and self.persistent_workers
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            worker_init_fn=_lam_worker_init_fn if self.num_workers > 0 else None,
            persistent_workers=use_persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            in_order=self.in_order if self.num_workers > 0 else True,
        )

    def set_mixture_epoch(self, epoch: int) -> bool:
        """Set epoch on train mixture dataset if available.

        Returns:
            bool: True if epoch was applied, False otherwise.
        """
        if self.train_dataset is None or not hasattr(self.train_dataset, "mixture"):
            return False
        mixture = getattr(self.train_dataset, "mixture", None)
        if mixture is None or not hasattr(mixture, "set_epoch"):
            return False
        mixture.set_epoch(int(epoch))
        return True

    def on_train_epoch_start(self, trainer) -> None:
        """Compatibility shim for manual callers.

        Note: Lightning does not automatically invoke `LightningDataModule.on_train_epoch_start`.
        Epoch sync is triggered from `VJEPA_LAM.on_train_epoch_start`.
        """
        self.set_mixture_epoch(getattr(trainer, "current_epoch", 0))

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError(
                "Validation dataset is not initialized. Set data.val_tail_ratio > 0 to enable validation split."
            )

        collate_fn = partial(lam_collate, max_state_dim=self.max_state_dim)
        use_persistent_workers = self.num_workers > 0 and self.persistent_workers
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            worker_init_fn=_lam_worker_init_fn if self.num_workers > 0 else None,
            persistent_workers=use_persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            in_order=self.in_order if self.num_workers > 0 else True,
        )
