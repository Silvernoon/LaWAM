from typing import Any, Dict, Sequence

import torch


def lam_collate(batch: Sequence[Dict], max_state_dim: int = 32) -> Dict[str, Any]:
    """
    Collate fixed-format clips and proprio into a batch.

    Input contract (upstream dataset must guarantee):
    - Stereo sample: {"frames": Tensor [V,T,C,H,W] uint8}; single-view sample: [T,C,H,W].
    - Within a batch all samples share their video shape and proprio T.
      If D > max_state_dim, proprio is truncated to max_state_dim.

    Returns:
        videos: [B,V,T,C,H,W] for stereo or [B,T,C,H,W] for single-view
        proprio: [B,2,max_state_dim] float32 (compat alias for states)
        delta_proprio: [B,max_state_dim] float32
        embodiment_ids: [B] int64
        proprio_mask: [B,max_state_dim] float32 (compat dim-only mask)
        dataset_names: list[str] (optional sample provenance)
        trajectory_ids: list[Union[int, str]] (optional sample provenance)
        base_indices: [B] int64 (optional sample provenance)
    """
    if len(batch) == 0:
        raise ValueError("lam_collate received an empty batch.")

    batch_size = len(batch)
    videos = torch.stack([s["frames"] for s in batch], dim=0)
    states_t = torch.zeros((batch_size, 2, max_state_dim), dtype=torch.float32)
    state_mask_t = torch.zeros((batch_size, 2, max_state_dim), dtype=torch.bool)
    delta_t = torch.zeros((batch_size, max_state_dim), dtype=torch.float32)
    proprio_mask_t = torch.zeros((batch_size, max_state_dim), dtype=torch.float32)
    for i, sample in enumerate(batch):
        proprio = sample["proprio"]
        if not isinstance(proprio, torch.Tensor):
            proprio = torch.as_tensor(proprio)
        proprio = proprio.to(torch.float32)
        if proprio.ndim != 2:
            raise ValueError(
                f"Expected sample['proprio'] to have shape [T, D], got {tuple(proprio.shape)} "
                f"for sample index {i}."
            )
        D = int(proprio.shape[1])
        if D <= 0:
            raise ValueError(f"Invalid proprio dim D={D} for sample index {i}.")
        if D > max_state_dim:
            proprio = proprio[:, :max_state_dim]
            D = int(max_state_dim)

        start = proprio[0, :]
        end = proprio[-1, :]
        states_t[i, 0, :D] = start
        states_t[i, 1, :D] = end
        state_mask_t[i, :, :D] = True
        delta_t[i, :D] = end - start
        proprio_mask_t[i, :D] = 1.0

    embodiment_ids_t = torch.tensor(
        [s["embodiment_id"] for s in batch],
        dtype=torch.long,
    )
    dataset_names = [str(s.get("dataset_name", "unknown")) for s in batch]
    trajectory_ids = [s.get("trajectory_id", -1) for s in batch]
    base_indices = torch.tensor(
        [int(s.get("base_index", -1)) for s in batch],
        dtype=torch.long,
    )

    return {
        "videos": videos,
        "states": states_t,
        "state_mask": state_mask_t,
        "proprio": states_t,
        "delta_proprio": delta_t,
        "embodiment_ids": embodiment_ids_t,
        "proprio_mask": proprio_mask_t,
        "dataset_names": dataset_names,
        "trajectory_ids": trajectory_ids,
        "base_indices": base_indices,
    }
