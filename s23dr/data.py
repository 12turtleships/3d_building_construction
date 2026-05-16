"""
PyTorch Dataset for the S23DR 2026 HF dataset.

Each sample is a ZIP payload containing .npy arrays.
This module caches the full split in memory on first access (the sampled
dataset is small enough) and returns tensors ready for the model.
"""

from __future__ import annotations

import io
import zipfile
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


def _unpack_row(row: dict[str, Any]) -> dict[str, np.ndarray]:
    blob = row["data"]
    out: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            if name.endswith(".npy"):
                raw = zf.read(name)
                key = name.replace(".npy", "")
                out[key] = np.load(io.BytesIO(raw), allow_pickle=False)
    out["order_id"] = row.get("order_id", "")
    return out


class S23DRDataset(Dataset):
    """
    Parameters
    ----------
    split         : "train" or "validation"
    dataset_id    : HF dataset identifier
    n_points      : number of points to use per sample.
                    If < 4096, points are randomly subsampled each __getitem__
                    call (acts as data augmentation during training).
                    Validation uses a deterministic first-N slice.
    streaming     : if True, load on-the-fly (no caching)
    """

    def __init__(self,
                 split: str = "train",
                 dataset_id: str = "usm3d/s23dr-2026-sampled_4096_v2",
                 n_points: int = 1024,
                 streaming: bool = False) -> None:
        self.split = split
        self.n_points = n_points
        self.is_train = (split == "train")

        ds = load_dataset(dataset_id, split=split, streaming=streaming)
        self._rows = [_unpack_row(r) for r in ds]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self._rows[idx]
        total = len(r["xyz_norm"])
        N = min(self.n_points, total)

        # Random subsample for train (augmentation), deterministic for val
        if self.is_train and N < total:
            sel = np.random.choice(total, N, replace=False)
            sel.sort()
        else:
            sel = np.arange(N)

        # --- inputs ---
        xyz       = torch.from_numpy(r["xyz_norm"][sel]).float()           # (N, 3)
        vote_frac = torch.from_numpy(r["vote_frac"][sel]).float()          # (N,)
        n_views   = torch.from_numpy(
            r["n_views_voted"][sel].astype(np.float32)).float()            # (N,)
        mask      = torch.from_numpy(
            r["mask"][sel].astype(np.float32)).float()                      # (N,)
        class_id  = torch.from_numpy(
            r["class_id"][sel].astype(np.int64))                            # (N,)

        # --- targets (always full, not subsampled) ---
        gt_edges   = torch.from_numpy(r["gt_edges"]).long()                # (E, 2)
        gt_classes = torch.from_numpy(r["gt_edge_classes"]).long()         # (E,)
        gt_segs    = torch.from_numpy(r["gt_segments"]).float()            # (E, 2, 3)

        # World-space denormalization: world = xyz_norm * scale + center
        # gt_vertices is stored in world space; convert to normalised space so
        # pred_pos (which lives in xyz_norm space) can be directly compared.
        scale  = float(r["scale"])
        center = torch.from_numpy(r["center"].astype(np.float32))          # (3,)
        gt_verts = (torch.from_numpy(r["gt_vertices"]).float() - center) / scale  # (V, 3)

        return {
            "order_id":   r["order_id"],
            "xyz":        xyz,
            "vote_frac":  vote_frac,
            "n_views":    n_views,
            "mask":       mask,
            "class_id":   class_id,
            "gt_verts":   gt_verts,
            "gt_edges":   gt_edges,
            "gt_classes": gt_classes,
            "gt_segs":    gt_segs,
            "scale":      scale,
            "center":     center,
        }


def collate_fn(batch: list[dict]) -> dict[str, Any]:
    """
    Stack per-point tensors; keep GT lists as lists (variable-length per sample).
    """
    return {
        "order_id":  [b["order_id"]  for b in batch],
        "xyz":       torch.stack([b["xyz"]       for b in batch]),
        "vote_frac": torch.stack([b["vote_frac"] for b in batch]),
        "n_views":   torch.stack([b["n_views"]   for b in batch]),
        "mask":      torch.stack([b["mask"]      for b in batch]),
        "class_id":  torch.stack([b["class_id"]  for b in batch]),
        # Variable-length targets: kept as lists
        "gt_verts":   [b["gt_verts"]   for b in batch],
        "gt_edges":   [b["gt_edges"]   for b in batch],
        "gt_classes": [b["gt_classes"] for b in batch],
        "gt_segs":    [b["gt_segs"]    for b in batch],
        "scale":      [b["scale"]      for b in batch],
        "center":     [b["center"]     for b in batch],
    }
