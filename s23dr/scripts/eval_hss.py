#!/usr/bin/env python3
"""
Evaluate a trained RoofWireframeNet checkpoint on the validation split
using the Hausdorff Segment Score (HSS).

Usage
-----
    # Local dataset (saved with dataset.save_to_disk)
    python s23dr/scripts/eval_hss.py \
        --ckpt s23dr/models/s23dr_last.pt \
        --data-dir s23dr/data

    # Remote HF dataset (requires network + token)
    python s23dr/scripts/eval_hss.py \
        --ckpt s23dr/models/s23dr_last.pt \
        --dataset usm3d/s23dr-2026-sampled_4096_v2

    # Tune confidence threshold
    python s23dr/scripts/eval_hss.py --ckpt s23dr/models/s23dr_last.pt \
        --data-dir s23dr/data --conf-thresh 0.3
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from s23dr.model import RoofWireframeNet, WireframeLoss
from s23dr.metrics import hss, decode_to_segments


# ---------------------------------------------------------------------------
# Dataset that supports both local (save_to_disk) and remote HF datasets
# ---------------------------------------------------------------------------

def _unpack_row(row: dict[str, Any]) -> dict[str, Any]:
    blob = row["data"]
    out: dict[str, Any] = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            if name.endswith(".npy"):
                raw = zf.read(name)
                out[name[:-4]] = np.load(io.BytesIO(raw), allow_pickle=False)
    out["order_id"] = row.get("order_id", "")
    return out


class _LocalS23DRDataset(Dataset):
    def __init__(self, rows: list[dict], n_points: int = 1024) -> None:
        self._rows = [_unpack_row(r) for r in rows]
        self.n_points = n_points

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self._rows[idx]
        N = min(self.n_points, len(r["xyz_norm"]))
        sel = np.arange(N)

        xyz       = torch.from_numpy(r["xyz_norm"][sel]).float()
        vote_frac = torch.from_numpy(r["vote_frac"][sel]).float()
        n_views   = torch.from_numpy(r["n_views_voted"][sel].astype(np.float32)).float()
        mask      = torch.from_numpy(r["mask"][sel].astype(np.float32)).float()
        class_id  = torch.from_numpy(r["class_id"][sel].astype(np.int64))
        gt_verts  = torch.from_numpy(r["gt_vertices"]).float()
        gt_edges  = torch.from_numpy(r["gt_edges"]).long()
        gt_segs   = torch.from_numpy(r["gt_segments"]).float()

        return {
            "order_id": r["order_id"],
            "xyz": xyz, "vote_frac": vote_frac, "n_views": n_views,
            "mask": mask, "class_id": class_id,
            "gt_verts": gt_verts, "gt_edges": gt_edges, "gt_segs": gt_segs,
        }


def _collate(batch: list[dict]) -> dict[str, Any]:
    return {
        "order_id":  [b["order_id"]  for b in batch],
        "xyz":       torch.stack([b["xyz"]       for b in batch]),
        "vote_frac": torch.stack([b["vote_frac"] for b in batch]),
        "n_views":   torch.stack([b["n_views"]   for b in batch]),
        "mask":      torch.stack([b["mask"]      for b in batch]),
        "class_id":  torch.stack([b["class_id"]  for b in batch]),
        "gt_segs":   [b["gt_segs"].numpy() for b in batch],
    }


def load_val_dataset(args) -> Dataset:
    if args.data_dir:
        from datasets import load_from_disk
        hf_ds = load_from_disk(args.data_dir)
        split_ds = hf_ds[args.split] if hasattr(hf_ds, "__getitem__") and args.split in hf_ds else hf_ds
        rows = list(split_ds)
        print(f"Loaded {len(rows)} samples from {args.data_dir} [{args.split}]")
    else:
        from datasets import load_dataset
        hf_ds = load_dataset(args.dataset, split=args.split)
        rows = list(hf_ds)
        print(f"Loaded {len(rows)} samples from HF [{args.split}]")
    return _LocalS23DRDataset(rows, n_points=args.n_points)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str, device: torch.device) -> dict:
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_eval(model: RoofWireframeNet,
             loader: DataLoader,
             loss_fn: WireframeLoss,
             device: torch.device,
             conf_thresh: float) -> dict[str, float]:

    model.eval()
    all_hss, all_prec, all_rec = [], [], []
    t0 = time.time()

    for i, batch in enumerate(loader):
        xyz       = batch["xyz"].to(device)
        vote_frac = batch["vote_frac"].to(device)
        n_views   = batch["n_views"].to(device)
        mask      = batch["mask"].to(device)
        class_id  = batch["class_id"].to(device)

        out = model(xyz, vote_frac, n_views, mask, class_id)

        B = xyz.shape[0]
        for b in range(B):
            verts, edges, _ = loss_fn.decode(
                out["pred_pos"][b],
                out["pred_conf"][b],
                out["edge_logits"][b],
                conf_thresh=conf_thresh,
            )
            pred_segs = decode_to_segments(verts, edges)
            gt_segs   = batch["gt_segs"][b]

            scores = hss(pred_segs, gt_segs)
            all_hss.append(scores["hss"])
            all_prec.append(scores["precision"])
            all_rec.append(scores["recall"])

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            done = (i + 1) * loader.batch_size
            print(f"  [{done}/{len(loader.dataset)}]  "
                  f"mean_hss={np.mean(all_hss):.4f}  ({elapsed:.1f}s)")

    return {
        "hss":       float(np.mean(all_hss)),
        "precision": float(np.mean(all_prec)),
        "recall":    float(np.mean(all_rec)),
        "n_samples": len(all_hss),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt",        required=True, help="Path to .pt checkpoint")
    p.add_argument("--data-dir",    default=None,  help="Local dataset dir (load_from_disk)")
    p.add_argument("--dataset",     default="usm3d/s23dr-2026-sampled_4096_v2")
    p.add_argument("--split",       default="validation", choices=("train", "validation"))
    p.add_argument("--n-points",    type=int,   default=1024)
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--conf-thresh", type=float, default=0.5,
                   help="Confidence threshold for vertex activation (0–1)")
    p.add_argument("--device",      default="cpu")
    p.add_argument("--n-workers",   type=int,   default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    if not args.data_dir and not args.dataset:
        print("Error: provide --data-dir or --dataset", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = load_checkpoint(args.ckpt, device)
    saved_args = ckpt.get("args", {})
    n_queries = saved_args.get("n_queries", 64)
    epoch     = ckpt.get("epoch", "?")
    val_loss  = ckpt.get("val_loss", float("nan"))
    print(f"  epoch={epoch}  val_loss={val_loss:.4f}  n_queries={n_queries}")

    model = RoofWireframeNet(n_queries=n_queries).to(device)
    model.load_state_dict(ckpt["model"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  {total_params:,} parameters")

    loss_fn = WireframeLoss()

    print("Loading dataset…")
    ds = load_val_dataset(args)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.n_workers, collate_fn=_collate)

    print(f"\nRunning HSS evaluation  conf_thresh={args.conf_thresh}  "
          f"device={device}  split={args.split}")
    print("-" * 60)

    results = run_eval(model, loader, loss_fn, device, args.conf_thresh)

    print("-" * 60)
    print(f"HSS        = {results['hss']:.4f}")
    print(f"Precision  = {results['precision']:.4f}")
    print(f"Recall     = {results['recall']:.4f}")
    print(f"Samples    = {results['n_samples']}")


if __name__ == "__main__":
    main()
