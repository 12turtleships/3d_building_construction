#!/usr/bin/env python3
"""
Evaluate a trained RoofWireframeNet checkpoint on the S23DR validation set.

Computes:
  - Mean validation loss (vert / conf / edge)
  - HSS (Hausdorff Segment Score) per sample, then averaged

Usage
-----
    python s23dr/scripts/evaluate.py --ckpt outputs/checkpoints/best.pt
    python s23dr/scripts/evaluate.py --ckpt outputs/checkpoints/best.pt --tau 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from s23dr.data    import S23DRDataset, collate_fn
from s23dr.metrics import hss, decode_to_segments
from s23dr.model   import RoofWireframeNet, WireframeLoss


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt",       required=True, help="Path to best.pt checkpoint")
    p.add_argument("--split",      default="validation", choices=("train", "validation"))
    p.add_argument("--n-points",   type=int,   default=1024)
    p.add_argument("--batch-size", type=int,   default=4)
    p.add_argument("--n-workers",  type=int,   default=0)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--conf-thresh",type=float, default=0.5)
    p.add_argument("--tau",        type=float, default=0.2,
                   help="HSS distance threshold (normalised space)")
    p.add_argument("--max-samples",type=int,   default=None,
                   help="Limit evaluation to first N samples (for quick checks)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    n_queries = saved_args.get("n_queries", 64)

    model = RoofWireframeNet(n_queries=n_queries).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss', '?'):.4f}")

    loss_fn = WireframeLoss()

    print(f"Loading {args.split} split…")
    ds = S23DRDataset(split=args.split, n_points=args.n_points)
    if args.max_samples:
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(min(args.max_samples, len(ds)))))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.n_workers, collate_fn=collate_fn)

    total_loss = total_vert = total_conf = total_edge = 0.0
    hss_scores = []

    with torch.no_grad():
        for batch in loader:
            xyz      = batch["xyz"].to(device)
            vf       = batch["vote_frac"].to(device)
            nv       = batch["n_views"].to(device)
            msk      = batch["mask"].to(device)
            cid      = batch["class_id"].to(device)

            out = model(xyz, vf, nv, msk, cid)
            losses = loss_fn(
                out["pred_pos"], out["pred_conf"], out["edge_logits"],
                batch["gt_verts"], batch["gt_edges"], batch["gt_classes"],
            )
            total_loss += losses["loss"].item()
            total_vert += losses["loss_vert"].item()
            total_conf += losses["loss_conf"].item()
            total_edge += losses["loss_edge"].item()

            # Per-sample HSS in normalised space
            B = out["pred_pos"].shape[0]
            for b in range(B):
                verts, edges, _ = loss_fn.decode(
                    out["pred_pos"][b],
                    out["pred_conf"][b],
                    out["edge_logits"][b],
                    conf_thresh=args.conf_thresh,
                )
                pred_segs = decode_to_segments(verts, edges)
                gt_segs   = batch["gt_segs"][b].numpy()
                s = hss(pred_segs, gt_segs, tau=args.tau)
                hss_scores.append(s["hss"])

    n = len(loader)
    print(f"\n{'─'*50}")
    print(f"Samples   : {len(hss_scores)}")
    print(f"Loss      : {total_loss/n:.4f}  "
          f"(vert={total_vert/n:.4f}  conf={total_conf/n:.4f}  edge={total_edge/n:.4f})")
    print(f"HSS       : {np.mean(hss_scores):.4f}  "
          f"(τ={args.tau}, conf_thresh={args.conf_thresh})")
    print(f"HSS p25/p50/p75 : "
          f"{np.percentile(hss_scores,25):.3f} / "
          f"{np.percentile(hss_scores,50):.3f} / "
          f"{np.percentile(hss_scores,75):.3f}")


if __name__ == "__main__":
    main()
