#!/usr/bin/env python3
"""
Generate a submission.json from a trained RoofWireframeNet checkpoint.

Output format (one JSON object per line):
  {"order_id": "...", "wf_vertices": [[x,y,z],...], "wf_edges": [[i,j],...]}

Vertices are denormalised to world space:
  world_xyz = norm_xyz * scale + center

Usage
-----
    python s23dr/scripts/make_submission.py \\
        --ckpt outputs/checkpoints/best.pt \\
        --out  outputs/submission.json

    # Quick smoke test on 10 samples:
    python s23dr/scripts/make_submission.py \\
        --ckpt outputs/checkpoints/best.pt \\
        --out  /tmp/test_sub.json --max-samples 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from s23dr.data  import S23DRDataset, collate_fn
from s23dr.model import RoofWireframeNet, WireframeLoss


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt",        required=True, help="Path to best.pt checkpoint")
    p.add_argument("--out",         default="outputs/submission.json")
    p.add_argument("--split",       default="validation",
                   choices=("train", "validation"),
                   help="Split to generate predictions for")
    p.add_argument("--n-points",    type=int,   default=4096,
                   help="Points per sample (use 4096=full for submission)")
    p.add_argument("--batch-size",  type=int,   default=4)
    p.add_argument("--n-workers",   type=int,   default=0)
    p.add_argument("--device",      default="cpu")
    p.add_argument("--conf-thresh", type=float, default=0.5)
    p.add_argument("--max-samples", type=int,   default=None,
                   help="Limit to first N samples (for testing)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    n_queries = ckpt.get("args", {}).get("n_queries", 64)

    model = RoofWireframeNet(n_queries=n_queries).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch')}  "
          f"val_loss={ckpt.get('val_loss', '?'):.4f}")

    loss_fn = WireframeLoss()

    print(f"Loading {args.split} split (n_points={args.n_points})…")
    ds = S23DRDataset(split=args.split, n_points=args.n_points)
    if args.max_samples:
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(min(args.max_samples, len(ds)))))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.n_workers, collate_fn=collate_fn)

    records = []
    n_processed = 0

    with torch.no_grad():
        for batch in loader:
            xyz = batch["xyz"].to(device)
            out = model(
                xyz,
                batch["vote_frac"].to(device),
                batch["n_views"].to(device),
                batch["mask"].to(device),
                batch["class_id"].to(device),
            )
            B = xyz.shape[0]
            for b in range(B):
                verts_norm, edges, _ = loss_fn.decode(
                    out["pred_pos"][b],
                    out["pred_conf"][b],
                    out["edge_logits"][b],
                    conf_thresh=args.conf_thresh,
                )

                # Denormalise to world space: world = norm * scale + center
                scale  = float(batch["scale"][b])
                center = batch["center"][b].to(verts_norm.device)   # (3,)
                verts_world = verts_norm * scale + center

                # Fallback: return a trivial 2-vertex, 1-edge wireframe
                if len(verts_world) < 2 or len(edges) == 0:
                    verts_world = center.unsqueeze(0).repeat(2, 1)
                    verts_world[1, 0] += 1.0   # shift x by 1 unit
                    edges = torch.tensor([[0, 1]], dtype=torch.long)

                records.append({
                    "order_id":   batch["order_id"][b],
                    "wf_vertices": verts_world.cpu().tolist(),
                    "wf_edges":    edges.cpu().tolist(),
                })
                n_processed += 1
                if n_processed % 100 == 0:
                    print(f"  {n_processed} samples…")

    with open(out_path, "w") as f:
        json.dump(records, f)

    print(f"\nWrote {len(records)} predictions → {out_path}")
    # Quick sanity check
    r = records[0]
    print(f"Sample[0]: order_id={r['order_id']}  "
          f"n_verts={len(r['wf_vertices'])}  n_edges={len(r['wf_edges'])}")


if __name__ == "__main__":
    main()
