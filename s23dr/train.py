#!/usr/bin/env python3
"""
Training script for RoofWireframeNet on S23DR 2026.

Usage
-----
    # Start fresh
    python s23dr/train.py

    # Resume — tries last.pt first, then best.pt
    python s23dr/train.py --resume

    # Debug smoke test
    python s23dr/train.py --epochs 5 --lr 1e-3 --batch-size 2 --split-debug
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from s23dr.data import S23DRDataset, collate_fn
from s23dr.model import RoofWireframeNet, WireframeLoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch-size", type=int,   default=4)
    p.add_argument("--n-queries",  type=int,   default=64)
    p.add_argument("--n-points",   type=int,   default=1024,
                   help="Points per sample (random subsample; 4096=full)")
    p.add_argument("--n-workers",  type=int,   default=0)
    p.add_argument("--device",     type=str,   default="cpu")
    p.add_argument("--ckpt-dir",   type=str,   default="outputs/checkpoints")
    p.add_argument("--log-every",  type=int,   default=10)
    p.add_argument("--split-debug", action="store_true",
                   help="Use only first 8 train samples for fast smoke test")
    p.add_argument("--resume",     action="store_true",
                   help="Resume from last.pt (or best.pt if last.pt missing)")
    return p.parse_args()


def _torch_load(path, map_location):
    """torch.load compatible with both old (<2.0) and new (>=2.0) PyTorch."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _save(path: Path, epoch: int, model, optimizer, scheduler,
          val_loss: float, args):
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "val_loss":  val_loss,
        "args":      vars(args),
    }, path)


def _find_resume_ckpt(ckpt_dir: Path) -> tuple[Path | None, str]:
    """Return (path, label) for the best available resume checkpoint."""
    for fname in ("last.pt", "best.pt"):
        p = ckpt_dir / fname
        if p.exists():
            return p, fname
    return None, ""


def _apply_resume(ckpt_path: Path, label: str, device,
                  model, optimizer, scheduler, total_epochs: int):
    """
    Load checkpoint into model/optimizer/scheduler.
    Handles old-format checkpoints that lack optimizer/scheduler state
    by fast-forwarding the scheduler to the saved epoch.
    Returns (start_epoch, best_val).
    """
    ckpt = _torch_load(ckpt_path, device)
    model.load_state_dict(ckpt["model"])

    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    # else: keep freshly initialised optimizer (learning rate will be off
    # until scheduler corrects it, so fast-forward below)

    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        # Old checkpoint without scheduler state — fast-forward to match epoch
        # so cosine-annealing LR is correct from this point.
        for _ in range(ckpt["epoch"]):
            scheduler.step()

    start_epoch = ckpt["epoch"] + 1
    best_val    = ckpt.get("val_loss", float("inf"))
    print(f"Resumed from {label}  epoch={ckpt['epoch']}  "
          f"val_loss={best_val:.4f}  → continuing from epoch {start_epoch}")
    return start_epoch, best_val


def train_one_epoch(model, loader, loss_fn, optimizer, device, log_every):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for step, batch in enumerate(loader):
        xyz       = batch["xyz"].to(device)
        vote_frac = batch["vote_frac"].to(device)
        n_views   = batch["n_views"].to(device)
        mask      = batch["mask"].to(device)
        class_id  = batch["class_id"].to(device)

        out = model(xyz, vote_frac, n_views, mask, class_id)
        losses = loss_fn(
            out["pred_pos"], out["pred_conf"], out["edge_logits"],
            batch["gt_verts"], batch["gt_edges"], batch["gt_classes"],
        )

        optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += losses["loss"].item()
        if (step + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(f"  step {step+1:4d}  "
                  f"loss={losses['loss'].item():.4f}  "
                  f"vert={losses['loss_vert'].item():.4f}  "
                  f"conf={losses['loss_conf'].item():.4f}  "
                  f"edge={losses['loss_edge'].item():.4f}  "
                  f"({elapsed:.1f}s)")
            t0 = time.time()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        out = model(
            batch["xyz"].to(device), batch["vote_frac"].to(device),
            batch["n_views"].to(device), batch["mask"].to(device),
            batch["class_id"].to(device),
        )
        losses = loss_fn(
            out["pred_pos"], out["pred_conf"], out["edge_logits"],
            batch["gt_verts"], batch["gt_edges"], batch["gt_classes"],
        )
        total_loss += losses["loss"].item()
    return total_loss / max(len(loader), 1)


def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    last_ckpt = ckpt_dir / "last.pt"
    best_ckpt = ckpt_dir / "best.pt"

    print("Loading datasets…")
    train_ds = S23DRDataset(split="train",      n_points=args.n_points)
    val_ds   = S23DRDataset(split="validation", n_points=args.n_points)

    if args.split_debug:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(min(8, len(train_ds)))))
        val_ds   = Subset(val_ds,   list(range(min(4, len(val_ds)))))
        print(f"  Debug mode: {len(train_ds)} train, {len(val_ds)} val samples")
    else:
        print(f"  {len(train_ds)} train, {len(val_ds)} val samples")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.n_workers, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.n_workers, collate_fn=collate_fn,
    )

    model     = RoofWireframeNet(n_queries=args.n_queries).to(device)
    loss_fn   = WireframeLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_val    = float("inf")

    if args.resume:
        ckpt_path, label = _find_resume_ckpt(ckpt_dir)
        if ckpt_path:
            start_epoch, best_val = _apply_resume(
                ckpt_path, label, device, model, optimizer, scheduler, args.epochs)
            # If resumed from best.pt, best_val is already correct.
            # If resumed from last.pt and best.pt also exists, use best.pt's val_loss.
            if label == "last.pt" and best_ckpt.exists():
                best_val = _torch_load(best_ckpt, "cpu").get("val_loss", best_val)
        else:
            print(f"No checkpoint found in {ckpt_dir} — starting fresh.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters  device={device}")

    for epoch in range(start_epoch, args.epochs + 1):
        t_start = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn,
                                     optimizer, device, args.log_every)
        val_loss   = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        elapsed = time.time() - t_start
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  ({elapsed:.1f}s)")

        _save(last_ckpt, epoch, model, optimizer, scheduler, val_loss, args)

        if val_loss < best_val:
            best_val = val_loss
            _save(best_ckpt, epoch, model, optimizer, scheduler, val_loss, args)
            print(f"  ✓ best checkpoint saved (val={val_loss:.4f})")


if __name__ == "__main__":
    main()
