"""
HSS — Hausdorff Segment Score.

Implementation follows the definition in arXiv:2503.08208:
  "Explaining Human Preferences via Metrics for Structured 3D Reconstruction"

A wireframe is represented as a set of line segments {(p0, p1)}.
The score measures how well predicted segments cover GT segments and vice versa.

Algorithm
---------
Given predicted segments P and GT segments G:

1. For each segment in P, compute directed coverage to G (min Hausdorff distance
   over uniformly sampled points along the segment).
2. For each segment in G, compute directed coverage to P.
3. precision = fraction of P-segments with coverage distance < tau
4. recall    = fraction of G-segments with coverage distance < tau
5. HSS       = 2 * precision * recall / (precision + recall)  [F1-style]

Default tau = 0.2 (in normalised coordinate space where scene ≈ unit cube).
"""

from __future__ import annotations

import numpy as np
import torch


# ------------------------------------------------------------------
def _point_to_seg_dist(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point in pts (N,3) to segment [a, b]."""
    ab = b - a                           # (3,)
    len2 = float(np.dot(ab, ab))
    if len2 < 1e-12:
        return np.linalg.norm(pts - a, axis=-1)
    t = np.clip(((pts - a) @ ab) / len2, 0.0, 1.0)   # (N,)
    closest = a + t[:, None] * ab                     # (N, 3)
    return np.linalg.norm(pts - closest, axis=-1)     # (N,)


def _directed_seg_to_seg_dist(
    seg: np.ndarray,         # (2, 3) — query segment
    others: np.ndarray,      # (M, 2, 3) — reference segments
    n_samples: int = 8,
) -> float:
    """
    Directed distance from `seg` to the nearest segment in `others`.
    Sample n_samples points uniformly along `seg`, find minimum distance
    from each to any segment in `others`, return the max (Hausdorff).
    """
    a, b = seg[0], seg[1]
    ts = np.linspace(0.0, 1.0, n_samples)
    pts = a + ts[:, None] * (b - a)          # (n_samples, 3)

    min_dists = np.full(n_samples, np.inf)
    for ref in others:
        d = _point_to_seg_dist(pts, ref[0], ref[1])
        min_dists = np.minimum(min_dists, d)

    return float(min_dists.max())


def hss(
    pred_segs: np.ndarray | None,   # (P, 2, 3) or None
    gt_segs:   np.ndarray,          # (G, 2, 3)
    tau: float = 0.2,
    n_samples: int = 8,
) -> dict[str, float]:
    """
    Compute HSS between predicted and GT segments.

    Returns dict with keys: hss, precision, recall.
    Returns zeros if either set is empty.
    """
    if pred_segs is None or len(pred_segs) == 0 or len(gt_segs) == 0:
        return {"hss": 0.0, "precision": 0.0, "recall": 0.0}

    pred_segs = np.asarray(pred_segs, dtype=np.float32)
    gt_segs   = np.asarray(gt_segs,   dtype=np.float32)

    # precision: fraction of predicted segments covered by GT
    prec_covered = sum(
        _directed_seg_to_seg_dist(s, gt_segs, n_samples) < tau
        for s in pred_segs
    )
    precision = prec_covered / len(pred_segs)

    # recall: fraction of GT segments covered by predictions
    rec_covered = sum(
        _directed_seg_to_seg_dist(s, pred_segs, n_samples) < tau
        for s in gt_segs
    )
    recall = rec_covered / len(gt_segs)

    denom = precision + recall
    hss_score = (2.0 * precision * recall / denom) if denom > 0 else 0.0

    return {"hss": hss_score, "precision": precision, "recall": recall}


# ------------------------------------------------------------------
def decode_to_segments(
    vertices: torch.Tensor,   # (V, 3)
    edges:    torch.Tensor,   # (E, 2)
) -> np.ndarray:
    """Convert (vertices, edges) wireframe to segment array (E, 2, 3)."""
    if len(edges) == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)
    v = vertices.cpu().numpy()
    e = edges.cpu().numpy()
    return np.stack([v[e[:, 0]], v[e[:, 1]]], axis=1).astype(np.float32)
