"""
Multi-plane RANSAC segmentation of roof point clouds.

fit_roof_planes(xyz)
    → list of plane dicts {'n': normal, 'd': offset, 'pts': inlier_xyz}

Algorithm
---------
Iterative RANSAC: fit one plane at a time, remove its inliers, repeat.
Only upward-facing planes (n_z > min_nz) are kept — wall planes are
skipped (inliers removed but plane discarded) so they don't consume
iterations needed for roof faces.

Each plane is parameterised as:  n · x = d
with n a unit normal and n[2] > 0 (pointing upward).
"""

from __future__ import annotations

import numpy as np


def fit_roof_planes(
    xyz: np.ndarray,
    max_planes: int = 8,
    min_inliers: int = 8,
    eps: float = 0.03,       # inlier distance threshold (normalised space)
    n_iter: int = 200,
    min_nz: float = 0.15,    # minimum n_z to count as a roof (not wall) plane
) -> list[dict]:
    """
    Iteratively fit up to *max_planes* planes via RANSAC.

    Parameters
    ----------
    xyz        : (N, 3) point cloud (already source+vote filtered)
    max_planes : stop after this many roof planes
    min_inliers: minimum inliers to accept a plane
    eps        : RANSAC inlier threshold (normalised coords)
    n_iter     : RANSAC iterations per round
    min_nz     : planes with n[2] < min_nz are treated as walls and skipped

    Returns
    -------
    list of {'n': (3,), 'd': float, 'pts': (M,3)}
    """
    rng = np.random.default_rng(42)
    remaining = xyz.copy()
    planes: list[dict] = []

    for _ in range(max_planes + 4):   # +4 budget for discarded wall planes
        if len(remaining) < max(min_inliers, 3):
            break
        if len(planes) >= max_planes:
            break

        # ── RANSAC: find best plane in remaining points ───────────────────────
        best_n, best_d, best_count = None, 0.0, 0
        for _ in range(n_iter):
            idx = rng.choice(len(remaining), 3, replace=False)
            v = remaining[idx]
            n = np.cross(v[1] - v[0], v[2] - v[0])
            nlen = float(np.linalg.norm(n))
            if nlen < 1e-8:
                continue
            n = n / nlen
            if n[2] < 0:
                n = -n
            d = float(n @ v[0])
            count = int((np.abs(remaining @ n - d) < eps).sum())
            if count > best_count:
                best_count = count
                best_n, best_d = n.copy(), d

        if best_n is None or best_count < min_inliers:
            break

        # ── Remove inliers regardless ─────────────────────────────────────────
        inlier_mask = np.abs(remaining @ best_n - best_d) < eps
        inlier_pts  = remaining[inlier_mask]
        remaining   = remaining[~inlier_mask]

        # ── Discard wall / near-vertical planes ───────────────────────────────
        if best_n[2] < min_nz:
            continue

        planes.append({'n': best_n, 'd': best_d, 'pts': inlier_pts})

    return planes
