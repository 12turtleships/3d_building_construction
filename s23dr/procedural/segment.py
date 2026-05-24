"""
Multi-plane RANSAC segmentation + adjacent-face merging for roof point clouds.

Public API
----------
fit_roof_planes(xyz)   → raw list of plane dicts
merge_coplanar(planes) → merged list (splits of the same face collapsed)
segment_roof(xyz)      → fit + merge in one call

Each plane dict: {'n': unit_normal (3,), 'd': float offset, 'pts': (M,3)}
Plane equation:  n · x = d,   n[2] > 0  (upward facing).

Merge algorithm
---------------
Two planes are considered the same roof face if:
  1. Their normals are nearly parallel  (angle < angle_thresh °)
  2. Their inlier point sets lie close to each other's plane
     (mean residual < dist_thresh in normalised space)

After grouping by connected components (union-find), each group is
re-fitted via SVD on all combined inlier points → one clean plane per face.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# RANSAC plane fitting
# ---------------------------------------------------------------------------

def fit_roof_planes(
    xyz: np.ndarray,
    max_planes: int = 8,
    min_inliers: int = 8,
    eps: float = 0.03,
    n_iter: int = 200,
    min_nz: float = 0.15,
) -> list[dict]:
    """
    Iteratively fit up to *max_planes* upward-facing planes via RANSAC.
    Wall planes (n_z < min_nz) consume the iteration budget but are discarded.
    """
    rng = np.random.default_rng(42)
    remaining = xyz.copy()
    planes: list[dict] = []

    for _ in range(max_planes + 4):
        if len(remaining) < max(min_inliers, 3):
            break
        if len(planes) >= max_planes:
            break

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

        inlier_mask = np.abs(remaining @ best_n - best_d) < eps
        inlier_pts  = remaining[inlier_mask]
        remaining   = remaining[~inlier_mask]

        if best_n[2] < min_nz:
            continue

        planes.append({'n': best_n, 'd': best_d, 'pts': inlier_pts})

    return planes


# ---------------------------------------------------------------------------
# Adjacent-plane merging
# ---------------------------------------------------------------------------

def merge_coplanar(
    planes: list[dict],
    angle_thresh_deg: float = 15.0,
    dist_thresh: float = 0.05,
) -> list[dict]:
    """
    Merge RANSAC planes that belong to the same physical roof face.

    Two planes are merged if:
      1. Their normals subtend an angle < angle_thresh_deg (nearly parallel).
      2. The inlier points of each plane lie close to the other's plane
         (mean residual < dist_thresh).  This prevents merging two parallel
         but height-separated faces (e.g. a stepped roof).

    Each merged group is re-fitted via SVD on all combined inliers so the
    representative plane is optimal for the full face, not just one RANSAC hit.
    """
    n = len(planes)
    if n <= 1:
        return planes

    cos_thresh = np.cos(np.radians(angle_thresh_deg))

    # ── Union-Find ────────────────────────────────────────────────────────────
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            ni, di = planes[i]['n'], planes[i]['d']
            nj, dj = planes[j]['n'], planes[j]['d']

            # 1. Normal angle check (normals are both upward → dot ∈ [-1, 1])
            dot = float(np.dot(ni, nj))
            if dot < cos_thresh:
                continue

            # 2. Mutual plane-distance check
            pts_i, pts_j = planes[i]['pts'], planes[j]['pts']
            # Mean distance of i's points from j's plane, and vice versa
            dist_ij = float(np.mean(np.abs(pts_i @ nj - dj)))
            dist_ji = float(np.mean(np.abs(pts_j @ ni - di)))
            if max(dist_ij, dist_ji) > dist_thresh:
                continue

            union(i, j)

    # ── Group by connected component ──────────────────────────────────────────
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # ── Refit each group via SVD ───────────────────────────────────────────────
    merged: list[dict] = []
    for indices in groups.values():
        all_pts = np.vstack([planes[i]['pts'] for i in indices])
        n_fit, d_fit = _svd_plane(all_pts)
        merged.append({'n': n_fit, 'd': d_fit, 'pts': all_pts})

    return merged


# ---------------------------------------------------------------------------
# Convenience: fit + merge in one call
# ---------------------------------------------------------------------------

def segment_roof(
    xyz: np.ndarray,
    max_planes: int = 8,
    min_inliers: int = 8,
    eps: float = 0.03,
    n_iter: int = 200,
    min_nz: float = 0.15,
    angle_thresh_deg: float = 15.0,
    dist_thresh: float = 0.05,
) -> list[dict]:
    """Fit roof planes then merge adjacent co-planar fragments."""
    planes = fit_roof_planes(xyz, max_planes=max_planes,
                             min_inliers=min_inliers, eps=eps,
                             n_iter=n_iter, min_nz=min_nz)
    return merge_coplanar(planes, angle_thresh_deg=angle_thresh_deg,
                          dist_thresh=dist_thresh)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _svd_plane(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Best-fit plane to *pts* via SVD. Returns (unit_normal, d)."""
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
    n = Vt[-1]                  # smallest singular vector
    if n[2] < 0:
        n = -n
    d = float(n @ centroid)
    return n, d
