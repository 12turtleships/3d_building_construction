"""
Point cloud preprocessing for roof wireframe extraction.

extract_roof_points(xyz)
    → roof-candidate points after removing ground and wall/facade points.

Pipeline
--------
1. Estimate per-point surface normals via k-NN PCA (vectorised SVD).
2. Remove walls / facades — near-vertical normals (pitch > wall_pitch_thresh).
3. Remove ground — RANSAC-fit one dominant plane to the near-horizontal
   point subset, then remove all near-horizontal points within eps of that
   plane. Using a fitted plane (rather than a flat z-threshold) handles
   sloped terrain correctly.
4. Restrict to the target building's XY neighbourhood (scene is centred on
   the target building).
"""

from __future__ import annotations

import numpy as np


def estimate_normals(xyz: np.ndarray, k: int = 12) -> np.ndarray:
    """
    Per-point surface normals via k-NN PCA (fully vectorised).
    Returns (N, 3) unit normals with z >= 0 (pointing upward).
    """
    from scipy.spatial import cKDTree

    k = min(k, len(xyz))
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k)          # (N, k)
    neighbors = xyz[idx]                    # (N, k, 3)
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)  # Vt: (N, 3, 3)
    normals = Vt[:, -1, :]                 # smallest singular vector → normal
    normals[normals[:, 2] < 0] *= -1      # ensure upward orientation
    return normals


def _ransac_ground_plane(pts: np.ndarray,
                         n_iter: int = 100,
                         eps: float = 0.04):
    """
    RANSAC fit of a single dominant plane to *pts*.
    Returns a (unit_normal, d) tuple or None if fitting fails.
    eps is the inlier distance threshold in normalised space.
    """
    if len(pts) < 3:
        return None
    rng = np.random.default_rng(0)
    best_n = 0
    best = None
    for _ in range(n_iter):
        idx = rng.choice(len(pts), 3, replace=False)
        v = pts[idx]
        n = np.cross(v[1] - v[0], v[2] - v[0])
        nlen = float(np.linalg.norm(n))
        if nlen < 1e-8:
            continue
        n = n / nlen
        if n[2] < 0:
            n = -n
        d = float(n @ v[0])
        count = int((np.abs(pts @ n - d) < eps).sum())
        if count > best_n:
            best_n = count
            best = (n, d)
    return best


def extract_roof_points(
    xyz: np.ndarray,
    k_normal: int = 12,
    wall_pitch_thresh: float = 55.0,   # normals with pitch > this → wall → remove
    ground_pitch_thresh: float = 20.0, # near-horizontal normals → ground candidates
    ground_eps: float = 0.04,          # RANSAC inlier distance for ground plane
    xy_radius: float = 0.38,           # keep only points within this XY dist from origin
) -> np.ndarray:
    """
    Return roof-candidate points by removing ground and wall/facade points,
    then restricting to the target building's XY neighbourhood.

    Steps
    -----
    1. Estimate surface normals.
    2. Remove walls  (pitch > wall_pitch_thresh ≈ 55°).
    3. Detect ground plane via RANSAC on near-horizontal points, then remove
       near-horizontal inliers. Using a fitted plane handles sloped terrain.
    4. Keep only points within xy_radius of the scene origin.
    """
    if len(xyz) < 20:
        return xyz

    # ── 1. Surface normals ────────────────────────────────────────────────────
    normals = estimate_normals(xyz, k=k_normal)
    pitch_deg = np.degrees(np.arccos(np.clip(np.abs(normals[:, 2]), 0.0, 1.0)))

    # ── 2. Remove walls ───────────────────────────────────────────────────────
    is_wall = pitch_deg > wall_pitch_thresh

    # ── 3. RANSAC ground plane ────────────────────────────────────────────────
    # Fit one dominant plane to the near-horizontal point subset. Handles
    # sloped terrain: a tilted ground surface has consistent near-horizontal
    # normals that RANSAC resolves into one plane regardless of z level.
    horizontal = pitch_deg < ground_pitch_thresh
    h_pts = xyz[horizontal]

    is_ground = np.zeros(len(xyz), dtype=bool)
    if len(h_pts) >= 15:
        result = _ransac_ground_plane(h_pts, eps=ground_eps)
        if result is not None:
            gn, gd = result
            dists = np.abs(xyz @ gn - gd)
            # Remove only points that are BOTH near the plane AND near-horizontal
            # — this avoids stripping pitched eave points at the same elevation.
            is_ground = (dists < ground_eps) & horizontal
    else:
        # Fallback: conservative z-percentile
        z_thresh = np.percentile(xyz[:, 2], 30)
        is_ground = (xyz[:, 2] < z_thresh) & horizontal

    roof = xyz[~is_wall & ~is_ground]

    if len(roof) < 10:
        roof = xyz[~is_wall]
    if len(roof) < 10:
        return xyz

    # ── 4. XY origin filter — isolate target building ─────────────────────────
    xy_dist = np.sqrt(roof[:, 0] ** 2 + roof[:, 1] ** 2)
    nearby = roof[xy_dist <= xy_radius]
    return nearby if len(nearby) >= 10 else roof
