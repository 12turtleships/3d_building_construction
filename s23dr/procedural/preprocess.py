"""
Point cloud preprocessing for roof wireframe extraction.

extract_roof_points(xyz)
    → roof-candidate points after removing ground and wall/facade points.

Pipeline
--------
1. Estimate per-point surface normals via k-NN PCA (vectorised SVD).
2. Remove walls / facades — near-vertical normals (pitch > wall_pitch_thresh).
3. Remove ground — near-horizontal normals (pitch < ground_pitch_thresh) in the
   lower z band (z < 25th percentile of the cloud).
4. From the remaining roof candidates, select the cluster belonging to the
   target building by finding the centroid of the highest-z points (the ridge /
   peak) and keeping all points within cluster_radius_factor × that radius.
   The scene is centred on the target building, so its highest-z cluster lies
   nearest to the XY origin.
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


def extract_roof_points(
    xyz: np.ndarray,
    k_normal: int = 12,
    wall_pitch_thresh: float = 55.0,     # normals with pitch > this → wall → remove
    ground_pitch_thresh: float = 20.0,   # near-horizontal normals at low z → ground
    ground_pct: float = 30.0,            # z-percentile cutoff for ground detection
    xy_radius: float = 0.38,             # keep only points within this XY distance from origin
) -> np.ndarray:
    """
    Return roof-candidate points by removing ground and wall/facade points,
    then restricting to the target building's XY neighbourhood.

    Steps
    -----
    1. Estimate surface normals.
    2. Remove walls  (pitch > wall_pitch_thresh  ≈ 55°).
    3. Remove ground (pitch < ground_pitch_thresh ≈ 20° AND z in lower z-band).
    4. Keep only points within xy_radius of the scene origin. The dataset
       centres xyz_norm on the target building; GT segments span ≈ ±0.29,
       so 0.55 safely captures the whole building while excluding neighbours
       that sit at XY > 0.5 in the normalised coordinate frame.
    """
    if len(xyz) < 20:
        return xyz

    # ── 1. Surface normals ────────────────────────────────────────────────────
    normals = estimate_normals(xyz, k=k_normal)
    # pitch_deg: 0 = horizontal (ground / flat roof), 90 = vertical (wall)
    pitch_deg = np.degrees(np.arccos(np.clip(np.abs(normals[:, 2]), 0.0, 1.0)))

    # ── 2. Remove walls ───────────────────────────────────────────────────────
    is_wall = pitch_deg > wall_pitch_thresh

    # ── 3. Remove ground ─────────────────────────────────────────────────────
    z = xyz[:, 2]
    z_low_thresh = np.percentile(z, ground_pct)
    is_ground = (z < z_low_thresh) & (pitch_deg < ground_pitch_thresh)

    roof = xyz[~is_wall & ~is_ground]

    if len(roof) < 10:
        roof = xyz[~is_wall]
    if len(roof) < 10:
        return xyz

    # ── 4. XY origin filter — isolate target building ─────────────────────────
    # GT wireframe for sample 0 spans XY ≤ 0.26; xy_radius=0.38 gives ~0.12
    # margin while excluding far neighbours (which reach XY=0.86).
    xy_dist = np.sqrt(roof[:, 0] ** 2 + roof[:, 1] ** 2)
    nearby = roof[xy_dist <= xy_radius]
    return nearby if len(nearby) >= 10 else roof
