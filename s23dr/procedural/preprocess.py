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
    ground_pct: float = 60.0,            # z-percentile cutoff for ground detection
    peak_pct: float = 5.0,               # top z-percentile used for cluster centroid
    cluster_radius_factor: float = 3.0,  # keep points within this × building_radius
) -> np.ndarray:
    """
    Return roof-candidate points by removing ground and wall/facade points,
    then isolating the target building cluster.

    Steps
    -----
    1. Estimate surface normals.
    2. Remove walls  (pitch > wall_pitch_thresh  ≈ 55°).
    3. Remove ground (pitch < ground_pitch_thresh ≈ 20° AND z in lower z-band).
    4. Estimate the target building's spatial extent from its highest-z points
       (roof ridge / peak), then keep only points within cluster_radius_factor
       times that radius of the ridge centroid.
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
    # Use the 60th z-percentile as the ground cutoff: in a typical scene with
    # ~40-60% ground/pavement points below the building, this threshold sits
    # just above the ground layer, so any near-horizontal point below it is
    # classified as ground.
    z = xyz[:, 2]
    z_low_thresh = np.percentile(z, ground_pct)
    is_ground = (z < z_low_thresh) & (pitch_deg < ground_pitch_thresh)

    roof = xyz[~is_wall & ~is_ground]

    if len(roof) < 10:
        roof = xyz[~is_wall]   # fallback: remove walls only
    if len(roof) < 10:
        return xyz

    # ── 4. Isolate target building by its highest-z cluster ───────────────────
    # The dataset centres xyz_norm on the target building. The target building's
    # peak (ridge / highest flat section) has the highest z values and lies
    # nearest to the XY origin. Use those peak points to estimate the building's
    # XY extent, then keep all roof points within that radius.
    z_roof = roof[:, 2]
    # Use only the very top z points (top 5%) to estimate the peak cluster.
    # These extreme-high-z points belong to the TALLEST structure, which in
    # the S23DR dataset is always the target building (the scene is centred on it).
    n_top = max(10, int(peak_pct / 100.0 * len(roof)))
    top_idx = np.argsort(z_roof)[-n_top:]
    top_pts = roof[top_idx]

    centroid_xy = top_pts[:, :2].mean(axis=0)   # XY centre of the peak cluster
    # RMS distance of peak points from their centroid = characteristic radius
    xy_dev = top_pts[:, :2] - centroid_xy
    radius = float(np.sqrt(np.mean(xy_dev[:, 0] ** 2 + xy_dev[:, 1] ** 2)))
    radius = max(radius, 0.10)   # floor: at least 10 cm in normalised space

    xy_dist = np.sqrt(
        (roof[:, 0] - centroid_xy[0]) ** 2 +
        (roof[:, 1] - centroid_xy[1]) ** 2
    )
    nearby = roof[xy_dist <= cluster_radius_factor * radius]

    return nearby if len(nearby) >= 10 else roof
