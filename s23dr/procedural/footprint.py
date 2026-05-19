"""
Extract a 2-D floor-plan footprint from a normalised point cloud.

Strategy
--------
1. Select "wall-band" points — the z band between the 10th and 60th
   percentile (roughly: not ground, not high roof peak).
2. Project to XY and compute the 2-D convex hull.
3. Regularise to a minimum-area rotated rectangle so downstream
   primitive fitting works on a clean outline.

The returned Shapely Polygon is in the same normalised coordinate space
as xyz_norm (roughly [-1, 1]).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon


def extract_footprint(
    xyz: np.ndarray,                          # (N, 3)
    valid_mask: np.ndarray | None = None,     # (N,) bool — dataset "mask" field (unused here)
    vote_frac: np.ndarray | None = None,      # (N,) float — per-point view-agreement score
    vote_thresh: float = 0.3,                 # keep points with vote_frac >= this
    class_id: np.ndarray | None = None,       # (N,) optional semantic labels
    wall_class_ids: set[int] | None = None,   # which IDs count as wall/eave
    z_lo_pct: float = 10.0,                   # lower z percentile for wall band
    z_hi_pct: float = 60.0,                   # upper z percentile for wall band
    simplify_tolerance: float = 0.01,
    regularise: bool = True,
) -> Polygon:
    """Return a 2-D Shapely Polygon representing the building footprint."""

    # --- restrict to high-confidence (voted) points ---------------------------
    # vote_frac > 0 means multiple aerial views agreed this point is a real
    # surface; context / background points typically have vote_frac = 0.
    if vote_frac is not None:
        voted = vote_frac >= vote_thresh
        if voted.sum() >= 4:
            xyz = xyz[voted]
            if class_id is not None:
                class_id = class_id[voted]

    # --- select wall-like points -----------------------------------------------
    if class_id is not None and wall_class_ids:
        mask = np.isin(class_id, list(wall_class_ids))
        pts = xyz[mask]
    else:
        z = xyz[:, 2]
        lo = np.percentile(z, z_lo_pct)
        hi = np.percentile(z, z_hi_pct)
        pts = xyz[(z >= lo) & (z <= hi)]

    if len(pts) < 4:
        pts = xyz  # fallback

    xy = pts[:, :2]

    # --- convex hull -----------------------------------------------------------
    try:
        hull = ConvexHull(xy)
        hull_pts = xy[hull.vertices]
        polygon = Polygon(hull_pts)
    except Exception:
        lo2, hi2 = xy.min(0), xy.max(0)
        polygon = Polygon([
            (lo2[0], lo2[1]), (hi2[0], lo2[1]),
            (hi2[0], hi2[1]), (lo2[0], hi2[1]),
        ])

    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    if simplify_tolerance > 0:
        polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)

    if regularise:
        polygon = polygon.minimum_rotated_rectangle

    return polygon


def rect_corners(rect: Polygon) -> np.ndarray:
    """Return the 4 corners of a rectangle Polygon as (4, 2) float array."""
    coords = np.array(rect.exterior.coords[:-1])
    # Ensure exactly 4 corners (minimum_rotated_rectangle may have 5 coords)
    if len(coords) > 4:
        coords = coords[:4]
    return coords.astype(np.float64)
