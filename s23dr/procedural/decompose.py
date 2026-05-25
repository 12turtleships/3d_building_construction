"""
BSP rectangle decomposition guided by surface normal fitting cost.

decompose_footprint(footprint, xyz, normals, max_rects=4)
    → list of (4, 2) corner arrays in world XY

Greedily splits the footprint's minimum bounding rectangle into up to
max_rects sub-rectangles, accepting each split only when it strictly
reduces the weighted normal-misfit cost (see normal_fit.fit_best).

Algorithm
---------
1. Start: one rectangle = MBR of footprint.
2. For each current rectangle, try n_cuts cut positions along both axes.
3. Accept the globally best split whose cost improvement exceeds improve_thresh.
4. Repeat until max_rects reached or no beneficial split found.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from .normal_fit import fit_best, _rect_frame


def decompose_footprint(
    footprint: Polygon,
    xyz: np.ndarray,            # (N, 3) normalised points
    normals: np.ndarray,        # (N, 3) unit surface normals
    max_rects: int = 4,
    n_cuts: int = 8,            # candidate cut positions per axis
    min_pts: int = 4,           # minimum points required in each child
    improve_thresh: float = 0.005,  # minimum cost drop to accept a split
) -> list[np.ndarray]:          # list of (4, 2) corner arrays
    """
    Decompose footprint into ≤ max_rects rectangles via greedy BSP.
    Returns a list of (4, 2) world-XY corner arrays.
    """
    if footprint.is_empty or len(xyz) < min_pts:
        return [_polygon_corners(footprint.minimum_rotated_rectangle)]

    rects = [_polygon_corners(footprint.minimum_rotated_rectangle)]

    for _ in range(max_rects - 1):
        best_gain   = improve_thresh
        best_result = None   # (rect_idx, corners_a, corners_b)

        for ri, corners in enumerate(rects):
            origin, u_hat, v_hat, width, height = _rect_frame(corners)
            mask = mask_in_rect(xyz, origin, u_hat, v_hat, width, height)
            if mask.sum() < min_pts:
                continue
            pts, nrm = xyz[mask], normals[mask]
            cost_before, _, _ = fit_best(pts, nrm, corners)

            for axis in (0, 1):   # 0 = cut along long axis, 1 = cut along short axis
                for k in range(1, n_cuts + 1):
                    frac = k / (n_cuts + 1)
                    ca, cb = _split_rect(origin, u_hat, v_hat, width, height, axis, frac)
                    o_a, u_a, v_a, w_a, h_a = _rect_frame(ca)
                    o_b, u_b, v_b, w_b, h_b = _rect_frame(cb)
                    m_a = mask_in_rect(xyz, o_a, u_a, v_a, w_a, h_a)
                    m_b = mask_in_rect(xyz, o_b, u_b, v_b, w_b, h_b)
                    if m_a.sum() < min_pts or m_b.sum() < min_pts:
                        continue
                    c_a, _, _ = fit_best(xyz[m_a], normals[m_a], ca)
                    c_b, _, _ = fit_best(xyz[m_b], normals[m_b], cb)
                    wt = float(m_a.sum()) / float(m_a.sum() + m_b.sum())
                    cost_after = wt * c_a + (1 - wt) * c_b
                    gain = cost_before - cost_after
                    if gain > best_gain:
                        best_gain   = gain
                        best_result = (ri, ca, cb)

        if best_result is None:
            break

        ri, ca, cb = best_result
        rects.pop(ri)
        rects.extend([ca, cb])

    return rects


# ---------------------------------------------------------------------------
# Helpers (also imported by pipeline)
# ---------------------------------------------------------------------------

def mask_in_rect(
    xyz: np.ndarray,
    origin: np.ndarray,
    u_hat: np.ndarray,
    v_hat: np.ndarray,
    width: float,
    height: float,
    margin: float = 1e-4,
) -> np.ndarray:
    """Boolean mask: which xyz[:, :2] points fall inside the rectangle."""
    delta = xyz[:, :2] - origin
    u_loc = delta @ u_hat
    v_loc = delta @ v_hat
    return (np.abs(u_loc) <= height / 2 + margin) & (np.abs(v_loc) <= width / 2 + margin)


def _split_rect(
    origin: np.ndarray,
    u_hat: np.ndarray,
    v_hat: np.ndarray,
    width: float,
    height: float,
    axis: int,
    frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split rectangle at *frac* along *axis* (0=long/u, 1=short/v)."""
    if axis == 0:   # cut perpendicular to u (partitions the long span)
        h_a = frac * height
        h_b = (1.0 - frac) * height
        ctr_a = origin + (-height / 2.0 + h_a / 2.0) * u_hat
        ctr_b = origin + (-height / 2.0 + h_a + h_b / 2.0) * u_hat
        return (_make_corners(ctr_a, u_hat, v_hat, h_a, width),
                _make_corners(ctr_b, u_hat, v_hat, h_b, width))
    else:           # cut perpendicular to v (partitions the short span)
        w_a = frac * width
        w_b = (1.0 - frac) * width
        ctr_a = origin + (-width / 2.0 + w_a / 2.0) * v_hat
        ctr_b = origin + (-width / 2.0 + w_a + w_b / 2.0) * v_hat
        return (_make_corners(ctr_a, u_hat, v_hat, height, w_a),
                _make_corners(ctr_b, u_hat, v_hat, height, w_b))


def _make_corners(
    center: np.ndarray,
    u_hat: np.ndarray,
    v_hat: np.ndarray,
    height: float,
    width: float,
) -> np.ndarray:
    """Create (4, 2) corners from rectangle frame (center, axes, dimensions)."""
    hu = (height / 2.0) * u_hat
    hv = (width  / 2.0) * v_hat
    return np.array([
        center - hu - hv,
        center + hu - hv,
        center + hu + hv,
        center - hu + hv,
    ], dtype=np.float64)


def _polygon_corners(poly: Polygon) -> np.ndarray:
    """Extract first 4 XY corners from a rectangular Shapely polygon."""
    coords = list(poly.exterior.coords)[:-1]
    return np.array(coords[:4], dtype=np.float64)
