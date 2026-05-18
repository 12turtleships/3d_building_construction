"""
Decompose a 2-D footprint polygon into rectangular sections.

Simple buildings (nearly convex) → one bounding rectangle.
Complex buildings (L / T / U shapes) → 2–4 rectangles via iterative
convex peeling.

Each returned rectangle is a Shapely Polygon in the same normalised space
as the input footprint.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon


def decompose_footprint(
    footprint: Polygon,
    max_rects: int = 4,
    min_area_fraction: float = 0.05,
    convexity_threshold: float = 1.15,
) -> list[Polygon]:
    """
    Decompose *footprint* into up to *max_rects* rectangular sections.

    Parameters
    ----------
    convexity_threshold
        If convex_hull.area / footprint.area < this value the footprint is
        treated as convex and returned as a single rectangle.
    """
    if footprint.is_empty or footprint.area < 1e-8:
        return [footprint]

    hull_ratio = footprint.convex_hull.area / max(footprint.area, 1e-10)
    if hull_ratio < convexity_threshold:
        return [footprint.minimum_rotated_rectangle]

    rects = _convex_peel(footprint, max_rects)

    min_area = footprint.area * min_area_fraction
    rects = [r for r in rects if r.area >= min_area]

    return rects if rects else [footprint.minimum_rotated_rectangle]


def _convex_peel(polygon: Polygon, max_parts: int) -> list[Polygon]:
    """
    Iteratively peel convex pieces.

    Each iteration takes the convex hull of the remaining polygon, stores
    its minimum bounding rectangle, then subtracts the hull to get the
    non-convex remainder for the next iteration.
    """
    remaining = polygon
    parts: list[Polygon] = []

    for _ in range(max_parts):
        if remaining.is_empty or remaining.area < 1e-8:
            break
        hull = remaining.convex_hull
        parts.append(hull.minimum_rotated_rectangle)
        diff = remaining.difference(hull)
        if diff.is_empty:
            break
        # Continue with the largest remaining fragment
        if hasattr(diff, "geoms"):
            remaining = max(diff.geoms, key=lambda g: g.area)
        else:
            remaining = diff

    return parts
