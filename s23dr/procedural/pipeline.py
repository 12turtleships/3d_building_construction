"""
End-to-end procedural roof reconstruction pipeline.

reconstruct(xyz, class_id)
    → (vertices (V,3), edges list-of-(i,j))

reconstruct_to_segments(xyz, class_id)
    → segments (E, 2, 3)   — ready for hss()

The pipeline runs entirely in normalised coordinate space (xyz_norm).
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Point

from .footprint import extract_footprint
from .decompose import decompose_footprint
from .primitives import fit_best_primitive


def reconstruct(
    xyz: np.ndarray,                         # (N, 3) normalised
    vote_frac: np.ndarray | None = None,     # (N,) float — dataset "vote_frac" field
    valid_mask: np.ndarray | None = None,    # (N,) bool — kept for API compat
    class_id: np.ndarray | None = None,      # (N,) optional semantic labels
    wall_class_ids: set[int] | None = None,  # which IDs = wall/eave
    z_roof_pct: float = 35.0,               # z-percentile threshold for roof pts
    regularise_footprint: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Full pipeline: point cloud → wireframe vertices + edges.

    Returns
    -------
    vertices : (V, 3) float32  in normalised space
    edges    : list of (i, j) index pairs
    """

    # ── Step 0: restrict to high-confidence points for all processing ────────
    # vote_frac >= 0.3 isolates the target building; background/context points
    # have near-zero vote_frac and corrupt z_thresh and plane fitting if kept.
    vote_thresh = 0.3
    if vote_frac is not None:
        voted = vote_frac >= vote_thresh
        if voted.sum() >= 10:
            xyz = xyz[voted]
            if class_id is not None:
                class_id = class_id[voted]
            vote_frac = None   # already filtered; no need to re-filter in extract_footprint

    # ── Step 1: floor plan footprint ─────────────────────────────────────────
    footprint = extract_footprint(
        xyz,
        vote_frac=vote_frac,
        class_id=class_id,
        wall_class_ids=wall_class_ids,
        regularise=regularise_footprint,
    )

    # ── Step 2: decompose into rectangular sections ───────────────────────────
    sections = decompose_footprint(footprint)

    # ── Steps 3 & 4: fit primitive per section, collect wireframe ─────────────
    all_verts: list[np.ndarray] = []
    all_edges: list[tuple[int, int]] = []
    v_offset = 0

    z_thresh = float(np.percentile(xyz[:, 2], z_roof_pct))

    for section in sections:
        # Points inside this section footprint above the roof threshold
        roof_mask = _points_in_polygon(xyz[:, :2], section) & (xyz[:, 2] >= z_thresh)
        roof_pts = xyz[roof_mask]

        if len(roof_pts) < 10:
            # Fall back: all points inside section
            in_mask = _points_in_polygon(xyz[:, :2], section)
            roof_pts = xyz[in_mask]

        if len(roof_pts) < 5:
            roof_pts = xyz   # last resort: use all

        prim = fit_best_primitive(roof_pts, section)
        verts, edges = prim.wireframe()

        all_verts.append(verts)
        all_edges.extend((i + v_offset, j + v_offset) for i, j in edges)
        v_offset += len(verts)

    if not all_verts:
        return np.zeros((0, 3), dtype=np.float32), []

    vertices = np.vstack(all_verts).astype(np.float32)
    return vertices, all_edges


def reconstruct_to_segments(
    xyz: np.ndarray,
    vote_frac: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    class_id: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Convenience wrapper returning (E, 2, 3) segment array for hss().
    """
    verts, edges = reconstruct(xyz, vote_frac=vote_frac, valid_mask=valid_mask,
                               class_id=class_id, **kwargs)
    if len(edges) == 0 or len(verts) == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)
    segs = np.array([[verts[i], verts[j]] for i, j in edges], dtype=np.float32)
    return segs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _points_in_polygon(xy: np.ndarray, polygon) -> np.ndarray:
    """
    Return boolean mask: which rows of xy fall inside *polygon*.
    Uses a fast bounding-box pre-filter then exact Shapely test.
    """
    minx, miny, maxx, maxy = polygon.bounds
    rough = (
        (xy[:, 0] >= minx) & (xy[:, 0] <= maxx) &
        (xy[:, 1] >= miny) & (xy[:, 1] <= maxy)
    )
    mask = rough.copy()
    for i in np.where(rough)[0]:
        mask[i] = polygon.contains(Point(float(xy[i, 0]), float(xy[i, 1])))
    return mask
