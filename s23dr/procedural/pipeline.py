"""
End-to-end procedural roof reconstruction pipeline.

reconstruct(xyz, ...)
    → (vertices (V,3), edges list-of-(i,j))

reconstruct_to_segments(xyz, ...)
    → segments (E, 2, 3)   — ready for hss()

The pipeline runs entirely in normalised coordinate space (xyz_norm).

Architecture (CSG lower-envelope)
----------------------------------
0. Source + vote_frac filters
1. Ground removal (RANSAC plane on near-horizontal normals)
2. XY-radius filter (isolate target building)
3. Multi-plane RANSAC → N roof face planes
4. Footprint: XY convex hull of filtered points
5. Lower-envelope CSG: pairwise plane intersections clipped to footprint
   → ridge / hip / valley edges + eave perimeter
6. Edge pruning: min-length + point-support
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPoint, Point

from .preprocess import extract_roof_points
from .segment import fit_roof_planes
from .csg import lower_envelope_wireframe
from .footprint import extract_footprint


def reconstruct(
    xyz: np.ndarray,                       # (N, 3) normalised
    vote_frac: np.ndarray | None = None,   # (N,) float
    valid_mask: np.ndarray | None = None,  # (N,) bool — kept for API compat
    class_id: np.ndarray | None = None,    # (N,) optional semantic labels
    source: np.ndarray | None = None,      # (N,) uint8 — dataset "source" field
    target_source: int = 1,
    wall_class_ids: set[int] | None = None,
    z_roof_pct: float = 55.0,             # unused; kept for API compat
    regularise_footprint: bool = True,
    min_edge_len: float = 0.02,
    support_radius: float = 0.04,
    min_support: int = 2,
    # CSG / segmentation params
    max_planes: int = 8,
    plane_eps: float = 0.03,
    min_plane_inliers: int = 8,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Full pipeline: point cloud → wireframe vertices + edges.

    Returns
    -------
    vertices : (V, 3) float32  in normalised space
    edges    : list of (i, j) index pairs
    """

    # ── Step 0a: source filter ────────────────────────────────────────────────
    if source is not None:
        src_mask = source == target_source
        if src_mask.sum() >= 10:
            xyz = xyz[src_mask]
            if vote_frac is not None:
                vote_frac = vote_frac[src_mask]
            if class_id is not None:
                class_id = class_id[src_mask]

    # ── Step 0b: vote_frac filter ─────────────────────────────────────────────
    if vote_frac is not None:
        voted = vote_frac >= 0.3
        if voted.sum() >= 10:
            xyz = xyz[voted]
            if class_id is not None:
                class_id = class_id[voted]

    # ── Step 0c: ground removal + XY filter ──────────────────────────────────
    # extract_roof_points removes ground (RANSAC) and clips to XY radius.
    # No wall filter — on sparse clouds wall-pitch classification is too noisy.
    xyz = extract_roof_points(xyz)

    if len(xyz) < 5:
        return np.zeros((0, 3), dtype=np.float32), []

    # ── Step 1: footprint (XY convex hull) ───────────────────────────────────
    footprint = _convex_footprint(xyz)
    if footprint is None or footprint.is_empty or footprint.area < 1e-6:
        return np.zeros((0, 3), dtype=np.float32), []

    # ── Step 2: multi-plane RANSAC segmentation ───────────────────────────────
    planes = fit_roof_planes(
        xyz,
        max_planes=max_planes,
        min_inliers=min_plane_inliers,
        eps=plane_eps,
    )

    if not planes:
        # Fallback: treat entire cloud as one flat plane
        z_med = float(np.median(xyz[:, 2]))
        planes = [{'n': np.array([0.0, 0.0, 1.0]), 'd': z_med, 'pts': xyz}]

    # ── Step 3: CSG lower-envelope wireframe ─────────────────────────────────
    vertices, edges = lower_envelope_wireframe(planes, footprint)

    if len(vertices) == 0:
        return np.zeros((0, 3), dtype=np.float32), []

    # ── Step 4: edge pruning ──────────────────────────────────────────────────
    edges = _prune_edges(
        vertices, edges, xyz,
        min_len=min_edge_len,
        support_radius=support_radius,
        min_support=min_support,
    )

    return vertices, edges


def reconstruct_to_segments(
    xyz: np.ndarray,
    vote_frac: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    class_id: np.ndarray | None = None,
    **kwargs,
) -> np.ndarray:
    """Convenience wrapper → (E, 2, 3) segment array for hss()."""
    verts, edges = reconstruct(xyz, vote_frac=vote_frac, valid_mask=valid_mask,
                               class_id=class_id, **kwargs)
    if len(edges) == 0 or len(verts) == 0:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.array([[verts[i], verts[j]] for i, j in edges], dtype=np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convex_footprint(xyz: np.ndarray):
    """Convex hull of the XY projection of xyz."""
    from shapely.geometry import MultiPoint
    pts = MultiPoint(xyz[:, :2].tolist())
    hull = pts.convex_hull
    if hull.geom_type == 'Polygon':
        return hull
    if hull.geom_type == 'LineString':
        return hull.buffer(0.01)   # degenerate: inflate thin line
    return None


def _prune_edges(
    vertices: np.ndarray,
    edges: list[tuple[int, int]],
    roof_pts: np.ndarray,
    min_len: float = 0.02,
    support_radius: float = 0.04,
    min_support: int = 2,
) -> list[tuple[int, int]]:
    """Drop short edges and edges with insufficient point-cloud support."""
    kept: list[tuple[int, int]] = []
    has_pts = len(roof_pts) >= min_support

    for i, j in edges:
        a = vertices[i]
        b = vertices[j]

        if float(np.linalg.norm(b - a)) < min_len:
            continue

        if has_pts:
            ab = b - a
            ab_len_sq = float(ab @ ab)
            if ab_len_sq < 1e-12:
                continue
            t = np.clip(((roof_pts - a) @ ab) / ab_len_sq, 0.0, 1.0)
            closest = a + t[:, None] * ab
            dists = np.linalg.norm(roof_pts - closest, axis=1)
            if int((dists < support_radius).sum()) < min_support:
                continue

        kept.append((i, j))

    return kept
