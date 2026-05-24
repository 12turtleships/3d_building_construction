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
from shapely.geometry import Point

from .preprocess import extract_roof_points
from .segment import segment_roof
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
    merge_angle_deg: float = 15.0,   # merge faces with normals within this angle
    merge_dist: float = 0.05,        # merge faces whose points lie within this of each other
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

    # ── Step 1: footprint + save full cloud for edge support ─────────────────
    # Footprint uses source+vote filtered cloud before any spatial clipping
    # so eave/corner points are not stripped.
    xyz_full = xyz.copy()   # kept for point-support check in edge pruning
    footprint = extract_footprint(xyz_full, regularise=regularise_footprint,
                                  z_lo_pct=0.0)
    if footprint is None or footprint.is_empty or footprint.area < 1e-6:
        return np.zeros((0, 3), dtype=np.float32), []

    # ── Step 0c: ground removal + XY filter → roof-face candidate points ────
    # extract_roof_points: RANSAC ground removal (capped at 40th z-pct to
    # protect ridge pts) + XY radius clip.  Used only for plane fitting.
    xyz = extract_roof_points(xyz)
    if len(xyz) < 5:
        return np.zeros((0, 3), dtype=np.float32), []

    # ── Step 2: multi-plane RANSAC + adjacent-face merging ───────────────────
    # fit_roof_planes may over-segment (one face → multiple RANSAC hits due to
    # noise); merge_coplanar collapses nearly-parallel planes whose inlier sets
    # are spatially consistent into a single SVD-refitted representative plane.
    planes = segment_roof(
        xyz,
        max_planes=max_planes,
        min_inliers=min_plane_inliers,
        eps=plane_eps,
        angle_thresh_deg=merge_angle_deg,
        dist_thresh=merge_dist,
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
    # Use xyz_full (pre-ground-removal) so eave edges aren't pruned for having
    # no support in the ground-removed cloud.
    edges = _prune_edges(
        vertices, edges, xyz_full,
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
