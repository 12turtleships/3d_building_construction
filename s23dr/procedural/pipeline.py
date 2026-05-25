"""
End-to-end procedural roof reconstruction pipeline (BSP + normal-fit primitives).

reconstruct(xyz, ...)
    → (vertices (V,3), edges list-of-(i,j))

reconstruct_to_segments(xyz, ...)
    → segments (E, 2, 3)   — ready for hss()

Architecture
------------
0. Source + vote_frac filters
1. Footprint extraction from filtered cloud (before spatial clipping)
2. Ground removal + XY-radius filter → roof-face candidate points
3. Surface normal estimation on filtered points
4. BSP rectangle decomposition guided by normal-misfit cost
5. Per-rectangle primitive fitting (flat / gable / hip)
6. Wireframe generation + concatenation
7. Edge pruning: min-length + point-support
"""

from __future__ import annotations

import numpy as np

from .preprocess import extract_roof_points, estimate_normals
from .footprint import extract_footprint
from .decompose import decompose_footprint, mask_in_rect
from .normal_fit import fit_best, wireframe_from_params, _rect_frame


def reconstruct(
    xyz: np.ndarray,                       # (N, 3) normalised
    vote_frac: np.ndarray | None = None,   # (N,) float
    valid_mask: np.ndarray | None = None,  # (N,) bool — kept for API compat
    class_id: np.ndarray | None = None,    # (N,) optional semantic labels
    source: np.ndarray | None = None,      # (N,) uint8
    target_source: int = 1,
    wall_class_ids: set[int] | None = None,
    z_roof_pct: float = 55.0,             # unused; kept for API compat
    regularise_footprint: bool = True,
    min_edge_len: float = 0.02,
    support_radius: float = 0.04,
    min_support: int = 2,
    max_rects: int = 4,
    n_cuts: int = 8,
    # Legacy CSG params (ignored, kept for call-site compatibility)
    max_planes: int = 8,
    plane_eps: float = 0.03,
    min_plane_inliers: int = 8,
    merge_angle_deg: float = 15.0,
    merge_dist: float = 0.05,
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

    # ── Step 1: footprint from source+vote filtered cloud ────────────────────
    # Use the full (unclipped) cloud so eave/corner points are not stripped.
    xyz_full = xyz.copy()
    footprint = extract_footprint(xyz_full, regularise=regularise_footprint,
                                  z_lo_pct=0.0)
    if footprint is None or footprint.is_empty or footprint.area < 1e-6:
        return np.zeros((0, 3), dtype=np.float32), []

    # ── Step 2: ground removal + XY filter ───────────────────────────────────
    xyz_roof = extract_roof_points(xyz)
    if len(xyz_roof) < 5:
        return np.zeros((0, 3), dtype=np.float32), []

    # ── Step 3: surface normals on filtered points ────────────────────────────
    normals = estimate_normals(xyz_roof)

    # ── Step 4: BSP decomposition ─────────────────────────────────────────────
    rects = decompose_footprint(footprint, xyz_roof, normals,
                                max_rects=max_rects, n_cuts=n_cuts)

    # ── Step 5 + 6: primitive fitting → wireframes ────────────────────────────
    all_verts: list[np.ndarray] = []
    all_edges: list[tuple[int, int]] = []

    for corners in rects:
        origin, u_hat, v_hat, width, height = _rect_frame(corners)
        mask = mask_in_rect(xyz_roof, origin, u_hat, v_hat, width, height)
        pts_r  = xyz_roof[mask]
        nrm_r  = normals[mask]

        _, rtype, params = fit_best(pts_r, nrm_r, corners)

        verts_r, edges_r = wireframe_from_params(rtype, params, corners)
        offset = len(all_verts)
        all_verts.extend(verts_r)
        all_edges.extend((i + offset, j + offset) for i, j in edges_r)

    if not all_verts:
        return np.zeros((0, 3), dtype=np.float32), []

    vertices = np.array(all_verts, dtype=np.float32)
    edges = list(set(all_edges))

    # ── Step 7: edge pruning ──────────────────────────────────────────────────
    # Use xyz_full (pre-ground-removal) so eave edges are not pruned.
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
