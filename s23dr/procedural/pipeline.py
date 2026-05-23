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
from .preprocess import extract_roof_points


def reconstruct(
    xyz: np.ndarray,                         # (N, 3) normalised
    vote_frac: np.ndarray | None = None,     # (N,) float — dataset "vote_frac" field
    valid_mask: np.ndarray | None = None,    # (N,) bool — kept for API compat
    class_id: np.ndarray | None = None,      # (N,) optional semantic labels
    source: np.ndarray | None = None,        # (N,) uint8 — dataset "source" field
    target_source: int = 1,                  # value in source[] that marks target building
    wall_class_ids: set[int] | None = None,  # which IDs = wall/eave
    z_roof_pct: float = 55.0,               # unused after preprocessing; kept for API compat
    regularise_footprint: bool = True,
    min_edge_len: float = 0.02,             # drop edges shorter than this (normalised)
    support_radius: float = 0.04,           # tube radius for point-support check
    min_support: int = 3,                   # min roof points inside tube to keep edge
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Full pipeline: point cloud → wireframe vertices + edges.

    Returns
    -------
    vertices : (V, 3) float32  in normalised space
    edges    : list of (i, j) index pairs
    """

    # ── Step 0a: source filter (target building isolation) ─────────────────────
    # The dataset "source" field is a binary label. If source==target_source
    # marks the target building's own points, filtering to those gives a clean
    # per-building point cloud without neighbouring buildings or ground context.
    if source is not None:
        src_mask = source == target_source
        if src_mask.sum() >= 10:
            xyz = xyz[src_mask]
            if vote_frac is not None:
                vote_frac = vote_frac[src_mask]
            if class_id is not None:
                class_id = class_id[src_mask]

    # ── Step 0b: vote_frac filter ──────────────────────────────────────────────────
    vote_thresh = 0.3
    if vote_frac is not None:
        voted = vote_frac >= vote_thresh
        if voted.sum() >= 10:
            xyz = xyz[voted]
            if class_id is not None:
                class_id = class_id[voted]

    # ── Step 0c: surface normal segmentation ───────────────────────────────────
    xyz = extract_roof_points(xyz)

    # ── Step 1: floor plan footprint from roof-surface points ─────────────────
    # xyz is now XY-filtered (target building only) — use all points for hull.
    footprint = extract_footprint(
        xyz,
        class_id=class_id if wall_class_ids else None,
        wall_class_ids=wall_class_ids,
        regularise=regularise_footprint,
        z_lo_pct=0.0,
    )

    # ── Step 2: decompose into rectangular sections ───────────────────────────
    sections = decompose_footprint(footprint)

    # ── Steps 3 & 4: fit primitive per section, collect wireframe ──────────────
    all_verts: list[np.ndarray] = []
    all_edges: list[tuple[int, int]] = []
    v_offset = 0

    for section in sections:
        # All preprocessed points inside this section are roof-surface points
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

    # ── Step 5: edge pruning ───────────────────────────────────────────────────
    all_edges = _prune_edges(
        vertices, all_edges, xyz,
        min_len=min_edge_len,
        support_radius=support_radius,
        min_support=min_support,
    )

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


def _prune_edges(
    vertices: np.ndarray,
    edges: list[tuple[int, int]],
    roof_pts: np.ndarray,
    min_len: float = 0.02,
    support_radius: float = 0.04,
    min_support: int = 3,
) -> list[tuple[int, int]]:
    """
    Remove low-quality edges by two criteria applied in order:

    1. Minimum length: edges shorter than *min_len* are phantom artifacts
       from near-duplicate vertices (tiny section footprints, shared corners).

    2. Point support: for each edge [A, B] compute the distance from every
       preprocessed roof point to the segment.  If fewer than *min_support*
       points fall within *support_radius* of the segment, the edge has no
       point-cloud evidence and is dropped.  This removes template-geometry
       lines that don't correspond to any real architectural edge.
    """
    kept: list[tuple[int, int]] = []
    has_pts = len(roof_pts) >= min_support

    for i, j in edges:
        a = vertices[i]
        b = vertices[j]

        # ── filter 1: minimum length ──────────────────────────────────────
        seg_len = float(np.linalg.norm(b - a))
        if seg_len < min_len:
            continue

        # ── filter 2: point support ───────────────────────────────────────
        if has_pts:
            ab = b - a
            ab_len_sq = float(ab @ ab)
            # project each point onto the segment parameter t in [0, 1]
            t = np.clip(((roof_pts - a) @ ab) / ab_len_sq, 0.0, 1.0)  # (N,)
            closest = a + t[:, None] * ab                               # (N, 3)
            dists = np.linalg.norm(roof_pts - closest, axis=1)          # (N,)
            if int((dists < support_radius).sum()) < min_support:
                continue

        kept.append((i, j))

    return kept
