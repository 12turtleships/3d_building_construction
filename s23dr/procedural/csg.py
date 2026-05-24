"""
CSG lower-envelope wireframe from fitted roof planes.

lower_envelope_wireframe(planes, footprint)
    → (vertices (V,3), edges list[(i,j)])

Theory
------
Each roof plane i is parameterised as  n_i · x = d_i  (n_iz > 0).
At any XY position (x, y) the height of plane i is:

    z_i(x, y) = (d_i  -  n_ix·x  -  n_iy·y) / n_iz

The physical roof surface is the *lower envelope*:

    z(x, y) = min_i  z_i(x, y)

The wireframe edges are exactly the boundaries in the XY footprint where
two planes tie for the minimum (ridge, hip, valley lines).

For planes i and j, their equal-height locus is a straight line in XY:

    (n_jx/n_jz - n_ix/n_iz)·x + (n_jy/n_jz - n_iy/n_iz)·y
        = d_j/n_jz - d_i/n_iz

A segment of this line is *active* (part of the lower envelope) only if
no other plane k gives a strictly lower z along it.  We test this by
sampling the midpoint of each clipped candidate segment.

Footprint perimeter edges (eaves) are also included.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lower_envelope_wireframe(
    planes: list[dict],
    footprint,                   # Shapely Polygon
    tol: float = 1e-4,           # vertex merge tolerance
    z_eps: float = 5e-3,         # tolerance for "another plane is lower"
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Compute the 3-D wireframe of the CSG lower envelope.

    Parameters
    ----------
    planes    : list of {'n': (3,), 'd': float} dicts from segment.fit_roof_planes
    footprint : Shapely Polygon — XY outline of the roof
    tol       : 2-D distance below which two vertices are merged
    z_eps     : a candidate ridge is discarded if any other plane is more
                than z_eps lower at its midpoint (it's inside another face)

    Returns
    -------
    vertices  : (V, 3) float32
    edges     : list of (i, j) index pairs into vertices
    """
    if not planes or footprint is None or footprint.is_empty:
        return np.zeros((0, 3), dtype=np.float32), []

    # ── 1. Candidate ridge segments: pairwise plane intersections ─────────────
    raw_segs: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    # (plane_i, plane_j, p0_2d, p1_2d)

    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            seg = _plane_pair_segment(planes[i], planes[j], footprint)
            if seg is not None:
                raw_segs.append((i, j, seg[0], seg[1]))

    # ── 2. Keep only active segments (on the lower envelope) ──────────────────
    active_segs: list[tuple[np.ndarray, np.ndarray]] = []
    for i, j, p0, p1 in raw_segs:
        mid = (p0 + p1) * 0.5
        z_i = _z_at(planes[i], mid)
        z_j = _z_at(planes[j], mid)
        z_boundary = min(z_i, z_j)
        dominated = False
        for k, pk in enumerate(planes):
            if k == i or k == j:
                continue
            if _z_at(pk, mid) < z_boundary - z_eps:
                dominated = True
                break
        if not dominated:
            active_segs.append((p0, p1))

    # ── 3. Collect all 2-D vertices ───────────────────────────────────────────
    pts_2d: list[np.ndarray] = []

    # Footprint perimeter vertices
    fp_ring = list(footprint.exterior.coords)[:-1]
    fp_n = len(fp_ring)
    for xy in fp_ring:
        pts_2d.append(np.array(xy, dtype=float))

    # Ridge endpoint vertices
    ridge_edge_indices: list[tuple[int, int]] = []
    for p0, p1 in active_segs:
        i0 = len(pts_2d);  pts_2d.append(p0)
        i1 = len(pts_2d);  pts_2d.append(p1)
        ridge_edge_indices.append((i0, i1))

    # ── 4. Merge nearby 2-D vertices ──────────────────────────────────────────
    unique_2d: list[np.ndarray] = []
    idx_map: list[int] = []
    for v in pts_2d:
        found = -1
        for k, u in enumerate(unique_2d):
            if np.linalg.norm(v - u) < tol:
                found = k
                break
        if found < 0:
            found = len(unique_2d)
            unique_2d.append(v)
        idx_map.append(found)

    # ── 5. Lift to 3-D ────────────────────────────────────────────────────────
    vertices_3d: list[np.ndarray] = []
    for v2d in unique_2d:
        z = _lower_z(planes, v2d)
        vertices_3d.append(np.array([v2d[0], v2d[1], z], dtype=np.float32))
    vertices = np.array(vertices_3d, dtype=np.float32)

    # ── 6. Build edge list ────────────────────────────────────────────────────
    edges: list[tuple[int, int]] = []

    # Footprint perimeter (eave) edges
    for k in range(fp_n):
        a = idx_map[k]
        b = idx_map[(k + 1) % fp_n]
        if a != b:
            edges.append((min(a, b), max(a, b)))

    # Ridge / hip / valley edges
    for (raw_i0, raw_i1) in ridge_edge_indices:
        a = idx_map[raw_i0]
        b = idx_map[raw_i1]
        if a != b:
            edges.append((min(a, b), max(a, b)))

    edges = list(set(edges))
    return vertices, edges


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _z_at(plane: dict, xy: np.ndarray) -> float:
    """Height of *plane* at the 2-D position *xy*."""
    n, d = plane['n'], plane['d']
    if abs(n[2]) < 1e-8:
        return 1e9
    return float((d - n[0] * xy[0] - n[1] * xy[1]) / n[2])


def _lower_z(planes: list[dict], xy: np.ndarray) -> float:
    """Lower-envelope z at *xy*: minimum over all planes."""
    return min(_z_at(p, xy) for p in planes)


def _plane_pair_segment(
    pi: dict, pj: dict, footprint
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Compute the 2-D line segment where plane *pi* and *pj* have equal height,
    clipped to *footprint*.  Returns (p0, p1) or None if the intersection
    doesn't cross the footprint.
    """
    ni, di = pi['n'], pi['d']
    nj, dj = pj['n'], pj['d']

    if abs(ni[2]) < 1e-8 or abs(nj[2]) < 1e-8:
        return None

    # Equal-height locus: a·x + b·y = c
    a = nj[0] / nj[2] - ni[0] / ni[2]
    b = nj[1] / nj[2] - ni[1] / ni[2]
    c = dj   / nj[2] - di   / ni[2]

    if abs(a) < 1e-9 and abs(b) < 1e-9:
        return None   # parallel planes at same height

    # A point on the line + direction
    if abs(a) >= abs(b):
        # Parameterise by y
        p0_xy = np.array([c / a, 0.0])
        direction = np.array([-b, a])
    else:
        p0_xy = np.array([0.0, c / b])
        direction = np.array([-b, a])
    direction = direction / np.linalg.norm(direction)

    # Long line clipped to footprint bounding box + buffer
    t = 5.0
    line = LineString([p0_xy - t * direction, p0_xy + t * direction])
    clipped = line.intersection(footprint)

    if clipped.is_empty:
        return None

    # Collect individual LineString pieces
    if clipped.geom_type == 'LineString':
        pieces = [clipped]
    elif clipped.geom_type in ('MultiLineString', 'GeometryCollection'):
        pieces = [g for g in clipped.geoms if g.geom_type == 'LineString']
    else:
        return None

    # Return the longest piece
    best_len, best_seg = 0.0, None
    for piece in pieces:
        coords = list(piece.coords)
        if len(coords) < 2:
            continue
        p0 = np.array(coords[0],  dtype=float)
        p1 = np.array(coords[-1], dtype=float)
        seg_len = float(np.linalg.norm(p1 - p0))
        if seg_len > best_len:
            best_len, best_seg = seg_len, (p0, p1)

    return best_seg
