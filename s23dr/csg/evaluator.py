"""
CSG wireframe evaluator.

evaluate_csg(node) → (vertices (V,3), edges (E,2), edge_classes (E,))

Algorithm
---------
For each node in the CSG tree we maintain a list of EdgeSeg (3D line segments
with a class label).  At a leaf the segments come from the primitive wireframe.

At a binary node (Union / Intersection / Difference) we:

1. Recursively evaluate left → segs_L  and right → segs_R.
2. Split every segment at the leaf-face planes of the *other* subtree, then
   midpoint-test each piece with the other subtree's CSGNode.inside() method.
   This handles both convex and non-convex subtrees correctly.
     Union        : keep parts of L outside R,  keep parts of R outside L
     Intersection : keep parts of L inside  R,  keep parts of R inside  L
     Difference   : keep parts of L outside R,  keep parts of R inside  L
3. Add *seam segments* — new edges born where a leaf-face of L crosses a
   leaf-face of R, with the midpoint inside both subtrees.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .tree import CSGNode
from .primitives import HalfSpace, FaceInfo, VALLEY, HIP, RIDGE, EAVE, GABLE, MISC


_MIN_SEG_LEN = 1e-5   # discard degenerate segments


# ---------------------------------------------------------------------------
# Internal segment type
# ---------------------------------------------------------------------------

@dataclass
class EdgeSeg:
    p0: np.ndarray   # (3,) float
    p1: np.ndarray   # (3,) float
    cls: int

    def length(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def _collect_leaf_faces(node: CSGNode) -> list[FaceInfo]:
    """All FaceInfo objects from leaf primitives in the subtree."""
    if node.op == "leaf":
        return list(node.primitive.faces())
    return _collect_leaf_faces(node.left) + _collect_leaf_faces(node.right)


def _split_at_planes(seg: EdgeSeg, planes: list[HalfSpace]) -> list[EdgeSeg]:
    """
    Split segment at each plane boundary.  Returns sub-segments, each
    guaranteed to lie on one side of every plane.
    """
    d = seg.p1 - seg.p0
    t_vals = {0.0, 1.0}
    for hs in planes:
        denom = float(hs.normal @ d)
        if abs(denom) < 1e-12:
            continue
        t = -(float(hs.normal @ seg.p0) + hs.offset) / denom
        if 1e-6 < t < 1.0 - 1e-6:
            t_vals.add(t)
    t_sorted = sorted(t_vals)
    result = []
    for i in range(len(t_sorted) - 1):
        t0, t1 = t_sorted[i], t_sorted[i + 1]
        p0 = seg.p0 + t0 * d
        p1 = seg.p0 + t1 * d
        s = EdgeSeg(p0, p1, seg.cls)
        if s.length() > _MIN_SEG_LEN:
            result.append(s)
    return result


def _filter_segs(segs: list[EdgeSeg],
                 node: CSGNode,
                 keep_inside: bool) -> list[EdgeSeg]:
    """
    Split each segment at node's leaf-face planes, then keep sub-segments
    whose midpoint is inside (keep_inside=True) or outside (=False) node.
    """
    planes = [f.plane for f in _collect_leaf_faces(node)]
    result = []
    for seg in segs:
        for ss in _split_at_planes(seg, planes):
            mid = (ss.p0 + ss.p1) * 0.5
            is_in = bool(node.inside(mid[np.newaxis])[0])
            if is_in == keep_inside:
                result.append(ss)
    return result


# ---------------------------------------------------------------------------
# Plane–plane intersection line
# ---------------------------------------------------------------------------

def _plane_plane_line(hs_a: HalfSpace,
                      hs_b: HalfSpace
                      ) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (point_on_line, unit_direction) of the two planes' intersection."""
    na, nb = hs_a.normal, hs_b.normal
    direction = np.cross(na, nb)
    if np.linalg.norm(direction) < 1e-10:
        return None
    direction = direction / np.linalg.norm(direction)
    A = np.array([na, nb, direction])
    b = np.array([-hs_a.offset, -hs_b.offset, 0.0])
    try:
        point = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return point, direction


def _clip_line_to_face(point: np.ndarray, direction: np.ndarray,
                       face: FaceInfo,
                       tol: float = 1e-6) -> tuple[float, float] | None:
    """
    Clip the parametric line  p(t) = point + t*direction  to the face polygon
    using the face's boundary edge half-spaces.
    Returns (t_min, t_max) or None.
    """
    verts = face.vertices
    if len(verts) < 3:
        return None

    face_n = face.plane.normal
    t_min, t_max = -1e9, 1e9
    n = len(verts)

    for i in range(n):
        v0 = verts[i]
        v1 = verts[(i + 1) % n]
        edge_dir = v1 - v0
        inward = np.cross(face_n, edge_dir)
        norm = np.linalg.norm(inward)
        if norm < 1e-10:
            continue
        inward /= norm
        denom = float(inward @ direction)
        numer = float(inward @ v0) - float(inward @ point)
        if abs(denom) < 1e-12:
            if numer < -tol:
                return None
            continue
        t = numer / denom
        if denom > 0:
            t_min = max(t_min, t)
        else:
            t_max = min(t_max, t)

    if t_min >= t_max - tol or t_max < -1e8 or t_min > 1e8:
        return None
    return t_min, t_max


def _intersection_class(cls_a: int, cls_b: int, op: str) -> int:
    """Edge class for the seam created by a CSG operation."""
    if op == "union":
        return HIP if HIP in (cls_a, cls_b) else VALLEY
    if op == "intersection":
        return HIP if HIP in (cls_a, cls_b) else RIDGE
    return MISC   # difference cut surface


def _seam_segs(node_a: CSGNode, node_b: CSGNode, op: str) -> list[EdgeSeg]:
    """
    Compute new wireframe edges at the seam where the two subtrees meet.
    For each pair of leaf faces (one from each subtree), intersect their planes
    and keep the segment if the midpoint is inside both subtrees.
    """
    faces_a = _collect_leaf_faces(node_a)
    faces_b = _collect_leaf_faces(node_b)
    segs = []

    for fa, fb in itertools.product(faces_a, faces_b):
        result = _plane_plane_line(fa.plane, fb.plane)
        if result is None:
            continue
        pt, dirn = result

        ta = _clip_line_to_face(pt, dirn, fa)
        if ta is None:
            continue
        tb = _clip_line_to_face(pt, dirn, fb)
        if tb is None:
            continue

        t0 = max(ta[0], tb[0])
        t1 = min(ta[1], tb[1])
        if t1 - t0 < _MIN_SEG_LEN:
            continue

        p0 = pt + t0 * dirn
        p1 = pt + t1 * dirn
        mid = (p0 + p1) * 0.5

        in_a = bool(node_a.inside(mid[np.newaxis])[0])
        in_b = bool(node_b.inside(mid[np.newaxis])[0])
        if not (in_a and in_b):
            continue

        cls = _intersection_class(fa.edge_class, fb.edge_class, op)
        segs.append(EdgeSeg(p0, p1, cls))

    return segs


# ---------------------------------------------------------------------------
# Recursive evaluator
# ---------------------------------------------------------------------------

def _eval(node: CSGNode) -> list[EdgeSeg]:
    if node.op == "leaf":
        prim = node.primitive
        verts, edges, classes = prim.wireframe()
        segs = []
        for k, (i, j) in enumerate(edges):
            s = EdgeSeg(verts[i].copy(), verts[j].copy(), int(classes[k]))
            if s.length() > _MIN_SEG_LEN:
                segs.append(s)
        return segs

    segs_l = _eval(node.left)
    segs_r = _eval(node.right)
    op = node.op

    if op == "union":
        kept_l = _filter_segs(segs_l, node.right, keep_inside=False)
        kept_r = _filter_segs(segs_r, node.left,  keep_inside=False)
    elif op == "intersection":
        kept_l = _filter_segs(segs_l, node.right, keep_inside=True)
        kept_r = _filter_segs(segs_r, node.left,  keep_inside=True)
    elif op == "difference":
        kept_l = _filter_segs(segs_l, node.right, keep_inside=False)
        kept_r = _filter_segs(segs_r, node.left,  keep_inside=True)
    else:
        raise ValueError(f"Unknown op: {op}")

    return kept_l + kept_r + _seam_segs(node.left, node.right, op)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_csg(node: CSGNode,
                 merge_tol: float = 1e-4
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a CSG tree and return its wireframe.

    Returns
    -------
    vertices     : (V, 3) float32
    edges        : (E, 2) int32   — indices into vertices
    edge_classes : (E,)   int64
    """
    segs = _eval(node)
    if not segs:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 2), dtype=np.int32),
                np.zeros((0,),   dtype=np.int64))

    raw_verts = np.array([p for seg in segs for p in (seg.p0, seg.p1)],
                         dtype=np.float64)

    vert_list: list[np.ndarray] = []
    vert_idx = np.full(len(raw_verts), -1, dtype=np.int32)

    for i, v in enumerate(raw_verts):
        if vert_idx[i] >= 0:
            continue
        idx = len(vert_list)
        vert_list.append(v)
        dists = np.linalg.norm(raw_verts[i:] - v, axis=1)
        close = np.where(dists < merge_tol)[0] + i
        vert_idx[close] = idx

    vertices = np.array(vert_list, dtype=np.float32)

    edge_list, cls_list, seen = [], [], set()
    for k, seg in enumerate(segs):
        i0, i1 = int(vert_idx[2 * k]), int(vert_idx[2 * k + 1])
        if i0 == i1:
            continue
        key = (min(i0, i1), max(i0, i1))
        if key in seen:
            continue
        seen.add(key)
        edge_list.append([i0, i1])
        cls_list.append(seg.cls)

    if not edge_list:
        return (vertices,
                np.zeros((0, 2), dtype=np.int32),
                np.zeros((0,),   dtype=np.int64))

    return (vertices,
            np.array(edge_list, dtype=np.int32),
            np.array(cls_list,  dtype=np.int64))
