"""
Roof CSG primitives.

Each primitive is a convex solid described by:
  - half_spaces(): list of (normal, offset) in world coords,
      where the solid interior satisfies  normal @ x + offset <= 0  for all planes.
  - wireframe(): (vertices (V,3), edges (E,2), edge_classes (E,))
      — the boundary edge graph in world coordinates.

Edge class vocabulary (mirrors S23DR gt_edge_classes convention):
  0 = ridge   1 = hip   2 = valley   3 = eave   4 = gable   5 = misc
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Edge class constants
# ---------------------------------------------------------------------------
RIDGE  = 0
HIP    = 1
VALLEY = 2
EAVE   = 3
GABLE  = 4
MISC   = 5


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


class HalfSpace(NamedTuple):
    """Outward half-space: normal @ x + offset <= 0  →  point is inside solid."""
    normal: np.ndarray   # unit (3,) outward normal
    offset: float        # scalar

    def signed_dist(self, pts: np.ndarray) -> np.ndarray:
        """(N,3) → (N,) signed distance; negative = inside."""
        return pts @ self.normal + self.offset

    def inside(self, pts: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        return self.signed_dist(pts) <= tol


class FaceInfo(NamedTuple):
    """One planar face of a primitive."""
    plane: HalfSpace          # outward half-space defining this face's plane
    vertices: np.ndarray      # (M,3) convex polygon vertices in world coords
    edge_class: int           # class of edges *on* this face (used for intersection edges)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Primitive:
    def half_spaces(self) -> list[HalfSpace]:
        raise NotImplementedError

    def inside(self, pts: np.ndarray, tol: float = 1e-6) -> np.ndarray:
        """(N,3) bool — True if point is inside the solid."""
        pts = np.atleast_2d(pts)
        result = np.ones(len(pts), dtype=bool)
        for hs in self.half_spaces():
            result &= hs.inside(pts, tol)
        return result

    def faces(self) -> list[FaceInfo]:
        raise NotImplementedError

    def wireframe(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (vertices (V,3), edges (E,2 int), edge_classes (E,))."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# P1 — RidgePrism
#
# Ridge runs along the local x-axis.  In local frame:
#   - Ridge line:      y=0,  z=H,  x ∈ [-L/2, L/2]
#   - Left eave:       y=W_l, z=0,  x ∈ [-L/2, L/2]
#   - Right eave:      y=-W_r, z=0, x ∈ [-L/2, L/2]
#   - Left end gable:  x=-L/2 (triangle)
#   - Right end gable: x=+L/2 (triangle)
# ---------------------------------------------------------------------------

class RidgePrism(Primitive):
    """
    Symmetric or asymmetric gable wedge.

    Parameters
    ----------
    center  : (3,) eave-floor midpoint in world coords (ridge is at center + H*z)
    theta   : ridge azimuth (radians, CCW from +x)
    L       : ridge length
    W_l     : left eave half-width  (local +y side)
    W_r     : right eave half-width (local -y side)
    H       : ridge height above eave plane
    """

    def __init__(self, center, theta: float, L: float,
                 W_l: float, W_r: float, H: float) -> None:
        self.center = np.asarray(center, dtype=float)
        self.theta = float(theta)
        self.L = float(L)
        self.W_l = float(W_l)
        self.W_r = float(W_r)
        self.H = float(H)
        self.R = _rot_z(theta)          # local→world rotation
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        L, Wl, Wr, H = self.L, self.W_l, self.W_r, self.H
        R, c = self.R, self.center

        def to_world(p_local):
            return np.asarray(p_local, dtype=float) @ R.T + c

        # ---- vertices (local) ----------------------------------------
        v = np.array([
            [-L/2,   0,  H],   # 0  ridge left
            [ L/2,   0,  H],   # 1  ridge right
            [-L/2,  Wl,  0],   # 2  left eave, left end
            [ L/2,  Wl,  0],   # 3  left eave, right end
            [-L/2, -Wr,  0],   # 4  right eave, left end
            [ L/2, -Wr,  0],   # 5  right eave, right end
        ], dtype=float)
        self._verts_local = v
        self._verts_world = np.array([to_world(p) for p in v])

        # ---- half-spaces (local, then rotated to world) ---------------
        def hs_local(n_local, p_on_plane_local):
            n_w = R @ n_local
            n_w /= np.linalg.norm(n_w)
            d = -float(n_w @ to_world(p_on_plane_local))
            return HalfSpace(n_w, d)

        # Left slope — outward normal points up-left in local frame
        n_ls = np.array([0.0,  H, Wl]); n_ls /= np.linalg.norm(n_ls)
        # Right slope — outward normal points up-right
        n_rs = np.array([0.0, -H, Wr]); n_rs /= np.linalg.norm(n_rs)
        # Floor — outward normal points down
        n_fl = np.array([0.0, 0.0, -1.0])
        # Left end (x=-L/2) — outward normal points in -x
        n_le = np.array([-1.0, 0.0, 0.0])
        # Right end (x=+L/2) — outward normal points in +x
        n_re = np.array([ 1.0, 0.0, 0.0])

        self._hs = [
            hs_local(n_ls, v[0]),   # 0 left slope
            hs_local(n_rs, v[0]),   # 1 right slope
            hs_local(n_fl, v[2]),   # 2 floor
            hs_local(n_le, v[0]),   # 3 left end
            hs_local(n_re, v[1]),   # 4 right end
        ]
        self._hs_classes = [GABLE, GABLE, EAVE, GABLE, GABLE]

        # ---- edges: (i, j, class) ------------------------------------
        vw = self._verts_world
        self._edges = [
            (0, 1, RIDGE),   # ridge
            (2, 3, EAVE),    # left eave
            (4, 5, EAVE),    # right eave
            (0, 2, GABLE),   # left gable, left slope edge
            (2, 4, EAVE),    # left gable, bottom
            (4, 0, GABLE),   # left gable, right slope edge
            (1, 3, GABLE),   # right gable, left slope edge
            (3, 5, EAVE),    # right gable, bottom
            (5, 1, GABLE),   # right gable, right slope edge
        ]

        # ---- faces: each face as polygon + dominant edge class --------
        self._faces = [
            FaceInfo(self._hs[0], vw[[0,1,3,2]], GABLE),   # left slope quad
            FaceInfo(self._hs[1], vw[[0,4,5,1]], GABLE),   # right slope quad
            FaceInfo(self._hs[2], vw[[2,3,5,4]], EAVE),    # floor
            FaceInfo(self._hs[3], vw[[0,2,4]],   GABLE),   # left end gable
            FaceInfo(self._hs[4], vw[[1,5,3]],   GABLE),   # right end gable
        ]

    # ------------------------------------------------------------------
    def half_spaces(self):
        return self._hs

    def faces(self):
        return self._faces

    def wireframe(self):
        vw = self._verts_world
        edges = np.array([(i, j) for i, j, _ in self._edges], dtype=np.int32)
        classes = np.array([c for _, _, c in self._edges], dtype=np.int64)
        return vw.copy(), edges, classes


# ---------------------------------------------------------------------------
# P2 — HipEnd
#
# Pyramidal cap for hip roof termination.  Used with Intersection to replace
# gable triangles of a RidgePrism with four hip slopes.
#
# In local frame:
#   - Apex: (0, 0, H)
#   - Base corners: (±D/2, ±W/2, 0)  — rectangular footprint
# ---------------------------------------------------------------------------

class HipEnd(Primitive):
    """
    Four-sided pyramid (hip end).

    Parameters
    ----------
    center  : (3,) apex base center (midpoint of the rectangular footprint)
    theta   : azimuth (radians)
    W       : footprint width (local y extent)
    D       : footprint depth (local x extent, along ridge direction)
    H       : apex height
    """

    def __init__(self, center, theta: float, W: float, D: float, H: float) -> None:
        self.center = np.asarray(center, dtype=float)
        self.theta = float(theta)
        self.W = float(W)
        self.D = float(D)
        self.H = float(H)
        self.R = _rot_z(theta)
        self._build()

    def _build(self):
        W, D, H = self.W, self.D, self.H
        R, c = self.R, self.center

        def to_world(p):
            return np.asarray(p, dtype=float) @ R.T + c

        # Vertices
        apex   = np.array([0.0, 0.0, H])
        bl = np.array([-D/2, -W/2, 0.0])   # back-left
        br = np.array([ D/2, -W/2, 0.0])   # back-right (front of ridge)
        fl = np.array([-D/2,  W/2, 0.0])   # far-left
        fr = np.array([ D/2,  W/2, 0.0])   # far-right

        verts_local = np.array([apex, bl, br, fl, fr])
        self._verts_world = np.array([to_world(p) for p in verts_local])

        def plane_hs(pts_on_face, outward_ref):
            """Outward half-space from three face points + interior reference."""
            a, b, cc = np.asarray(pts_on_face[0], dtype=float), \
                       np.asarray(pts_on_face[1], dtype=float), \
                       np.asarray(pts_on_face[2], dtype=float)
            n = np.cross(b - a, cc - a)
            n = n / np.linalg.norm(n)
            # Flip if points toward interior reference
            ref_w = to_world(outward_ref)
            if np.dot(n, ref_w - to_world(a)) > 0:
                n = -n
            d = -float(n @ to_world(a))
            return HalfSpace(n, d)

        interior_local = np.array([0.0, 0.0, H / 4])  # below apex, inside

        # 5 half-spaces: 4 sloped faces + floor
        self._hs = [
            plane_hs([apex, bl, fl], interior_local),   # back slope  (-x face)
            plane_hs([apex, fr, br], interior_local),   # front slope (+x face)
            plane_hs([apex, fl, fr], interior_local),   # left slope  (+y face)
            plane_hs([apex, br, bl], interior_local),   # right slope (-y face)
            HalfSpace(np.array([0.0, 0.0, -1.0]),
                      float(np.dot([0.0, 0.0, -1.0], to_world(bl)))),   # floor z=0
        ]

        vw = self._verts_world
        # 0=apex, 1=bl, 2=br, 3=fl, 4=fr
        self._edges = [
            (0, 1, HIP), (0, 2, HIP), (0, 3, HIP), (0, 4, HIP),   # 4 hip lines
            (1, 2, EAVE), (2, 4, EAVE), (4, 3, EAVE), (3, 1, EAVE),  # base rectangle
        ]
        self._faces = [
            FaceInfo(self._hs[0], vw[[0,1,3]], HIP),
            FaceInfo(self._hs[1], vw[[0,4,2]], HIP),
            FaceInfo(self._hs[2], vw[[0,3,4]], HIP),
            FaceInfo(self._hs[3], vw[[0,2,1]], HIP),
            FaceInfo(self._hs[4], vw[[1,2,4,3]], EAVE),
        ]

    def half_spaces(self):
        return self._hs

    def faces(self):
        return self._faces

    def wireframe(self):
        vw = self._verts_world
        edges = np.array([(i, j) for i, j, _ in self._edges], dtype=np.int32)
        classes = np.array([c for _, _, c in self._edges], dtype=np.int64)
        return vw.copy(), edges, classes


# ---------------------------------------------------------------------------
# P3 — FlatSlab
#
# Axis-aligned rectangular box (flat roof, parapet, terrace).
# ---------------------------------------------------------------------------

class FlatSlab(Primitive):
    """
    Flat rectangular slab.

    Parameters
    ----------
    center    : (3,) center of the top face
    theta     : azimuth rotation (radians)
    W         : width (local y)
    D         : depth (local x)
    thickness : slab thickness (extends downward from center z)
    """

    def __init__(self, center, theta: float, W: float, D: float,
                 thickness: float = 0.05) -> None:
        self.center = np.asarray(center, dtype=float)
        self.theta = float(theta)
        self.W = float(W)
        self.D = float(D)
        self.thickness = float(thickness)
        self.R = _rot_z(theta)
        self._build()

    def _build(self):
        W, D, T = self.W, self.D, self.thickness
        R, c = self.R, self.center

        def to_world(p):
            return np.asarray(p, dtype=float) @ R.T + c

        # 8 corners (local): top face at z=0, bottom at z=-T
        corners_local = np.array([
            [-D/2, -W/2,  0], [ D/2, -W/2,  0],   # 0,1 top
            [ D/2,  W/2,  0], [-D/2,  W/2,  0],   # 2,3 top
            [-D/2, -W/2, -T], [ D/2, -W/2, -T],   # 4,5 bottom
            [ D/2,  W/2, -T], [-D/2,  W/2, -T],   # 6,7 bottom
        ], dtype=float)
        self._verts_world = np.array([to_world(p) for p in corners_local])

        def hs(n_local, p_local):
            nw = R @ np.asarray(n_local, dtype=float)
            nw /= np.linalg.norm(nw)
            return HalfSpace(nw, -float(nw @ to_world(p_local)))

        self._hs = [
            hs([0, 0,  1], corners_local[0]),   # top    z=0
            hs([0, 0, -1], corners_local[4]),   # bottom z=-T
            hs([-1, 0, 0], corners_local[0]),   # -x
            hs([ 1, 0, 0], corners_local[1]),   # +x
            hs([0, -1, 0], corners_local[0]),   # -y
            hs([0,  1, 0], corners_local[3]),   # +y
        ]

        vw = self._verts_world
        self._edges = [
            (0,1,EAVE),(1,2,EAVE),(2,3,EAVE),(3,0,EAVE),   # top perimeter
            (4,5,MISC),(5,6,MISC),(6,7,MISC),(7,4,MISC),   # bottom
            (0,4,MISC),(1,5,MISC),(2,6,MISC),(3,7,MISC),   # verticals
        ]
        self._faces = [
            FaceInfo(self._hs[0], vw[[0,1,2,3]], EAVE),
            FaceInfo(self._hs[1], vw[[4,7,6,5]], MISC),
            FaceInfo(self._hs[2], vw[[0,3,7,4]], MISC),
            FaceInfo(self._hs[3], vw[[1,5,6,2]], MISC),
            FaceInfo(self._hs[4], vw[[0,4,5,1]], MISC),
            FaceInfo(self._hs[5], vw[[3,2,6,7]], MISC),
        ]

    def half_spaces(self):
        return self._hs

    def faces(self):
        return self._faces

    def wireframe(self):
        vw = self._verts_world
        edges = np.array([(i, j) for i, j, _ in self._edges], dtype=np.int32)
        classes = np.array([c for _, _, c in self._edges], dtype=np.int64)
        return vw.copy(), edges, classes


# ---------------------------------------------------------------------------
# P4 — ValleyPrism
#
# Inverted wedge — solid below a V-shaped trough.  Used with Difference to
# carve a valley between two joined RidgePrisms.
# ---------------------------------------------------------------------------

class ValleyPrism(Primitive):
    """
    Inverted wedge (valley / gutter).

    In local frame the trough runs along x with opening angle 2*alpha:
      solid = { z <= -depth + tan(alpha) * |y|,  z <= 0,  |x| <= L/2 }

    Parameters
    ----------
    center  : (3,) valley midpoint
    theta   : valley azimuth (radians)
    L       : valley length
    W       : half-width at top of trough
    depth   : trough depth at centre (valley nadir below eave plane)
    """

    def __init__(self, center, theta: float, L: float, W: float,
                 depth: float) -> None:
        self.center = np.asarray(center, dtype=float)
        self.theta = float(theta)
        self.L = float(L)
        self.W = float(W)
        self.depth = float(depth)
        self.R = _rot_z(theta)
        self._build()

    def _build(self):
        L, W, depth = self.L, self.W, self.depth
        R, c = self.R, self.center

        def to_world(p):
            return np.asarray(p, dtype=float) @ R.T + c

        # 6 vertices in local coords
        v = np.array([
            [-L/2, 0, -depth],   # 0 trough left nadir
            [ L/2, 0, -depth],   # 1 trough right nadir
            [-L/2,  W, 0],       # 2 left rim, left end
            [ L/2,  W, 0],       # 3 left rim, right end
            [-L/2, -W, 0],       # 4 right rim, left end
            [ L/2, -W, 0],       # 5 right rim, right end
        ], dtype=float)
        self._verts_world = np.array([to_world(p) for p in v])

        def hs_local(n_local, anchor_local):
            nw = R @ np.asarray(n_local, dtype=float)
            nw /= np.linalg.norm(nw)
            return HalfSpace(nw, -float(nw @ to_world(anchor_local)))

        # Outward normals: two sloped walls + top cap + two end caps
        # Left wall (y>0 side): outward points up-left
        n_lw = np.array([0.0, depth, W]) ; n_lw /= np.linalg.norm(n_lw)
        # Right wall (y<0 side): outward points up-right
        n_rw = np.array([0.0, -depth, W]); n_rw /= np.linalg.norm(n_rw)

        self._hs = [
            hs_local( n_lw, v[0]),                  # left wall
            hs_local( n_rw, v[0]),                  # right wall
            hs_local([0,0, 1], v[2]),               # top (z<=0)
            hs_local([-1,0,0], v[0]),               # left end
            hs_local([ 1,0,0], v[1]),               # right end
        ]

        vw = self._verts_world
        self._edges = [
            (0,1,VALLEY),(2,3,EAVE),(4,5,EAVE),
            (0,2,VALLEY),(0,4,VALLEY),(1,3,VALLEY),(1,5,VALLEY),
            (2,4,EAVE),(3,5,EAVE),
        ]
        self._faces = [
            FaceInfo(self._hs[0], vw[[0,1,3,2]], VALLEY),
            FaceInfo(self._hs[1], vw[[0,4,5,1]], VALLEY),
            FaceInfo(self._hs[2], vw[[2,3,5,4]], EAVE),
            FaceInfo(self._hs[3], vw[[0,2,4]],   MISC),
            FaceInfo(self._hs[4], vw[[1,5,3]],   MISC),
        ]

    def half_spaces(self):
        return self._hs

    def faces(self):
        return self._faces

    def wireframe(self):
        vw = self._verts_world
        edges = np.array([(i, j) for i, j, _ in self._edges], dtype=np.int32)
        classes = np.array([c for _, _, c in self._edges], dtype=np.int64)
        return vw.copy(), edges, classes


# ---------------------------------------------------------------------------
# P5 — DormerBox
#
# Vertical rectangular box for dormers, chimneys, mechanical units.
# Intended to be Union-ed onto a main roof body (and optionally topped with
# a small RidgePrism as a dormer roof).
# ---------------------------------------------------------------------------

class DormerBox(Primitive):
    """
    Vertical rectangular box (dormer / chimney).

    Parameters
    ----------
    center  : (3,) base-center (bottom face midpoint)
    theta   : azimuth (radians)
    W       : width (local y)
    D       : depth (local x)
    H       : height
    """

    def __init__(self, center, theta: float, W: float, D: float,
                 H: float) -> None:
        self.center = np.asarray(center, dtype=float)
        self.theta = float(theta)
        self.W = float(W)
        self.D = float(D)
        self.H = float(H)
        self.R = _rot_z(theta)
        self._build()

    def _build(self):
        W, D, H = self.W, self.D, self.H
        R, c = self.R, self.center

        def to_world(p):
            return np.asarray(p, dtype=float) @ R.T + c

        corners_local = np.array([
            [-D/2, -W/2, 0], [ D/2, -W/2, 0],
            [ D/2,  W/2, 0], [-D/2,  W/2, 0],
            [-D/2, -W/2, H], [ D/2, -W/2, H],
            [ D/2,  W/2, H], [-D/2,  W/2, H],
        ], dtype=float)
        self._verts_world = np.array([to_world(p) for p in corners_local])

        def hs(n_local, p_local):
            nw = R @ np.asarray(n_local, dtype=float)
            nw /= np.linalg.norm(nw)
            return HalfSpace(nw, -float(nw @ to_world(p_local)))

        self._hs = [
            hs([0, 0, -1], corners_local[0]),   # bottom
            hs([0, 0,  1], corners_local[4]),   # top
            hs([-1, 0, 0], corners_local[0]),
            hs([ 1, 0, 0], corners_local[1]),
            hs([0, -1, 0], corners_local[0]),
            hs([0,  1, 0], corners_local[2]),
        ]

        vw = self._verts_world
        self._edges = [
            (0,1,MISC),(1,2,MISC),(2,3,MISC),(3,0,MISC),   # bottom
            (4,5,MISC),(5,6,MISC),(6,7,MISC),(7,4,MISC),   # top
            (0,4,MISC),(1,5,MISC),(2,6,MISC),(3,7,MISC),   # verticals
        ]
        self._faces = [
            FaceInfo(self._hs[0], vw[[0,3,2,1]], MISC),
            FaceInfo(self._hs[1], vw[[4,5,6,7]], MISC),
            FaceInfo(self._hs[2], vw[[0,4,7,3]], MISC),
            FaceInfo(self._hs[3], vw[[1,2,6,5]], MISC),
            FaceInfo(self._hs[4], vw[[0,1,5,4]], MISC),
            FaceInfo(self._hs[5], vw[[3,7,6,2]], MISC),
        ]

    def half_spaces(self):
        return self._hs

    def faces(self):
        return self._faces

    def wireframe(self):
        vw = self._verts_world
        edges = np.array([(i, j) for i, j, _ in self._edges], dtype=np.int32)
        classes = np.array([c for _, _, c in self._edges], dtype=np.int64)
        return vw.copy(), edges, classes
