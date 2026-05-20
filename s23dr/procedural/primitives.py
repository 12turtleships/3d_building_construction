"""
Roof primitive classes.

Each primitive encapsulates one roof section.

Interface
---------
  prim.fit(xyz, footprint) -> bool        fit parameters from data
  prim.wireframe()         -> (V, E)      analytically exact vertices + edges

Vertex/edge indices are local to each primitive; the pipeline offsets them
when merging multiple sections into one wireframe.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from shapely.geometry import Polygon

from .footprint import rect_corners
from .roof_fit import FittedPlane, ransac_planes, classify_roof_type


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class RoofPrimitive(ABC):
    roof_type: str = "unknown"

    @abstractmethod
    def fit(self, xyz: np.ndarray, footprint: Polygon) -> bool:
        """Fit to *xyz* points above *footprint*. Return False on failure."""

    @abstractmethod
    def wireframe(self) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Return (vertices (V,3), edges list-of-(i,j))."""


# ---------------------------------------------------------------------------
# Flat roof
# ---------------------------------------------------------------------------

class FlatRoof(RoofPrimitive):
    roof_type = FLAT = "flat"

    def fit(self, xyz: np.ndarray, footprint: Polygon) -> bool:
        self._corners = rect_corners(footprint.minimum_rotated_rectangle)
        self._z = float(np.percentile(xyz[:, 2], 75)) if len(xyz) > 0 else 0.0
        return True

    def wireframe(self):
        c = self._corners
        z = self._z
        verts = np.array([[c[i][0], c[i][1], z] for i in range(4)])
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        return verts, edges


# ---------------------------------------------------------------------------
# Shed roof (single tilted plane)
# ---------------------------------------------------------------------------

class ShedRoof(RoofPrimitive):
    roof_type = "shed"

    def fit(self, xyz: np.ndarray, footprint: Polygon) -> bool:
        planes = ransac_planes(xyz, n_planes=1)
        if not planes:
            return False
        self._plane = planes[0]
        self._corners = rect_corners(footprint.minimum_rotated_rectangle)
        return True

    def wireframe(self):
        c = self._corners
        p = self._plane
        verts = np.array([[c[i][0], c[i][1], p.z_at(c[i])] for i in range(4)])
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        return verts, edges


# ---------------------------------------------------------------------------
# Gable roof (two symmetric slopes, horizontal ridge)
# ---------------------------------------------------------------------------

class GableRoof(RoofPrimitive):
    roof_type = "gable"

    def fit(self, xyz: np.ndarray, footprint: Polygon) -> bool:
        planes = ransac_planes(xyz, n_planes=2)
        if len(planes) < 1:
            return False
        self._planes = planes
        self._corners = rect_corners(footprint.minimum_rotated_rectangle)
        return True

    def wireframe(self):
        c = self._corners       # 4 corners of bounding rectangle
        planes = self._planes

        # Long axis: direction of the ridge
        # c[0]-c[1] and c[2]-c[3] are the two long edges;
        # c[1]-c[2] and c[3]-c[0] are the gable ends.
        # Figure out which axis is longer.
        d01 = np.linalg.norm(c[1] - c[0])
        d12 = np.linalg.norm(c[2] - c[1])
        if d01 >= d12:
            # Long axis: 0→1 and 3→2; ridge midpoints at midpoints of short edges
            ridge_a = (c[0] + c[3]) / 2
            ridge_b = (c[1] + c[2]) / 2
            eave_pairs = [(0, 3), (1, 2)]   # indices forming the two eave edges
        else:
            ridge_a = (c[0] + c[1]) / 2
            ridge_b = (c[2] + c[3]) / 2
            eave_pairs = [(0, 1), (2, 3)]

        def _eave_z(pt):
            if len(planes) == 1:
                return planes[0].z_at(pt)
            return min(p.z_at(pt) for p in planes)

        def _ridge_z(pt):
            if len(planes) == 1:
                return planes[0].z_at(pt)
            return float(np.mean([p.z_at(pt) for p in planes]))

        # 6 vertices: 4 eave corners + 2 ridge endpoints
        eave_verts = np.array([
            [c[i][0], c[i][1], _eave_z(c[i])] for i in range(4)
        ])
        ridge_verts = np.array([
            [ridge_a[0], ridge_a[1], _ridge_z(ridge_a)],
            [ridge_b[0], ridge_b[1], _ridge_z(ridge_b)],
        ])
        verts = np.vstack([eave_verts, ridge_verts])  # 0-3 eave, 4-5 ridge

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),   # eave perimeter
            (4, 5),                              # ridge
            # rakes: each gable end connects two eave corners to the ridge
            (eave_pairs[0][0], 4), (eave_pairs[0][1], 4),
            (eave_pairs[1][0], 5), (eave_pairs[1][1], 5),
        ]
        return verts, edges


# ---------------------------------------------------------------------------
# Hip roof (four slopes meeting at a central ridge)
# ---------------------------------------------------------------------------

class HipRoof(RoofPrimitive):
    roof_type = "hip"

    def fit(self, xyz: np.ndarray, footprint: Polygon) -> bool:
        planes = ransac_planes(xyz, n_planes=4)
        if not planes:
            return False
        self._planes = planes
        self._corners = rect_corners(footprint.minimum_rotated_rectangle)
        return True

    def wireframe(self):
        c = self._corners
        planes = self._planes

        def _z(pt):
            if not planes:
                return 0.0
            return float(np.mean([p.z_at(pt) for p in planes]))

        # Eave corners at the footprint corners
        # Ridge endpoints inset 25% from the short ends along the long axis
        d01 = np.linalg.norm(c[1] - c[0])
        d12 = np.linalg.norm(c[2] - c[1])
        if d01 >= d12:
            ridge_a = c[0] * 0.75 + c[1] * 0.25
            ridge_b = c[3] * 0.75 + c[2] * 0.25
        else:
            ridge_a = c[0] * 0.75 + c[3] * 0.25
            ridge_b = c[1] * 0.75 + c[2] * 0.25

        eave_verts = np.array([[c[i][0], c[i][1], _z(c[i])] for i in range(4)])
        ridge_verts = np.array([
            [ridge_a[0], ridge_a[1], _z(ridge_a)],
            [ridge_b[0], ridge_b[1], _z(ridge_b)],
        ])
        verts = np.vstack([eave_verts, ridge_verts])  # 0-3 eave, 4-5 ridge

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),   # eave perimeter
            (4, 5),                              # ridge
            (0, 4), (1, 4),                     # front hip lines
            (2, 5), (3, 5),                     # back hip lines
        ]
        return verts, edges


# ---------------------------------------------------------------------------
# Mansard roof (lower steep + upper flat or shallow)
# ---------------------------------------------------------------------------

class MansardRoof(RoofPrimitive):
    roof_type = "mansard"

    def fit(self, xyz: np.ndarray, footprint: Polygon) -> bool:
        planes = ransac_planes(xyz, n_planes=4)
        if len(planes) < 2:
            return False
        # Sort by pitch: steep planes first, flat last
        planes.sort(key=lambda p: p.pitch_deg, reverse=True)
        self._lower_planes = planes[:4]
        self._corners = rect_corners(footprint.minimum_rotated_rectangle)
        return True

    def wireframe(self):
        c = self._corners
        planes = self._lower_planes

        def _z(pt):
            return float(np.mean([p.z_at(pt) for p in planes]))

        # Two tiers: lower eave (footprint corners) and upper eave (inset 20%)
        cx = np.mean(c, axis=0)
        upper = c * 0.8 + cx * 0.2   # inset 20% toward centre

        lower_verts = np.array([[c[i][0], c[i][1], _z(c[i])] for i in range(4)])
        upper_verts = np.array([[upper[i][0], upper[i][1], _z(upper[i])] for i in range(4)])
        verts = np.vstack([lower_verts, upper_verts])   # 0-3 lower, 4-7 upper

        edges = [
            # lower eave perimeter
            (0, 1), (1, 2), (2, 3), (3, 0),
            # upper eave perimeter
            (4, 5), (5, 6), (6, 7), (7, 4),
            # hip lines connecting lower to upper
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        return verts, edges


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "flat":    FlatRoof,
    "shed":    ShedRoof,
    "gable":   GableRoof,
    "hip":     HipRoof,
    "mansard": MansardRoof,
    "complex": HipRoof,    # fallback for complex shapes
}


def fit_best_primitive(xyz: np.ndarray, footprint: Polygon) -> RoofPrimitive:
    """Classify the roof type and fit the appropriate primitive."""
    planes = ransac_planes(xyz, n_planes=4)
    roof_type = classify_roof_type(planes)

    # Override: large z-range signals a pitched roof even if RANSAC found only
    # one dominant plane (sparse SfM points cluster on edges, not on faces).
    if len(xyz) >= 5:
        z_range = float(np.percentile(xyz[:, 2], 90) - np.percentile(xyz[:, 2], 10))
        if roof_type in ("flat", "shed") and z_range > 0.15:
            roof_type = "hip" if len(planes) >= 3 else "gable"

    prim = _TYPE_MAP.get(roof_type, FlatRoof)()
    if not prim.fit(xyz, footprint):
        prim = FlatRoof()
        prim.fit(xyz, footprint)
    return prim
