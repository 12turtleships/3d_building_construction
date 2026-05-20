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


def _z_bounds(xyz: np.ndarray) -> tuple[float, float]:
    """Return (eave_z, ridge_z) as 10th/90th percentile of point z values."""
    z = xyz[:, 2]
    return float(np.percentile(z, 10)), float(np.percentile(z, 90))


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
        self._corners = rect_corners(footprint.minimum_rotated_rectangle)
        self._eave_z, self._ridge_z = _z_bounds(xyz)
        return True

    def wireframe(self):
        c = self._corners
        eave_z, ridge_z = self._eave_z, self._ridge_z
        # Determine which pair of edges is the low (eave) vs high end.
        # Use the longer axis as the ridge direction; the two short ends
        # get eave_z and ridge_z respectively.
        d01 = np.linalg.norm(c[1] - c[0])
        d12 = np.linalg.norm(c[2] - c[1])
        if d01 >= d12:
            zs = [eave_z, ridge_z, ridge_z, eave_z]
        else:
            zs = [eave_z, eave_z, ridge_z, ridge_z]
        verts = np.array([[c[i][0], c[i][1], zs[i]] for i in range(4)])
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        return verts, edges


# ---------------------------------------------------------------------------
# Gable roof (two symmetric slopes, horizontal ridge)
# ---------------------------------------------------------------------------

class GableRoof(RoofPrimitive):
    roof_type = "gable"

    def fit(self, xyz: np.ndarray, footprint: Polygon) -> bool:
        self._corners = rect_corners(footprint.minimum_rotated_rectangle)
        self._eave_z, self._ridge_z = _z_bounds(xyz)
        return True

    def wireframe(self):
        c = self._corners
        eave_z, ridge_z = self._eave_z, self._ridge_z

        d01 = np.linalg.norm(c[1] - c[0])
        d12 = np.linalg.norm(c[2] - c[1])
        if d01 >= d12:
            ridge_a = (c[0] + c[3]) / 2
            ridge_b = (c[1] + c[2]) / 2
            eave_pairs = [(0, 3), (1, 2)]
        else:
            ridge_a = (c[0] + c[1]) / 2
            ridge_b = (c[2] + c[3]) / 2
            eave_pairs = [(0, 1), (2, 3)]

        eave_verts = np.array([[c[i][0], c[i][1], eave_z] for i in range(4)])
        ridge_verts = np.array([
            [ridge_a[0], ridge_a[1], ridge_z],
            [ridge_b[0], ridge_b[1], ridge_z],
        ])
        verts = np.vstack([eave_verts, ridge_verts])  # 0-3 eave, 4-5 ridge

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),   # eave perimeter
            (4, 5),                              # ridge
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
        self._corners = rect_corners(footprint.minimum_rotated_rectangle)
        self._eave_z, self._ridge_z = _z_bounds(xyz)
        return True

    def wireframe(self):
        c = self._corners
        eave_z, ridge_z = self._eave_z, self._ridge_z

        d01 = np.linalg.norm(c[1] - c[0])
        d12 = np.linalg.norm(c[2] - c[1])
        if d01 >= d12:
            ridge_a = c[0] * 0.75 + c[1] * 0.25
            ridge_b = c[3] * 0.75 + c[2] * 0.25
        else:
            ridge_a = c[0] * 0.75 + c[3] * 0.25
            ridge_b = c[1] * 0.75 + c[2] * 0.25

        eave_verts = np.array([[c[i][0], c[i][1], eave_z] for i in range(4)])
        ridge_verts = np.array([
            [ridge_a[0], ridge_a[1], ridge_z],
            [ridge_b[0], ridge_b[1], ridge_z],
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
        self._corners = rect_corners(footprint.minimum_rotated_rectangle)
        self._lower_z, self._upper_z = _z_bounds(xyz)
        return True

    def wireframe(self):
        c = self._corners
        lower_z, upper_z = self._lower_z, self._upper_z

        cx = np.mean(c, axis=0)
        upper = c * 0.8 + cx * 0.2   # inset 20% toward centre

        lower_verts = np.array([[c[i][0], c[i][1], lower_z] for i in range(4)])
        upper_verts = np.array([[upper[i][0], upper[i][1], upper_z] for i in range(4)])
        verts = np.vstack([lower_verts, upper_verts])   # 0-3 lower, 4-7 upper

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
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
