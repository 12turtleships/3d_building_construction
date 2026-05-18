"""
Plane fitting and roof-type classification.

FittedPlane   — RANSAC plane with normal, offset, inliers, pitch angle
ransac_planes — iterative multi-plane RANSAC (removes inliers each round)
classify_roof_type — heuristic from plane count + geometry
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Plane primitive
# ---------------------------------------------------------------------------

@dataclass
class FittedPlane:
    normal: np.ndarray   # (3,) unit normal, z >= 0
    d: float             # normal · point = d
    inliers: np.ndarray  # (M, 3) points on this plane
    pitch_deg: float     # 0 = flat horizontal, 90 = vertical wall

    @classmethod
    def from_points(cls, pts: np.ndarray) -> "FittedPlane":
        """Fit a plane to *pts* via SVD (least-squares)."""
        if len(pts) < 3:
            raise ValueError("Need ≥ 3 points to fit a plane")
        centroid = pts.mean(0)
        _, _, Vt = np.linalg.svd(pts - centroid)
        normal = Vt[-1]
        if normal[2] < 0:
            normal = -normal
        d = float(normal @ centroid)
        # pitch: angle between normal and vertical (0,0,1)
        # flat roof → normal=(0,0,1) → pitch=0
        cos_a = float(np.clip(abs(normal[2]), 0.0, 1.0))
        pitch = float(np.degrees(np.arccos(cos_a)))
        return cls(normal=normal, d=d, inliers=pts, pitch_deg=pitch)

    def distance(self, pts: np.ndarray) -> np.ndarray:
        """Signed distance from each point to this plane."""
        return (pts @ self.normal) - self.d

    def z_at(self, xy: np.ndarray) -> float:
        """Solve ax+by+cz=d for z given a 2-D point xy."""
        a, b, c = self.normal
        if abs(c) < 1e-6:
            return float(self.d / max(abs(a), abs(b), 1e-9))
        x, y = float(xy[0]), float(xy[1])
        return (self.d - a * x - b * y) / c


# ---------------------------------------------------------------------------
# Multi-plane RANSAC
# ---------------------------------------------------------------------------

def ransac_planes(
    xyz: np.ndarray,
    n_planes: int = 4,
    n_iter: int = 150,
    inlier_thresh: float = 0.02,
    min_inliers: int = 15,
) -> list[FittedPlane]:
    """
    Fit up to *n_planes* planes by iterative RANSAC.
    Inliers from each fitted plane are removed before the next round.
    """
    rng = np.random.default_rng(42)
    remaining = xyz.copy()
    planes: list[FittedPlane] = []

    for _ in range(n_planes):
        if len(remaining) < min_inliers:
            break

        best_mask: np.ndarray | None = None
        best_count = 0

        for _ in range(n_iter):
            idx = rng.choice(len(remaining), 3, replace=False)
            try:
                plane = FittedPlane.from_points(remaining[idx])
            except Exception:
                continue
            dist = np.abs(plane.distance(remaining))
            mask = dist < inlier_thresh
            count = int(mask.sum())
            if count > best_count:
                best_count = count
                best_mask = mask

        if best_count < min_inliers or best_mask is None:
            break

        plane = FittedPlane.from_points(remaining[best_mask])
        planes.append(plane)
        remaining = remaining[~best_mask]

    return planes


# ---------------------------------------------------------------------------
# Roof-type classification
# ---------------------------------------------------------------------------

FLAT    = "flat"
GABLE   = "gable"
HIP     = "hip"
SHED    = "shed"
MANSARD = "mansard"
COMPLEX = "complex"


def classify_roof_type(planes: list[FittedPlane]) -> str:
    """Classify roof type from a list of fitted planes."""
    n = len(planes)

    if n == 0:
        return FLAT

    pitches = [p.pitch_deg for p in planes]
    n_flat = sum(1 for p in pitches if p < 12)

    if n == 1:
        return FLAT if pitches[0] < 12 else SHED

    if n == 2:
        # Two tilted planes with similar pitch → gable
        if abs(pitches[0] - pitches[1]) < 20 and pitches[0] >= 10:
            return GABLE
        return SHED

    if n == 3:
        # Two main slopes + one small end → still gable
        return GABLE

    if n == 4:
        if n_flat >= 1:
            return MANSARD     # flat mid + pitched lower
        return HIP             # four sloping faces

    return COMPLEX
