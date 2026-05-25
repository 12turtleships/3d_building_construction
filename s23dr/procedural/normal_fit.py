"""
Normal-based roof primitive fitting.

For a given set of points + their estimated normals, fits three primitive
types and returns the one with the lowest normal-misfit cost.

Primitive types
---------------
flat   : all normals → (0,0,1)   cost = mean(1 - n_z)
gable  : two slopes, ridge along long axis; slope optimised via golden section
hip    : four slopes, one per rectangle side; slope optimised similarly

Cost metric
-----------
  cost(primitive) = mean over points of  (1 - |n_obs · n_pred|)
Range [0, 1]:  0 = perfect alignment,  1 = orthogonal.

The fitting is done in a local 2-D frame (u, v) aligned to the rectangle
axes so that gable/hip formulas are axis-aligned.

Public API
----------
  fit_best(xyz, normals, rect_corners)
      → (cost, type_str, params_dict)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

# Roof type labels
FLAT  = "flat"
GABLE = "gable"
HIP   = "hip"


# ---------------------------------------------------------------------------
# Top-level: pick best primitive
# ---------------------------------------------------------------------------

def fit_best(
    xyz: np.ndarray,      # (N, 3) points inside rectangle
    normals: np.ndarray,  # (N, 3) unit surface normals
    corners: np.ndarray,  # (4, 2) rectangle corners in world XY
) -> tuple[float, str, dict]:
    """
    Fit flat / gable / hip to the points and return the lowest-cost result.

    Returns
    -------
    cost  : float in [0, 1]
    rtype : one of FLAT / GABLE / HIP
    params: dict with keys depending on rtype (see below)
    """
    if len(xyz) < 4:
        z_med = float(np.median(xyz[:, 2])) if len(xyz) else 0.0
        return 1.0, FLAT, {"z": z_med}

    # Build local orthonormal frame from rectangle corners
    origin, u_hat, v_hat, width, height = _rect_frame(corners)

    # Project points into local frame
    delta = xyz[:, :2] - origin            # (N, 2)
    u_local = delta @ u_hat               # signed coordinate along long axis
    v_local = delta @ v_hat               # signed coordinate along short axis

    # Rotate normals into local frame (only in-plane components matter)
    n_u = normals @ np.append(u_hat, 0)   # (N,)
    n_v = normals @ np.append(v_hat, 0)   # (N,)
    n_z = normals[:, 2]                    # (N,)

    # ── Flat ─────────────────────────────────────────────────────────────────
    flat_cost, flat_params = _fit_flat(normals, xyz)

    # ── Gable (ridge along long axis = u) ────────────────────────────────────
    gable_cost, gable_params = _fit_gable(
        u_local, v_local, n_v, n_z, xyz, width, height, origin, u_hat, v_hat,
        ridge_along_long=True,
    )

    # ── Hip ───────────────────────────────────────────────────────────────────
    hip_cost, hip_params = _fit_hip(
        u_local, v_local, n_u, n_v, n_z, xyz, width, height, origin, u_hat, v_hat,
    )

    best_cost = min(flat_cost, gable_cost, hip_cost)
    if best_cost == flat_cost:
        return flat_cost, FLAT, flat_params
    if best_cost == gable_cost:
        return gable_cost, GABLE, gable_params
    return hip_cost, HIP, hip_params


# ---------------------------------------------------------------------------
# Flat
# ---------------------------------------------------------------------------

def _fit_flat(
    normals: np.ndarray,
    xyz: np.ndarray,
) -> tuple[float, dict]:
    cost = float(1.0 - np.abs(normals[:, 2]).mean())
    z = float(np.percentile(xyz[:, 2], 75))
    return cost, {"z": z}


# ---------------------------------------------------------------------------
# Gable
# ---------------------------------------------------------------------------

def _fit_gable(
    u_local, v_local, n_v, n_z, xyz,
    width, height, origin, u_hat, v_hat,
    ridge_along_long: bool = True,
) -> tuple[float, dict]:
    """Ridge runs along the u-axis (long axis).  v is the transverse direction."""

    # Predicted normals for a gable at slope angle α (from horizontal):
    #   n_pred = (0, side*sin(α), cos(α))  in local (u, v, z) frame
    # where side = sign(v_local) from the ridge

    side = np.sign(v_local)
    side[side == 0] = 1.0

    def _cost(alpha_deg: float) -> float:
        alpha = np.radians(alpha_deg)
        pred_v = side * np.sin(alpha)
        pred_z = np.full(len(n_z), np.cos(alpha))
        # |n_obs · n_pred| = |n_v*pred_v + n_z*pred_z| (u-component is 0)
        dots = np.abs(n_v * pred_v + n_z * pred_z)
        return float(1.0 - np.clip(dots, 0, 1).mean())

    result = minimize_scalar(_cost, bounds=(5.0, 80.0), method="bounded")
    best_alpha_deg = float(result.x)
    best_cost = float(result.fun)

    alpha  = np.radians(best_alpha_deg)
    eave_z = float(np.percentile(xyz[:, 2], 10))

    # Use actual z of points near the ridge centerline (|v_local| < width/4)
    # rather than tan(α)*dim, which is unreliable with sparse/noisy normals.
    near_ridge = np.abs(v_local) < (width / 4.0)
    if near_ridge.sum() >= 2:
        ridge_z = float(np.percentile(xyz[near_ridge, 2], 90))
    else:
        ridge_z = eave_z + np.tan(alpha) * (height / 2.0)

    return best_cost, {
        "alpha_deg": best_alpha_deg,
        "eave_z":    eave_z,
        "ridge_z":   ridge_z,
        "origin":    origin,
        "u_hat":     u_hat,
        "v_hat":     v_hat,
        "width":     width,
        "height":    height,
        "corners":   None,
    }


# ---------------------------------------------------------------------------
# Hip
# ---------------------------------------------------------------------------

def _fit_hip(
    u_local, v_local, n_u, n_v, n_z, xyz,
    width, height, origin, u_hat, v_hat,
) -> tuple[float, dict]:
    """
    Hip roof: 4 slopes from 4 rectangle sides.
    At each point the active face is determined by which edge is closest
    in the perpendicular sense (same as the lower-envelope of 4 planes).
    Active face determines the predicted normal direction.
    """
    # Distance from each edge (in local frame):
    #   top/bottom (v-faces): dist_v = height/2 - |v_local|
    #   left/right (u-faces): dist_u = width/2  - |u_local|
    dist_v = height / 2.0 - np.abs(v_local)   # (N,)  positive inside
    dist_u = width  / 2.0 - np.abs(u_local)   # (N,)  positive inside

    # Active face per point: the one with the SMALLER "inward distance"
    # (i.e. the edge this point is closest to when walking inward)
    u_face_active = dist_u < dist_v           # left/right slope active

    side_v = np.sign(v_local);  side_v[side_v == 0] = 1.0
    side_u = np.sign(u_local);  side_u[side_u == 0] = 1.0

    def _cost(alpha_deg: float) -> float:
        alpha = np.radians(alpha_deg)
        sa, ca = np.sin(alpha), np.cos(alpha)
        # v-face: predicted normal has v-component
        pred_v_face = np.abs(n_v * (side_v * sa) + n_z * ca)
        # u-face: predicted normal has u-component
        pred_u_face = np.abs(n_u * (side_u * sa) + n_z * ca)
        dots = np.where(u_face_active, pred_u_face, pred_v_face)
        return float(1.0 - np.clip(dots, 0, 1).mean())

    result = minimize_scalar(_cost, bounds=(5.0, 80.0), method="bounded")
    best_alpha_deg = float(result.x)
    best_cost = float(result.fun)

    alpha  = np.radians(best_alpha_deg)
    eave_z = float(np.percentile(xyz[:, 2], 10))

    # Use actual z of points near the roof apex (small |u_local| and |v_local|)
    near_top = (np.abs(u_local) < height / 4.0) & (np.abs(v_local) < width / 4.0)
    if near_top.sum() >= 2:
        ridge_z = float(np.percentile(xyz[near_top, 2], 90))
    else:
        ridge_z = eave_z + np.tan(alpha) * min(width, height) / 2.0

    return best_cost, {
        "alpha_deg": best_alpha_deg,
        "eave_z":    eave_z,
        "ridge_z":   ridge_z,
        "origin":    origin,
        "u_hat":     u_hat,
        "v_hat":     v_hat,
        "width":     width,
        "height":    height,
    }


# ---------------------------------------------------------------------------
# Wireframe generation from fitted params
# ---------------------------------------------------------------------------

def wireframe_from_params(
    rtype: str,
    params: dict,
    corners: np.ndarray,   # (4, 2) rectangle corners
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Convert fitted primitive params to 3-D vertices + edges."""
    if rtype == FLAT:
        return _wf_flat(corners, params["z"])
    if rtype == GABLE:
        return _wf_gable(corners, params)
    if rtype == HIP:
        return _wf_hip(corners, params)
    return _wf_flat(corners, params.get("z", 0.0))


def _wf_flat(corners: np.ndarray, z: float):
    verts = np.array([[c[0], c[1], z] for c in corners], dtype=np.float32)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    return verts, edges


def _wf_gable(corners: np.ndarray, p: dict):
    """
    Gable: ridge along long axis (u).  Vertices at 4 corners (eave) + 2 ridge ends.

    Corners are rebuilt from the frame params so that the orientation is always
    correct regardless of the winding order in the input `corners` array.
    Frame convention:  c[0]=-u/-v  c[1]=+u/-v  c[2]=+u/+v  c[3]=-u/+v
    """
    origin  = p["origin"]
    u_hat   = p["u_hat"]
    v_hat   = p["v_hat"]
    width   = p["width"]    # along v (short axis)
    height  = p["height"]   # along u (long axis)
    eave_z  = p["eave_z"]
    ridge_z = p["ridge_z"]

    hu = (height / 2.0) * u_hat
    hv = (width  / 2.0) * v_hat
    c = np.array([
        origin - hu - hv,   # 0: −u, −v
        origin + hu - hv,   # 1: +u, −v
        origin + hu + hv,   # 2: +u, +v
        origin - hu + hv,   # 3: −u, +v
    ])

    verts = [[c[i][0], c[i][1], eave_z] for i in range(4)]

    # Ridge endpoints at the ends of the long axis (centre of each short side)
    r0 = origin - hu   # at −u end
    r1 = origin + hu   # at +u end
    verts.append([float(r0[0]), float(r0[1]), ridge_z])   # idx 4
    verts.append([float(r1[0]), float(r1[1]), ridge_z])   # idx 5

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),    # eave perimeter
        (0, 4), (3, 4),                      # gable end at −u
        (1, 5), (2, 5),                      # gable end at +u
        (4, 5),                              # ridge
    ]
    return np.array(verts, dtype=np.float32), edges


def _wf_hip(corners: np.ndarray, p: dict):
    """
    Hip: 4 slopes.  Vertices at 4 corners (eave) + ridge line (2 pts) or apex (1 pt).

    Corners rebuilt from frame params (same convention as _wf_gable):
    c[0]=−u/−v  c[1]=+u/−v  c[2]=+u/+v  c[3]=−u/+v
    r0 = origin + ridge_half * u_hat  (at +u),  idx 4
    r1 = origin − ridge_half * u_hat  (at −u),  idx 5
    Hip edges: +u corners (1,2) → r0 ; −u corners (0,3) → r1
    """
    origin  = p["origin"]
    u_hat   = p["u_hat"]
    v_hat   = p["v_hat"]
    width   = p["width"]
    height  = p["height"]
    eave_z  = p["eave_z"]
    ridge_z = p["ridge_z"]
    alpha   = np.radians(p["alpha_deg"])

    hu = (height / 2.0) * u_hat
    hv = (width  / 2.0) * v_hat
    c = np.array([
        origin - hu - hv,   # 0: −u, −v
        origin + hu - hv,   # 1: +u, −v
        origin + hu + hv,   # 2: +u, +v
        origin - hu + hv,   # 3: −u, +v
    ])

    verts = [[c[i][0], c[i][1], eave_z] for i in range(4)]

    half_u = height / 2.0
    half_v = width  / 2.0

    hip_run    = half_v / np.tan(alpha) if np.tan(alpha) > 1e-6 else half_u
    ridge_half = max(half_u - hip_run, 0.0)

    if ridge_half < 1e-3:
        # Pyramid: single apex at centre
        verts.append([float(origin[0]), float(origin[1]), ridge_z])  # idx 4
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (0, 4), (1, 4), (2, 4), (3, 4),
        ]
    else:
        # Ridge: two endpoints along u_hat
        r0_xy = origin + ridge_half * u_hat   # idx 4, at +u
        r1_xy = origin - ridge_half * u_hat   # idx 5, at −u
        verts.append([float(r0_xy[0]), float(r0_xy[1]), ridge_z])
        verts.append([float(r1_xy[0]), float(r1_xy[1]), ridge_z])

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),   # eave perimeter
            (1, 4), (2, 4),                     # hips at +u: corners 1,2 → r0
            (0, 5), (3, 5),                     # hips at −u: corners 0,3 → r1
            (4, 5),                             # ridge
        ]

    return np.array(verts, dtype=np.float32), edges


# ---------------------------------------------------------------------------
# Rectangle frame helper
# ---------------------------------------------------------------------------

def _rect_frame(corners: np.ndarray):
    """
    Return (origin, u_hat, v_hat, width, height) for a rectangle.
    u_hat is the longer axis, v_hat the shorter.
    origin is the rectangle centre.
    """
    # Two edge vectors from corner 0
    e1 = corners[1] - corners[0]
    e2 = corners[3] - corners[0]
    l1, l2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))

    if l1 >= l2:
        u_hat = e1 / max(l1, 1e-9)
        v_hat = e2 / max(l2, 1e-9)
        width, height = l2, l1
    else:
        u_hat = e2 / max(l2, 1e-9)
        v_hat = e1 / max(l1, 1e-9)
        width, height = l1, l2

    origin = corners.mean(axis=0)
    return origin, u_hat, v_hat, width, height
