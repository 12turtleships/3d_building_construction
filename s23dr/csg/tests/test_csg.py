"""
Geometric correctness tests for the CSG primitives and evaluator.

Run with:  python -m pytest s23dr/csg/tests/test_csg.py -v
"""

import numpy as np
import pytest

from s23dr.csg.primitives import (
    RidgePrism, HipEnd, FlatSlab, ValleyPrism, DormerBox,
    RIDGE, HIP, VALLEY, EAVE, GABLE,
)
from s23dr.csg.tree import leaf, union, intersection, difference
from s23dr.csg.evaluator import evaluate_csg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_wireframe(verts, edges, classes):
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert edges.ndim == 2 and edges.shape[1] == 2
    assert classes.ndim == 1
    assert len(edges) == len(classes)
    if len(edges) == 0:
        return
    # No self-loops
    assert np.all(edges[:, 0] != edges[:, 1])
    # All indices in range
    assert edges.max() < len(verts)
    assert edges.min() >= 0


# ---------------------------------------------------------------------------
# Primitive: RidgePrism
# ---------------------------------------------------------------------------

class TestRidgePrism:
    def setup_method(self):
        # center = eave-floor midpoint; ridge is at center_z + H = 1.0
        self.p = RidgePrism(
            center=[0, 0, 0],
            theta=0.0,
            L=4.0, W_l=2.0, W_r=2.0, H=1.0,
        )

    def test_inside_ridge(self):
        """Points between eave floor and ridge are inside."""
        pts = np.array([[0, 0, 0.5], [0, 0, 0.0]])
        assert self.p.inside(pts).all()

    def test_outside_above_ridge(self):
        """Points above ridge are outside."""
        pts = np.array([[0, 0, 1.5]])   # above ridge z=H=1
        assert not self.p.inside(pts).any()

    def test_outside_beyond_eave(self):
        pts = np.array([[0, 3.0, 0.0]])   # y > W_l
        assert not self.p.inside(pts).any()

    def test_wireframe_topology(self):
        v, e, c = self.p.wireframe()
        _check_wireframe(v, e, c)
        # Gable prism: 6 vertices, 9 edges
        assert len(v) == 6
        assert len(e) == 9

    def test_ridge_edge_class(self):
        v, e, c = self.p.wireframe()
        assert RIDGE in c

    def test_eave_edges(self):
        v, e, c = self.p.wireframe()
        assert np.sum(c == EAVE) >= 2


# ---------------------------------------------------------------------------
# Primitive: HipEnd
# ---------------------------------------------------------------------------

class TestHipEnd:
    def setup_method(self):
        self.p = HipEnd(center=[0, 0, 0], theta=0.0, W=4.0, D=4.0, H=2.0)

    def test_apex_inside(self):
        pts = np.array([[0, 0, 1.5]])   # below apex
        assert self.p.inside(pts).all()

    def test_outside_beyond_base(self):
        pts = np.array([[0, 3.0, 0.0]])   # y > W/2
        assert not self.p.inside(pts).any()

    def test_wireframe_topology(self):
        v, e, c = self.p.wireframe()
        _check_wireframe(v, e, c)
        # 5 vertices (1 apex + 4 base), 8 edges (4 hip + 4 base)
        assert len(v) == 5
        assert len(e) == 8

    def test_hip_edges(self):
        v, e, c = self.p.wireframe()
        assert np.sum(c == HIP) == 4


# ---------------------------------------------------------------------------
# Primitive: FlatSlab
# ---------------------------------------------------------------------------

class TestFlatSlab:
    def setup_method(self):
        self.p = FlatSlab(center=[0, 0, 1], theta=0.0, W=3.0, D=3.0, thickness=0.1)

    def test_inside(self):
        pts = np.array([[0, 0, 0.95]])   # just below top face
        assert self.p.inside(pts).all()

    def test_outside(self):
        pts = np.array([[0, 0, 1.05]])   # above top face
        assert not self.p.inside(pts).any()

    def test_wireframe_topology(self):
        v, e, c = self.p.wireframe()
        _check_wireframe(v, e, c)
        assert len(v) == 8
        assert len(e) == 12


# ---------------------------------------------------------------------------
# CSG: Hip roof = Intersection(RidgePrism, HipEnd_left, HipEnd_right)
# ---------------------------------------------------------------------------

class TestHipRoof:
    """
    Hip roof = Union(central RidgePrism, HipEnd_left, HipEnd_right).
    Each HipEnd apex sits at the ridge endpoint; base extends to the hip feet.
    """
    def setup_method(self):
        # Ridge L=4 (shorter than eave W=6 to leave room for hip ends)
        ridge = RidgePrism(center=[0, 0, 0], theta=0.0, L=4.0, W_l=3.0, W_r=3.0, H=2.0)
        hip_l = HipEnd(center=[-2, 0, 0], theta=0.0, W=6.0, D=6.0, H=2.0)
        hip_r = HipEnd(center=[ 2, 0, 0], theta=0.0, W=6.0, D=6.0, H=2.0)
        tree = union(union(leaf(ridge), leaf(hip_l)), leaf(hip_r))
        self.v, self.e, self.c = evaluate_csg(tree)

    def test_produces_wireframe(self):
        _check_wireframe(self.v, self.e, self.c)

    def test_has_hip_edges(self):
        assert HIP in self.c, "Hip roof must have HIP edges"

    def test_has_ridge_edge(self):
        assert RIDGE in self.c, "Hip roof must have a central ridge edge"

    def test_vertices_count(self):
        assert len(self.v) >= 6


# ---------------------------------------------------------------------------
# CSG: Cross-gable = Union(RidgePrism_main, RidgePrism_wing)
# ---------------------------------------------------------------------------

class TestCrossGable:
    def setup_method(self):
        main = RidgePrism(center=[0, 0, 0], theta=0.0,   L=8.0, W_l=3.0, W_r=3.0, H=2.0)
        wing = RidgePrism(center=[0, 0, 0], theta=np.pi/2, L=6.0, W_l=2.5, W_r=2.5, H=2.0)
        self.v, self.e, self.c = evaluate_csg(union(leaf(main), leaf(wing)))

    def test_produces_wireframe(self):
        _check_wireframe(self.v, self.e, self.c)

    def test_has_valley_edges(self):
        # Union of two RidgePrisms creates valley seams where they intersect
        assert VALLEY in self.c, "Cross-gable must have VALLEY edges"

    def test_has_edges(self):
        assert len(self.e) > 0


# ---------------------------------------------------------------------------
# CSG: Dormer = Union(RidgePrism_main, DormerBox + RidgePrism_dormer)
# ---------------------------------------------------------------------------

class TestDormerOnRoof:
    def setup_method(self):
        main = RidgePrism(center=[0, 0, 0], theta=0.0, L=8.0, W_l=3.0, W_r=3.0, H=2.0)
        box  = DormerBox(center=[1, 2, 0], theta=np.pi/2, W=1.5, D=1.0, H=1.5)
        dormer_ridge = RidgePrism(
            center=[1, 2, 1.5], theta=np.pi/2, L=1.0, W_l=0.5, W_r=0.5, H=0.5
        )
        dormer = union(leaf(box), leaf(dormer_ridge))
        self.v, self.e, self.c = evaluate_csg(union(leaf(main), dormer))

    def test_produces_wireframe(self):
        _check_wireframe(self.v, self.e, self.c)

    def test_has_edges(self):
        assert len(self.e) > 0


# ---------------------------------------------------------------------------
# Numerical: membership test matches evaluator geometry
# ---------------------------------------------------------------------------

class TestMembership:
    def test_union_inside(self):
        a = RidgePrism(center=[-2, 0, 0], theta=0.0, L=3.0, W_l=1.5, W_r=1.5, H=1.0)
        b = RidgePrism(center=[ 2, 0, 0], theta=0.0, L=3.0, W_l=1.5, W_r=1.5, H=1.0)
        tree = union(leaf(a), leaf(b))
        pts_in_a = np.array([[-2, 0, 0.5]])
        pts_in_b = np.array([[ 2, 0, 0.5]])
        assert tree.inside(pts_in_a).all()
        assert tree.inside(pts_in_b).all()

    def test_difference_removes(self):
        # RidgePrism: eave at z=0, ridge at z=2. FlatSlab: top at z=1.5, bottom at z=-0.5.
        a = RidgePrism(center=[0, 0, 0], theta=0.0, L=6.0, W_l=2.0, W_r=2.0, H=2.0)
        b = FlatSlab(center=[0, 0, 1.5], theta=0.0, W=5.0, D=5.0, thickness=2.0)
        tree = difference(leaf(a), leaf(b))
        # z=1.0 is inside both A (z∈[0,2]) and B (z∈[-0.5,1.5]) → removed
        pts = np.array([[0, 0, 1.0]])
        assert not tree.inside(pts).any()
        # z=1.8 is inside A but above B (B top=1.5) → kept
        pts2 = np.array([[0, 0, 1.8]])
        assert tree.inside(pts2).all()
