"""
CSG tree node.

A CSGNode is either:
  - a leaf (holds a Primitive)
  - an internal node with op ∈ {union, intersection, difference} and two children

Usage
-----
    t = union(leaf(RidgePrism(...)), leaf(RidgePrism(...)))
    t = difference(union(leaf(A), leaf(B)), leaf(C))
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .primitives import Primitive

Op = Literal["leaf", "union", "intersection", "difference"]


@dataclass
class CSGNode:
    op: Op
    primitive: Optional["Primitive"] = None   # set for leaf nodes only
    left:  Optional["CSGNode"] = None
    right: Optional["CSGNode"] = None

    def inside(self, pts):
        """Recursive CSG membership test — useful for debugging."""
        import numpy as np
        pts = np.atleast_2d(pts)
        if self.op == "leaf":
            return self.primitive.inside(pts)
        L = self.left.inside(pts)
        R = self.right.inside(pts)
        if self.op == "union":
            return L | R
        if self.op == "intersection":
            return L & R
        if self.op == "difference":
            return L & ~R
        raise ValueError(self.op)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def leaf(primitive: "Primitive") -> CSGNode:
    return CSGNode(op="leaf", primitive=primitive)

def union(a: CSGNode, b: CSGNode) -> CSGNode:
    return CSGNode(op="union", left=a, right=b)

def intersection(a: CSGNode, b: CSGNode) -> CSGNode:
    return CSGNode(op="intersection", left=a, right=b)

def difference(a: CSGNode, b: CSGNode) -> CSGNode:
    """Solid A minus solid B."""
    return CSGNode(op="difference", left=a, right=b)
