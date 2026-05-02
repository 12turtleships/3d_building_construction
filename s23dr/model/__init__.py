from .net import RoofWireframeNet
from .loss import WireframeLoss

# Dataset has 10 edge-type classes (0-9); +1 for "no edge" sentinel
N_EDGE_CLASSES = 10

__all__ = ["RoofWireframeNet", "WireframeLoss", "N_EDGE_CLASSES"]
