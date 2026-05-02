"""
RoofWireframeNet — full end-to-end model.

Input per sample
----------------
  xyz_norm  : (N, 3)   float32  normalised point positions
  vote_frac : (N,)     float32  per-point view-agreement score
  class_id  : (N,)     int64    semantic class label (0–255)
  mask      : (N,)     bool     valid point indicator

The 4-channel per-point feature = [vote_frac, n_views_voted (float), mask, class_id (float)].
class_id is embedded via a small learnable embedding table.

Pipeline
--------
  PointNetBackbone → VertexDecoder → EdgePredictor
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import PointNetBackbone
from .decoder  import VertexDecoder, EdgePredictor


class RoofWireframeNet(nn.Module):
    """
    Parameters
    ----------
    n_queries       : number of vertex query slots (K)
    n_edge_classes  : number of edge type classes (default 6, see primitives.py)
    n_sem_classes   : number of semantic class IDs in the point cloud
    """

    def __init__(self,
                 n_queries: int = 64,
                 n_edge_classes: int = 10,
                 n_sem_classes: int = 64,
                 backbone_global_dim: int = 1024,
                 vertex_feat_dim: int = 128) -> None:
        super().__init__()

        # Small embedding for semantic class id (avoids treating ordinal ID as continuous)
        self.class_embed = nn.Embedding(n_sem_classes, 8, padding_idx=0)

        # in_channels = 3(xyz) + 1(vote_frac) + 1(n_views) + 1(mask) + 8(class_embed)
        in_ch = 3 + 1 + 1 + 1 + 8

        self.backbone = PointNetBackbone(
            in_channels=in_ch,
            point_dims=[64, 128, 256],
            global_dims=[512, backbone_global_dim],
        )

        self.vertex_decoder = VertexDecoder(
            n_queries=n_queries,
            global_dim=backbone_global_dim,
            hidden_dim=256,
            feat_dim=vertex_feat_dim,
        )

        self.edge_predictor = EdgePredictor(
            feat_dim=vertex_feat_dim,
            global_dim=backbone_global_dim,
            n_edge_classes=n_edge_classes,
        )

    # ------------------------------------------------------------------
    def forward(self,
                xyz:        torch.Tensor,   # (B, N, 3)
                vote_frac:  torch.Tensor,   # (B, N)
                n_views:    torch.Tensor,   # (B, N)   — n_views_voted
                mask:       torch.Tensor,   # (B, N)   — bool/float
                class_id:   torch.Tensor,   # (B, N)   — int
                ) -> dict[str, torch.Tensor]:
        B, N, _ = xyz.shape

        # Build per-point feature tensor
        class_emb = self.class_embed(class_id.clamp(0, self.class_embed.num_embeddings - 1))
        # (B, N, 8)

        feats = torch.cat([
            vote_frac.unsqueeze(-1).float(),             # (B, N, 1)
            (n_views.float() / 8.0).unsqueeze(-1),       # (B, N, 1) normalised
            mask.float().unsqueeze(-1),                  # (B, N, 1)
            class_emb,                                   # (B, N, 8)
        ], dim=-1)                                       # (B, N, 11)

        # Backbone
        global_feat, _ = self.backbone(xyz, feats)       # (B, 1024), (B, N, 256)

        # Vertex proposals
        pred_pos, pred_conf, vert_feat = self.vertex_decoder(global_feat)
        # (B,K,3), (B,K), (B,K,feat_dim)

        # Edge predictions
        edge_logits = self.edge_predictor(vert_feat, pred_pos, global_feat)
        # (B, K, K, n_cls+1)

        return {
            "pred_pos":    pred_pos,     # (B, K, 3)
            "pred_conf":   pred_conf,    # (B, K)
            "edge_logits": edge_logits,  # (B, K, K, n_edge_classes+1)
        }
