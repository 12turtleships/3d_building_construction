"""
Vertex set decoder and edge predictor for roof wireframe extraction.

Vertex decoder
--------------
K learned query embeddings attend to the global backbone descriptor to
produce K candidate vertex positions and associated confidence scores.
This is a lightweight version of DETR's object decoder — one transformer
cross-attention layer followed by an MLP head.

Edge predictor
--------------
For each ordered pair (i, j) of the K vertex queries, predict:
  - edge existence probability  (scalar)
  - edge class logits           (n_edge_classes values)

The pair feature is: concat(v_feat_i, v_feat_j, global_feat, |pos_i - pos_j|).
We only predict the upper triangle at training time (edges are undirected).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mlp_head(dims: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(inplace=True)]
    layers += [nn.Linear(dims[-1], out_dim)]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Vertex decoder
# ---------------------------------------------------------------------------

class VertexDecoder(nn.Module):
    """
    K query embeddings → K vertex proposals.

    Each query cross-attends to per-point backbone features so each slot can
    focus on a specific region of the point cloud.  Vertex positions are
    predicted as an attention-weighted mean of input xyz (coarse anchor) plus
    a learned offset (fine refinement).

    Inputs
    ------
    global_feat  : (B, global_dim)   max-pooled global descriptor
    point_feats  : (B, N, point_dim) per-point features from backbone
    xyz          : (B, N, 3)         normalised point positions

    Outputs
    -------
    pos  : (B, K, 3)         predicted vertex positions
    conf : (B, K)            vertex-existence logit
    feat : (B, K, feat_dim)  per-vertex embedding for EdgePredictor
    """

    def __init__(self, n_queries: int = 64,
                 global_dim: int = 1024,
                 point_dim: int = 256,
                 hidden_dim: int = 256,
                 feat_dim: int = 128) -> None:
        super().__init__()
        self.n_queries = n_queries
        self.feat_dim = feat_dim

        # Learned query embeddings
        self.query_embed = nn.Embedding(n_queries, hidden_dim)

        # Cross-attention projections (queries → keys/values from point cloud)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(point_dim,  hidden_dim, bias=False)
        self.v_proj = nn.Linear(point_dim,  hidden_dim, bias=False)
        self._attn_scale = hidden_dim ** -0.5

        # Attended features + global → per-vertex feature
        self.feat_mlp = _mlp_head(
            [hidden_dim + global_dim, 512, 256],
            feat_dim,
        )

        # Position head: refine from attention-weighted xyz anchor
        self.pos_head = nn.Linear(feat_dim, 3)

        # Confidence head: feature → scalar logit
        self.conf_head = nn.Linear(feat_dim, 1)

    def forward(self,
                global_feat: torch.Tensor,   # (B, global_dim)
                point_feats: torch.Tensor,   # (B, N, point_dim)
                xyz:         torch.Tensor,   # (B, N, 3)
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = point_feats.shape
        K = self.n_queries

        # Learned query embeddings: (B, K, hidden_dim)
        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention: each query attends to all N points
        q_a = self.q_proj(q)                                        # (B, K, H)
        k_a = self.k_proj(point_feats)                              # (B, N, H)
        v_a = self.v_proj(point_feats)                              # (B, N, H)

        attn_logits = torch.bmm(q_a, k_a.transpose(1, 2)) * self._attn_scale  # (B, K, N)
        attn_w = F.softmax(attn_logits, dim=-1)                     # (B, K, N)

        attn_out  = torch.bmm(attn_w, v_a)                         # (B, K, H)
        pos_anchor = torch.bmm(attn_w, xyz)                         # (B, K, 3) coarse position

        # Per-vertex features: attended output + global context
        g = global_feat.unsqueeze(1).expand(-1, K, -1)             # (B, K, global_dim)
        x = torch.cat([attn_out, g], dim=-1)                        # (B, K, H+global_dim)

        feat = self.feat_mlp(x.view(B * K, -1)).view(B, K, self.feat_dim)

        # Fine-grained position offset added to coarse attention anchor
        pos  = pos_anchor + self.pos_head(feat)                     # (B, K, 3)
        conf = self.conf_head(feat).squeeze(-1)                     # (B, K)

        return pos, conf, feat


# ---------------------------------------------------------------------------
# Edge predictor
# ---------------------------------------------------------------------------

class EdgePredictor(nn.Module):
    """
    Predict edge existence + class for every pair of vertex proposals.

    Input
    -----
    feat        : (B, K, feat_dim)  per-vertex features from VertexDecoder
    pos         : (B, K, 3)         predicted vertex positions
    global_feat : (B, global_dim)

    Output
    ------
    edge_logits : (B, K, K, n_edge_classes + 1)
      Last class = "no edge".  Upper triangle is meaningful; lower triangle
      is set to the transposed upper triangle by the caller.
    """

    def __init__(self, feat_dim: int = 128,
                 global_dim: int = 1024,
                 n_edge_classes: int = 6,
                 hidden_dim: int = 256) -> None:
        super().__init__()
        # pair feature = feat_i + feat_j + global + dist + diff
        pair_in = feat_dim * 2 + global_dim + 1 + 3
        self.edge_mlp = _mlp_head(
            [pair_in, hidden_dim, hidden_dim],
            n_edge_classes + 1,   # last = "no edge"
        )

    def forward(self, feat: torch.Tensor,
                pos: torch.Tensor,
                global_feat: torch.Tensor) -> torch.Tensor:
        B, K, F = feat.shape

        # Broadcast to (B, K, K, feat_dim) pairs
        fi = feat.unsqueeze(2).expand(-1, -1, K, -1)   # (B, K, K, F)
        fj = feat.unsqueeze(1).expand(-1, K, -1, -1)

        # Geometric features
        pi = pos.unsqueeze(2).expand(-1, -1, K, -1)    # (B, K, K, 3)
        pj = pos.unsqueeze(1).expand(-1, K, -1, -1)
        diff = pi - pj                                  # (B, K, K, 3)
        dist = diff.norm(dim=-1, keepdim=True)          # (B, K, K, 1)

        # Global feat broadcast
        g = global_feat[:, None, None, :].expand(-1, K, K, -1)  # (B, K, K, G)

        pair = torch.cat([fi, fj, g, dist, diff], dim=-1)  # (B, K, K, pair_in)
        BKK = B * K * K
        logits = self.edge_mlp(pair.view(BKK, -1)).view(B, K, K, -1)

        # Symmetrise: average upper and lower triangle logits
        logits = (logits + logits.transpose(1, 2)) / 2.0

        return logits   # (B, K, K, n_edge_classes+1)
