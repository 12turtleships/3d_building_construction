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

    Each query is concatenated with the global backbone feature and passed
    through an MLP to produce:
      pos  : (B, K, 3)  — predicted vertex positions
      conf : (B, K)     — logit for vertex existence (whether the slot is active)
      feat : (B, K, feat_dim)  — per-vertex embedding for the edge predictor
    """

    def __init__(self, n_queries: int = 64,
                 global_dim: int = 1024,
                 hidden_dim: int = 256,
                 feat_dim: int = 128) -> None:
        super().__init__()
        self.n_queries = n_queries
        self.feat_dim = feat_dim

        # Learned query embeddings
        self.query_embed = nn.Embedding(n_queries, hidden_dim)

        # Query + global → per-vertex feature
        self.feat_mlp = _mlp_head(
            [hidden_dim + global_dim, 512, 256],
            feat_dim,
        )

        # Position head: feature → 3D position (sigmoid → [0,1], then rescale)
        self.pos_head = nn.Linear(feat_dim, 3)

        # Confidence head: feature → scalar logit
        self.conf_head = nn.Linear(feat_dim, 1)

    def forward(self, global_feat: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = global_feat.shape[0]
        K = self.n_queries

        # (K, hidden_dim) → expand to (B, K, hidden_dim)
        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Concat with global feat broadcast over K
        g = global_feat.unsqueeze(1).expand(-1, K, -1)   # (B, K, global_dim)
        x = torch.cat([q, g], dim=-1)                     # (B, K, hidden+global)

        # Flatten, MLP, reshape
        x = x.view(B * K, -1)
        feat = self.feat_mlp(x).view(B, K, self.feat_dim)  # (B, K, feat_dim)

        # Heads — no activation on pos so the model can predict any coordinate sign
        pos  = self.pos_head(feat)                          # (B, K, 3)
        conf = self.conf_head(feat).squeeze(-1)             # (B, K)

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
