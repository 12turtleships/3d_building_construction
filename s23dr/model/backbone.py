"""
PointNet backbone for per-point feature extraction and global pooling.

Processes an (B, N, C_in) point cloud through shared MLPs, returns both
per-point features and a global descriptor.

Architecture follows the original PointNet encoder:
  per-point MLP → (64, 128, 256) → global max-pool → (1024,)

We skip the T-Net (input/feature transform) for simplicity; it rarely helps
on normalised point clouds where the spatial extent is already canonical.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(channels: list[int], bn: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(channels) - 1):
        layers.append(nn.Linear(channels[i], channels[i + 1]))
        if bn:
            layers.append(nn.BatchNorm1d(channels[i + 1]))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class PointNetBackbone(nn.Module):
    """
    Shared-MLP PointNet encoder.

    Input
    -----
    xyz      : (B, N, 3)   normalised coordinates
    feats    : (B, N, C)   per-point features (e.g. vote_frac, class_id)

    Output
    ------
    global_feat   : (B, global_dim)    max-pooled global descriptor
    point_feat    : (B, N, point_dim)  per-point features before pooling
    """

    def __init__(self, in_channels: int = 4,
                 point_dims: list[int] | None = None,
                 global_dims: list[int] | None = None) -> None:
        super().__init__()
        point_dims  = point_dims  or [64, 128, 256]
        global_dims = global_dims or [512, 1024]

        # Per-point branch: shared MLP applied independently to each point
        self._point_mlp = _mlp([in_channels] + point_dims)
        self.point_dim = point_dims[-1]

        # Global branch: MLP on the concatenated per-point output
        self._global_mlp = _mlp([point_dims[-1]] + global_dims)
        self.global_dim = global_dims[-1]

    def forward(self, xyz: torch.Tensor,
                feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # xyz:   (B, N, 3)
        # feats: (B, N, C)
        x = torch.cat([xyz, feats], dim=-1)          # (B, N, 3+C)
        B, N, _ = x.shape

        # Apply shared MLP: flatten batch×points, apply, reshape back
        x = x.view(B * N, -1)
        for layer in self._point_mlp:
            if isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
            else:
                x = layer(x)
        x = x.view(B, N, self.point_dim)             # (B, N, point_dim)
        point_feat = x

        # Global branch
        x_flat = x.view(B * N, self.point_dim)
        for layer in self._global_mlp:
            if isinstance(layer, nn.BatchNorm1d):
                x_flat = layer(x_flat)
            else:
                x_flat = layer(x_flat)
        x_flat = x_flat.view(B, N, self.global_dim)
        global_feat = x_flat.max(dim=1).values        # (B, global_dim) — max pool

        return global_feat, point_feat
