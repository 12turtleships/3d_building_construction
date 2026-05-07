"""
Hungarian-matching wireframe loss.

Matching strategy (two-stage)
------------------------------
1. Vertex matching: use the scipy linear_sum_assignment solver to find the
   minimum-cost bijection between K predicted vertex proposals and V GT
   vertices (V ≤ K).  Unmatched slots are treated as "inactive".

   Cost matrix (K × V):  L1 distance between predicted and GT positions,
   weighted by predicted confidence (higher confidence → higher penalty for
   mis-assignment so the model is encouraged to be precise when confident).

2. Edge matching: once vertices are matched, GT edge (u, v) is re-indexed to
   the matched predicted slots (û, v̂).  Edge logits at (û, v̂) are supervised
   with the GT edge class.  All other (i, j) pairs are supervised as "no edge".

Losses
------
  L_vert   : L1 on matched vertex positions  (active slots only)
  L_conf   : BCE between predicted confidence and match indicator
  L_edge   : cross-entropy on all K×K pairs
              (class = actual class for matched edges, n_edge_classes for rest)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class WireframeLoss(nn.Module):
    def __init__(self,
                 w_vert: float = 5.0,
                 w_conf: float = 1.0,
                 w_edge: float = 2.0,
                 n_edge_classes: int = 10,
                 no_edge_weight: float = 0.05) -> None:
        super().__init__()
        self.w_vert = w_vert
        self.w_conf = w_conf
        self.w_edge = w_edge
        self.n_edge_classes = n_edge_classes
        self.no_edge_class = n_edge_classes   # last class = "no edge"

        # Down-weight the "no edge" class to counter the ~100:1 imbalance
        # between negative pairs (K=64 → ~4096 pairs) and actual edges (~20-30).
        edge_weight = torch.ones(n_edge_classes + 1)
        edge_weight[-1] = no_edge_weight
        self.register_buffer("_edge_weight", edge_weight)

    # ------------------------------------------------------------------
    def forward(self,
                pred_pos:    torch.Tensor,   # (B, K, 3)
                pred_conf:   torch.Tensor,   # (B, K)
                edge_logits: torch.Tensor,   # (B, K, K, n_cls+1)
                gt_verts:    list[torch.Tensor],  # list of (V_b, 3)
                gt_edges:    list[torch.Tensor],  # list of (E_b, 2)
                gt_classes:  list[torch.Tensor],  # list of (E_b,)
                ) -> dict[str, torch.Tensor]:

        B = pred_pos.shape[0]
        total_vert = pred_pos.new_zeros(())
        total_conf = pred_pos.new_zeros(())
        total_edge = pred_pos.new_zeros(())

        for b in range(B):
            ppos = pred_pos[b]     # (K, 3)
            pconf = pred_conf[b]   # (K,)
            elogits = edge_logits[b]  # (K, K, n_cls+1)

            gv = gt_verts[b].to(ppos.device)    # (V, 3)
            ge = gt_edges[b].to(ppos.device)    # (E, 2)
            gc = gt_classes[b].to(ppos.device)  # (E,)

            V = gv.shape[0]
            K = ppos.shape[0]

            # ---- 1. vertex matching ---------------------------------
            with torch.no_grad():
                cost = torch.cdist(ppos, gv, p=1)   # (K, V)  L1
                cost_np = cost.cpu().numpy()
                row_idx, col_idx = linear_sum_assignment(cost_np)
                # row_idx: matched prediction indices (length ≤ K)
                # col_idx: matched GT indices

            matched_pred = torch.tensor(row_idx, device=ppos.device)
            matched_gt   = torch.tensor(col_idx, device=ppos.device)

            # ---- 2. vertex position loss ---------------------------
            l_vert = F.l1_loss(ppos[matched_pred], gv[matched_gt])
            total_vert = total_vert + l_vert

            # ---- 3. confidence loss --------------------------------
            # Positive = matched predictions; negative = all unmatched
            conf_target = torch.zeros(K, device=ppos.device)
            conf_target[matched_pred] = 1.0
            l_conf = F.binary_cross_entropy_with_logits(pconf, conf_target)
            total_conf = total_conf + l_conf

            # ---- 4. edge loss --------------------------------------
            # Build target edge class matrix (K, K) initialised to "no edge"
            edge_target = torch.full((K, K), self.no_edge_class,
                                     dtype=torch.long, device=ppos.device)

            if len(ge) > 0:
                # Map GT vertex indices → matched prediction indices
                # gt_v_to_pred[gt_idx] = pred_idx (only for matched GT vertices)
                gt_v_to_pred = torch.full((V,), -1, dtype=torch.long,
                                          device=ppos.device)
                gt_v_to_pred[matched_gt] = matched_pred

                # Only include GT edges where both endpoints are matched
                eu = ge[:, 0]
                ev = ge[:, 1]
                pu = gt_v_to_pred[eu]
                pv = gt_v_to_pred[ev]
                valid = (pu >= 0) & (pv >= 0)

                if valid.any():
                    pu_v = pu[valid]
                    pv_v = pv[valid]
                    ec_v = gc[valid]
                    edge_target[pu_v, pv_v] = ec_v
                    edge_target[pv_v, pu_v] = ec_v   # symmetric

            # Cross-entropy on all K×K pairs (weighted to counter no-edge dominance)
            logits_flat = elogits.view(K * K, -1)
            target_flat = edge_target.view(K * K)
            l_edge = F.cross_entropy(
                logits_flat, target_flat,
                weight=self._edge_weight.to(logits_flat.device),
            )
            total_edge = total_edge + l_edge

        loss = (self.w_vert * total_vert / B
                + self.w_conf * total_conf / B
                + self.w_edge * total_edge / B)

        return {
            "loss":       loss,
            "loss_vert":  total_vert / B,
            "loss_conf":  total_conf / B,
            "loss_edge":  total_edge / B,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode(self,
               pred_pos:    torch.Tensor,   # (K, 3)  — single sample
               pred_conf:   torch.Tensor,   # (K,)
               edge_logits: torch.Tensor,   # (K, K, n_cls+1)
               conf_thresh: float = 0.5,
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert raw predictions to a wireframe graph.

        Returns
        -------
        vertices     : (V', 3)
        edges        : (E', 2) indices into vertices
        edge_classes : (E',)
        """
        # Active vertices: predicted confidence > threshold
        active = torch.sigmoid(pred_conf) > conf_thresh
        if not active.any():
            # Fall back to top-K by confidence
            active = pred_conf.topk(min(8, len(pred_conf))).indices
            mask = torch.zeros_like(pred_conf, dtype=torch.bool)
            mask[active] = True
            active = mask

        verts = pred_pos[active]           # (V', 3)
        active_idx = active.nonzero(as_tuple=True)[0]
        V = len(active_idx)

        # Edge prediction among active vertices
        sub_logits = edge_logits[active_idx][:, active_idx]  # (V', V', n_cls+1)
        edge_cls = sub_logits.argmax(dim=-1)   # (V', V')

        # Collect upper-triangle edges that are not "no edge"
        edge_list, cls_list = [], []
        for i in range(V):
            for j in range(i + 1, V):
                c = int(edge_cls[i, j])
                if c != self.no_edge_class:
                    edge_list.append([i, j])
                    cls_list.append(c)

        if not edge_list:
            return (verts,
                    torch.zeros((0, 2), dtype=torch.long, device=verts.device),
                    torch.zeros((0,),   dtype=torch.long, device=verts.device))

        edges   = torch.tensor(edge_list, dtype=torch.long, device=verts.device)
        classes = torch.tensor(cls_list,  dtype=torch.long, device=verts.device)
        return verts, edges, classes
