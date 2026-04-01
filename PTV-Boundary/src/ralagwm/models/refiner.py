from __future__ import annotations

import torch
import torch.nn as nn

from ralagwm.data.batch import BICChart, ChartState, RALAGGeometry


class SelectiveBoundaryRefiner(nn.Module):
    """Selective boundary refinement on low-margin / high-risk states."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        chart_budget: int,
        coord_dim: int | None = None,
        hidden_dim: int | None = None,
        margin_threshold: float = 0.2,
        risk_threshold: float = 0.6,
        refine_scale: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        action_dim = int(action_dim)
        chart_budget = int(chart_budget)
        self.coord_dim = int(coord_dim or max(action_dim, 1))
        hidden_dim = int(hidden_dim) if hidden_dim is not None else input_dim
        self.margin_threshold = float(margin_threshold)
        self.risk_threshold = float(risk_threshold)
        self.refine_scale = float(refine_scale)

        self.context = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.node_delta = nn.Sequential(
            nn.Linear(hidden_dim + self.coord_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.edge_delta = nn.Sequential(
            nn.Linear(hidden_dim + self.coord_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gate_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        geom: RALAGGeometry,
        chart_state: ChartState,
        chart: BICChart,
    ) -> tuple[RALAGGeometry, torch.Tensor]:
        if features.dim() > 2:
            features = features.view(features.shape[0], -1)
        h = self.context(features.float())

        action_mask = geom.metadata.get('action_mask') if isinstance(geom.metadata, dict) else None
        if action_mask is None:
            action_mask = torch.ones_like(geom.centered_scores, dtype=torch.bool)
        edge_mask = geom.metadata.get('edge_mask') if isinstance(geom.metadata, dict) else None
        if edge_mask is None:
            edge_mask = torch.ones_like(geom.edge_sensitivity, dtype=torch.bool)

        low_margin = (geom.margin < self.margin_threshold).float()
        high_risk = (geom.boundary_risk > self.risk_threshold).float()
        learned_gate = torch.sigmoid(self.gate_head(h)).squeeze(-1)
        refine_gate = torch.maximum(torch.maximum(low_margin, high_risk), (learned_gate > 0.5).float())

        node_feats = torch.cat([
            h.unsqueeze(1).expand(-1, chart.coords.shape[1], -1),
            chart.coords.float(),
            geom.centered_scores.unsqueeze(-1),
            chart_state.uncertainty.gather(1, chart.actions.long().clamp(min=0, max=chart_state.uncertainty.shape[-1] - 1)).unsqueeze(-1),
        ], dim=-1)
        node_delta = self.node_delta(node_feats).squeeze(-1) * action_mask.float()

        edges = chart.edges.long().clamp(min=0, max=chart.coords.shape[1] - 1)
        src = edges[..., 0]
        dst = edges[..., 1]
        src_coords = torch.gather(chart.coords, 1, src.unsqueeze(-1).expand(-1, -1, chart.coords.shape[-1]))
        dst_coords = torch.gather(chart.coords, 1, dst.unsqueeze(-1).expand(-1, -1, chart.coords.shape[-1]))
        edge_feats = torch.cat([
            h.unsqueeze(1).expand(-1, edges.shape[1], -1),
            (dst_coords - src_coords).float(),
            geom.edge_sensitivity.unsqueeze(-1),
            edge_mask.float().unsqueeze(-1),
        ], dim=-1)
        edge_delta = self.edge_delta(edge_feats).squeeze(-1) * edge_mask.float()

        gate_node = refine_gate.unsqueeze(-1)
        refined_scores = geom.centered_scores + gate_node * self.refine_scale * node_delta
        refined_scores = refined_scores * action_mask.float()
        denom = action_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        refined_scores = refined_scores - refined_scores.sum(dim=-1, keepdim=True) / denom
        refined_scores = refined_scores * action_mask.float()

        refined_edge = geom.edge_sensitivity + gate_node * self.refine_scale * edge_delta
        refined_edge = refined_edge * edge_mask.float()
        masked_top = refined_scores.masked_fill(~action_mask, -1e9)
        top_vals, top_idx = torch.topk(masked_top, k=min(2, refined_scores.shape[-1]), dim=-1)
        refined_margin = top_vals[:, 0] - top_vals[:, 1] if top_vals.shape[-1] >= 2 else geom.margin

        edge_var = []
        for b in range(refined_edge.shape[0]):
            vals = refined_edge[b][edge_mask[b]]
            edge_var.append(vals.var(unbiased=False) if vals.numel() > 1 else torch.zeros((), device=refined_edge.device))
        edge_var = torch.stack(edge_var, dim=0)
        uncertainty_term = (chart_state.uncertainty.gather(1, chart.actions.long().clamp(min=0, max=chart_state.uncertainty.shape[-1] - 1)) * action_mask.float()).sum(dim=-1) / denom.squeeze(-1)
        refined_risk = torch.sigmoid(-refined_margin) + 0.20 * edge_var + 0.10 * uncertainty_term
        refined_risk = refined_risk.clamp(0.0, 1.0)

        refined_geom = RALAGGeometry(
            centered_scores=refined_scores,
            margin=refined_margin,
            edge_sensitivity=refined_edge,
            top_action_index=top_idx[:, 0],
            boundary_risk=refined_risk,
            metadata={**dict(geom.metadata), 'refined': True},
        )
        return refined_geom, refine_gate
