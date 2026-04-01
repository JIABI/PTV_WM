from __future__ import annotations

import torch
import torch.nn as nn

from ralagwm.data.batch import BICChart, ChartState, RALAGGeometry


class GeometryDecoder(nn.Module):
    """Decode chart-local geometry from the bottleneck latent and predicted chart.

    This decoder predicts the score field on chart nodes and the edge-direction
    sensitivities on chart edges. Margin and boundary risk are derived from these
    primitives rather than treated as free, unrelated outputs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
        chart_budget: int,
        coord_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.chart_budget = int(chart_budget)
        self.coord_dim = int(coord_dim or max(action_dim, 1))

        self.context = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + self.coord_dim + self.coord_dim + 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + self.coord_dim + 3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def _node_inputs(self, z: torch.Tensor, chart_state: ChartState, chart: BICChart) -> torch.Tensor:
        coords = chart.coords.float()
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        batch_size, num_nodes, coord_dim = coords.shape
        action_mask = chart.metadata.get('action_mask')
        if action_mask is None:
            action_mask = torch.ones(batch_size, num_nodes, device=coords.device, dtype=torch.bool)
        action_mask_f = action_mask.float().unsqueeze(-1)
        anchor_idx = torch.argmax(chart_state.boundary_saliency, dim=-1)
        anchor_coords = torch.gather(
            chart_state.action_coords,
            1,
            anchor_idx[:, None, None].expand(-1, 1, chart_state.action_coords.shape[-1]),
        ).squeeze(1)
        rel = coords - anchor_coords.unsqueeze(1)
        sal = torch.gather(chart_state.boundary_saliency, 1, chart.actions.long().clamp(min=0, max=chart_state.boundary_saliency.shape[-1] - 1))
        unc = torch.gather(chart_state.uncertainty, 1, chart.actions.long().clamp(min=0, max=chart_state.uncertainty.shape[-1] - 1))
        context = self.context(z.float()).unsqueeze(1).expand(-1, num_nodes, -1)
        feats = torch.cat([context, coords, rel, sal.unsqueeze(-1), unc.unsqueeze(-1)], dim=-1)
        feats = feats * action_mask_f
        return feats

    def _edge_inputs(self, z: torch.Tensor, chart_state: ChartState, chart: BICChart) -> tuple[torch.Tensor, torch.Tensor]:
        edges = chart.edges
        if edges.dim() == 2:
            edges = edges.unsqueeze(0)
        batch_size, max_edges, _ = edges.shape
        edge_mask = chart.metadata.get('edge_mask')
        if edge_mask is None:
            edge_mask = torch.ones(batch_size, max_edges, device=edges.device, dtype=torch.bool)
        context = self.context(z.float()).unsqueeze(1).expand(-1, max_edges, -1)
        src = edges[..., 0].clamp(min=0, max=chart.coords.shape[1] - 1)
        dst = edges[..., 1].clamp(min=0, max=chart.coords.shape[1] - 1)
        src_coords = torch.gather(chart.coords, 1, src.unsqueeze(-1).expand(-1, -1, chart.coords.shape[-1]))
        dst_coords = torch.gather(chart.coords, 1, dst.unsqueeze(-1).expand(-1, -1, chart.coords.shape[-1]))
        rel = dst_coords - src_coords
        norm = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
        src_sal = torch.gather(chart_state.boundary_saliency, 1, torch.gather(chart.actions.long(), 1, src).clamp(min=0, max=chart_state.boundary_saliency.shape[-1] - 1))
        dst_sal = torch.gather(chart_state.boundary_saliency, 1, torch.gather(chart.actions.long(), 1, dst).clamp(min=0, max=chart_state.boundary_saliency.shape[-1] - 1))
        feats = torch.cat([context, rel, norm, (dst_sal - src_sal).unsqueeze(-1), torch.gather(chart.weights, 1, src).unsqueeze(-1)], dim=-1)
        feats = feats * edge_mask.float().unsqueeze(-1)
        return feats, edge_mask

    def forward(self, z: torch.Tensor, chart_state: ChartState, chart: BICChart) -> RALAGGeometry:
        if z.dim() > 2:
            z = z.view(z.shape[0], -1)
        node_inputs = self._node_inputs(z, chart_state, chart)
        raw_scores = self.node_mlp(node_inputs).squeeze(-1)
        action_mask = chart.metadata.get('action_mask')
        if action_mask is None:
            action_mask = torch.ones_like(raw_scores, dtype=torch.bool)
        masked_scores = raw_scores.masked_fill(~action_mask, 0.0)
        denom = action_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        centered_scores = masked_scores - masked_scores.sum(dim=-1, keepdim=True) / denom
        centered_scores = centered_scores * action_mask.float()

        masked_for_topk = centered_scores.masked_fill(~action_mask, -1e9)
        top_vals, top_idx = torch.topk(masked_for_topk, k=min(2, centered_scores.shape[-1]), dim=-1)
        margin = top_vals[:, 0] - top_vals[:, 1] if top_vals.shape[-1] >= 2 else torch.zeros(centered_scores.shape[0], device=centered_scores.device)

        edge_inputs, edge_mask = self._edge_inputs(z, chart_state, chart)
        raw_edge = self.edge_mlp(edge_inputs).squeeze(-1)
        raw_edge = raw_edge * edge_mask.float()
        edge_var = []
        for b in range(raw_edge.shape[0]):
            vals = raw_edge[b][edge_mask[b]]
            edge_var.append(vals.var(unbiased=False) if vals.numel() > 1 else torch.zeros((), device=raw_edge.device))
        edge_var = torch.stack(edge_var, dim=0)
        local_unc = torch.gather(
            chart_state.uncertainty,
            1,
            chart.actions.long().clamp(min=0, max=chart_state.uncertainty.shape[-1] - 1),
        )
        uncertainty_term = (local_unc * action_mask.float()).sum(dim=-1) / denom.squeeze(-1)
        boundary_risk = torch.sigmoid(-margin) + 0.20 * edge_var + 0.10 * uncertainty_term
        boundary_risk = boundary_risk.clamp(0.0, 1.0)

        metadata = {
            'action_mask': action_mask,
            'edge_mask': edge_mask,
            'chart_budget': self.chart_budget,
            'action_dim': self.action_dim,
        }
        return RALAGGeometry(
            centered_scores=centered_scores,
            margin=margin,
            edge_sensitivity=raw_edge,
            top_action_index=top_idx[:, 0],
            boundary_risk=boundary_risk,
            metadata=metadata,
        )
