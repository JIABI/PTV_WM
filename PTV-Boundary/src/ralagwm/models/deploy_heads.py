from __future__ import annotations

import torch
import torch.nn as nn

from ralagwm.data.batch import BICChart, ChartState, RALAGGeometry


class _BaseDeployHead(nn.Module):
    def __init__(self, hidden_dim: int, chart_budget: int, coord_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.chart_budget = int(chart_budget)
        self.coord_dim = int(coord_dim)

    def _node_features(
        self,
        features: torch.Tensor,
        geom: RALAGGeometry,
        chart: BICChart,
        chart_state: ChartState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.dim() > 2:
            features = features.view(features.shape[0], -1)
        action_mask = chart.metadata.get('action_mask') if isinstance(chart.metadata, dict) else None
        if action_mask is None:
            action_mask = torch.ones(features.shape[0], chart.coords.shape[1], device=features.device, dtype=torch.bool)
        local_unc = torch.gather(
            chart_state.uncertainty,
            1,
            chart.actions.long().clamp(min=0, max=chart_state.uncertainty.shape[-1] - 1),
        )
        feats = torch.cat([
            features.unsqueeze(1).expand(-1, chart.coords.shape[1], -1),
            chart.coords.float(),
            geom.centered_scores.unsqueeze(-1),
            local_unc.unsqueeze(-1),
            geom.boundary_risk[:, None, None].expand(-1, chart.coords.shape[1], 1),
        ], dim=-1)
        return feats, action_mask


class LinearDeployHead(_BaseDeployHead):
    def __init__(self, hidden_dim: int, action_dim: int, chart_budget: int, coord_dim: int) -> None:
        super().__init__(hidden_dim, chart_budget, coord_dim)
        self.node_proj = nn.Linear(int(hidden_dim) + int(coord_dim) + 3, 1)

    def forward(self, features: torch.Tensor, geom: RALAGGeometry, chart: BICChart, chart_state: ChartState) -> torch.Tensor:
        feats, mask = self._node_features(features, geom, chart, chart_state)
        logits = self.node_proj(feats).squeeze(-1) + geom.centered_scores.float()
        return logits.masked_fill(~mask, -1e9)


class MLPDeployHead(_BaseDeployHead):
    def __init__(self, hidden_dim: int, action_dim: int, chart_budget: int, coord_dim: int) -> None:
        super().__init__(hidden_dim, chart_budget, coord_dim)
        hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + int(coord_dim) + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, geom: RALAGGeometry, chart: BICChart, chart_state: ChartState) -> torch.Tensor:
        feats, mask = self._node_features(features, geom, chart, chart_state)
        logits = self.net(feats).squeeze(-1) + geom.centered_scores.float()
        logits = logits - 0.25 * chart_state.uncertainty.gather(1, chart.actions.long().clamp(min=0, max=chart_state.uncertainty.shape[-1] - 1))
        return logits.masked_fill(~mask, -1e9)


class PlannerDeployHead(_BaseDeployHead):
    def __init__(self, hidden_dim: int, action_dim: int, chart_budget: int, coord_dim: int) -> None:
        super().__init__(hidden_dim, chart_budget, coord_dim)
        self.bias = nn.Linear(int(hidden_dim), 1)

    def forward(self, features: torch.Tensor, geom: RALAGGeometry, chart: BICChart, chart_state: ChartState) -> torch.Tensor:
        feats, mask = self._node_features(features, geom, chart, chart_state)
        base = self.bias(features.float())
        uncertainty_penalty = chart_state.uncertainty.gather(1, chart.actions.long().clamp(min=0, max=chart_state.uncertainty.shape[-1] - 1))
        logits = base + geom.centered_scores.float() - 0.5 * uncertainty_penalty - 0.25 * geom.boundary_risk.unsqueeze(-1)
        return logits.masked_fill(~mask, -1e9)


LightweightPlannerHead = PlannerDeployHead


def build_deploy_head(kind: str, hidden_dim: int, action_dim: int, chart_budget: int, coord_dim: int = 8) -> nn.Module:
    kind = str(kind).lower()
    if kind == 'linear':
        return LinearDeployHead(hidden_dim, action_dim, chart_budget, coord_dim)
    if kind == 'mlp':
        return MLPDeployHead(hidden_dim, action_dim, chart_budget, coord_dim)
    if kind in {'planner', 'shallow_planner'}:
        return PlannerDeployHead(hidden_dim, action_dim, chart_budget, coord_dim)
    raise ValueError(f'Unknown deploy head kind: {kind}')
