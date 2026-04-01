from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ralagwm.data.batch import ChartState


class ChartStatePredictor(nn.Module):
    """Predict the chart generator state :math:`\zeta_t`.

    The predicted state contains the anchor action parameterization, metric
    matrix, boundary-saliency field, uncertainty field, and candidate action
    coordinates. Chart generation then uses only this state, matching the paper
    plan's ``\hat\zeta \rightarrow ChartGen`` deployment chain.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_dim: int = 1,
        coord_dim: int | None = None,
        metric_rank: int | None = None,
        pool_budget: int = 16,
        action_type: str = 'discrete',
        chart_mode: str = 'discrete',
        **_: object,
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        self.action_dim = max(int(action_dim), 1)
        self.coord_dim = int(coord_dim or (8 if action_type == 'discrete' else self.action_dim))
        self.metric_rank = int(metric_rank or self.coord_dim)
        self.pool_budget = int(pool_budget)
        self.action_type = str(action_type)
        self.chart_mode = str(chart_mode)

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.anchor_head = nn.Linear(hidden_dim, 1 if self.action_type == 'discrete' else self.action_dim)
        self.metric_head = nn.Linear(hidden_dim, self.coord_dim * self.coord_dim)
        self.saliency_head = nn.Linear(hidden_dim, self.pool_budget)
        self.uncertainty_head = nn.Linear(hidden_dim, self.pool_budget)
        self.coord_head = nn.Linear(hidden_dim, self.pool_budget * self.coord_dim)

    def forward(self, x: torch.Tensor) -> ChartState:
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
        h = self.backbone(x.float())
        bsz = h.shape[0]

        anchor_action = self.anchor_head(h)
        if self.action_type == 'continuous':
            anchor_action = torch.tanh(anchor_action)
        else:
            anchor_action = torch.sigmoid(anchor_action)

        metric_raw = self.metric_head(h).view(bsz, self.coord_dim, self.coord_dim)
        metric_matrix = metric_raw.transpose(-1, -2) @ metric_raw
        eye = torch.eye(self.coord_dim, device=h.device).unsqueeze(0).expand(bsz, -1, -1)
        metric_matrix = metric_matrix + 1e-3 * eye

        boundary_saliency = self.saliency_head(h)
        uncertainty = F.softplus(self.uncertainty_head(h)) + 1e-4
        action_coords = self.coord_head(h).view(bsz, self.pool_budget, self.coord_dim)
        if self.action_type != 'discrete':
            action_coords = torch.tanh(action_coords)
            anchor_embed = anchor_action
            if anchor_embed.shape[-1] < self.coord_dim:
                pad = torch.zeros(bsz, self.coord_dim - anchor_embed.shape[-1], device=h.device, dtype=anchor_embed.dtype)
                anchor_embed = torch.cat([anchor_embed, pad], dim=-1)
            anchor_embed = anchor_embed[:, : self.coord_dim]
            if self.pool_budget > 0:
                # Avoid in-place writes on the output of tanh, which would break autograd.
                action_coords = torch.cat([anchor_embed.unsqueeze(1), action_coords[:, 1:, :]], dim=1)

        return ChartState(
            anchor_action=anchor_action,
            metric_matrix=metric_matrix,
            boundary_saliency=boundary_saliency,
            uncertainty=uncertainty,
            action_coords=action_coords,
            metadata={
                'coord_dim': self.coord_dim,
                'metric_rank': self.metric_rank,
                'pool_budget': self.pool_budget,
                'action_dim': self.action_dim,
                'action_type': self.action_type,
                'chart_mode': self.chart_mode,
            },
        )
