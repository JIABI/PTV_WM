"""Manual chart used for ablation against BIC-Chart."""
from __future__ import annotations

import torch

from ralagwm.chart.graph import anchor_edges, knn_edges, merge_edges
from ralagwm.typing import BICChart, ChartState


class ManualChartGenerator:
    def __init__(self, chart_budget: int = 8, knn: int = 3):
        self.chart_budget = chart_budget
        self.knn = knn

    def generate(self, chart_state: ChartState, consensus: torch.Tensor, disagreement: torch.Tensor) -> BICChart:
        del disagreement
        num = min(self.chart_budget, consensus.shape[0])
        top_idx = torch.topk(consensus, k=num).indices
        coords = chart_state.action_coords[top_idx]
        edges = merge_edges(knn_edges(coords, self.knn), anchor_edges(coords.shape[0], anchor_index=0, device=coords.device))
        info_matrix = torch.eye(coords.shape[-1], device=coords.device)
        weights = torch.ones(num, device=coords.device)
        return BICChart(
            actions=top_idx,
            coords=coords,
            edges=edges,
            weights=weights,
            info_matrix=info_matrix,
            selected_indices=top_idx,
        )
