"""Candidate action pool generation for BIC-Chart."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ralagwm.typing import ChartState


@dataclass
class PoolOutput:
    actions: torch.Tensor
    coords: torch.Tensor


class PoolGenerator:
    """Shared candidate pool generator.

    All object-level baselines must reuse this pool/chart interface so that
    performance differences are attributable to learned objects rather than
    private candidate-selection advantages.
    """

    def __init__(self, action_dim: int, pool_budget: int = 32, mode: str = "discrete"):
        self.action_dim = action_dim
        self.pool_budget = pool_budget
        self.mode = mode

    def discrete_action_pool(self, num_actions: int, chart_state: ChartState) -> PoolOutput:
        num = min(num_actions, chart_state.action_coords.shape[0])
        actions = torch.arange(num, device=chart_state.action_coords.device)
        return PoolOutput(actions=actions, coords=chart_state.action_coords[:num])

    def continuous_lowdim_pool(self, chart_state: ChartState) -> PoolOutput:
        num = min(self.pool_budget, chart_state.action_coords.shape[0])
        actions = torch.arange(num, device=chart_state.action_coords.device)
        jitter = 0.05 * torch.randn_like(chart_state.action_coords[:num])
        return PoolOutput(actions=actions, coords=chart_state.action_coords[:num] + jitter)

    def highdim_path_pool(self, chart_state: ChartState) -> PoolOutput:
        """Deterministic sparse-path placeholder.

        This scaffold version approximates path proposals by selecting the first
        ``pool_budget`` coordinates and adding low-rank perturbations. The API is
        intentionally stable so a stronger path generator can be dropped in
        later without changing downstream modules.
        """
        num = min(self.pool_budget, chart_state.action_coords.shape[0])
        actions = torch.arange(num, device=chart_state.action_coords.device)
        coords = chart_state.action_coords[:num]
        if coords.shape[0] > 1:
            low_rank = coords.mean(dim=0, keepdim=True)
            coords = coords + 0.02 * (coords - low_rank)
        return PoolOutput(actions=actions, coords=coords)

    def build(self, chart_state: ChartState) -> PoolOutput:
        if self.mode == "continuous":
            return self.continuous_lowdim_pool(chart_state)
        if self.mode == "highdim_continuous":
            return self.highdim_path_pool(chart_state)
        return self.discrete_action_pool(self.action_dim, chart_state)
