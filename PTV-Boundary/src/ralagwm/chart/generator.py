"""Shared chart generators with batched BIC selection and explicit masks."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ralagwm.chart.bic import BICObjective
from ralagwm.chart.features import compute_bic_features, compute_boundary_weights, resolve_anchor_index_and_coord
from ralagwm.chart.graph import anchor_edges, knn_edges, merge_edges
from ralagwm.chart.pool import PoolGenerator
from ralagwm.data.batch import BICChart, ChartState


@dataclass
class _SingleChart:
    actions: torch.Tensor
    coords: torch.Tensor
    edges: torch.Tensor
    weights: torch.Tensor
    info_matrix: torch.Tensor
    selected_indices: torch.Tensor
    action_mask: torch.Tensor
    edge_mask: torch.Tensor


class BaseChartGenerator:
    def generate(self, chart_state: ChartState, consensus: torch.Tensor, disagreement: torch.Tensor) -> BICChart:
        raise NotImplementedError


class BICChartGenerator(BaseChartGenerator):
    """Shared BIC chart generator for main model and all baselines.

    The generator operates on either a single state or a batched chart state and
    returns a padded batched ``BICChart`` with masks in ``metadata``.
    """

    def __init__(
        self,
        chart_budget: int = 8,
        feature_dim: int = 8,
        knn: int = 3,
        tau_delta: float = 0.25,
        pool_budget: int = 32,
        mode: str = "discrete",
    ):
        self.chart_budget = int(chart_budget)
        self.feature_dim = int(feature_dim)
        self.knn = int(knn)
        self.tau_delta = float(tau_delta)
        self.pool_generator = PoolGenerator(action_dim=pool_budget, pool_budget=pool_budget, mode=mode)
        self.objective = BICObjective(feature_dim=feature_dim)
        self.mode = str(mode)

    def _generate_single(self, chart_state: ChartState, consensus: torch.Tensor, disagreement: torch.Tensor) -> _SingleChart:
        pool = self.pool_generator.build(chart_state)
        actions = pool.actions.long()
        coords = pool.coords.float()
        if actions.numel() == 0:
            raise RuntimeError('PoolGenerator returned an empty action pool.')

        actions = actions.clamp(min=0, max=consensus.shape[0] - 1)
        psi = compute_bic_features(chart_state, actions, self.feature_dim)
        weights_all = compute_boundary_weights(consensus[actions], disagreement[actions], self.tau_delta)

        anchor_pool_index, _ = resolve_anchor_index_and_coord(chart_state)
        anchor_pool_index = min(anchor_pool_index, actions.shape[0] - 1)
        selected: list[int] = [anchor_pool_index]
        max_steps = min(self.chart_budget, int(actions.shape[0]))
        for _ in range(1, max_steps):
            candidates = [i for i in range(actions.shape[0]) if i not in selected]
            if not candidates:
                break
            gains = [self.objective.score(psi, weights_all, selected + [cand]) for cand in candidates]
            best = candidates[int(torch.stack(gains).argmax().item())]
            selected.append(best)

        selected_idx = torch.tensor(selected, device=consensus.device, dtype=torch.long)
        selected_actions = actions[selected_idx]
        selected_coords = coords[selected_idx]
        selected_weights = weights_all[selected_idx]
        edges = merge_edges(
            knn_edges(selected_coords, self.knn),
            anchor_edges(selected_coords.shape[0], anchor_index=0, device=selected_coords.device),
        )
        info = self.objective.information_matrix(psi, weights_all, selected)

        action_mask = torch.zeros(self.chart_budget, device=consensus.device, dtype=torch.bool)
        action_mask[: selected_idx.shape[0]] = True
        pad_nodes = self.chart_budget - selected_idx.shape[0]
        if pad_nodes > 0:
            selected_actions = torch.cat([selected_actions, selected_actions.new_zeros(pad_nodes)], dim=0)
            selected_coords = torch.cat([selected_coords, selected_coords.new_zeros(pad_nodes, selected_coords.shape[-1])], dim=0)
            selected_weights = torch.cat([selected_weights, selected_weights.new_zeros(pad_nodes)], dim=0)
            selected_idx = torch.cat([selected_idx, selected_idx.new_zeros(pad_nodes)], dim=0)

        max_edges = self.chart_budget * max(self.knn, 1) + max(self.chart_budget - 1, 0)
        edge_mask = torch.zeros(max_edges, device=consensus.device, dtype=torch.bool)
        if edges.numel() == 0:
            padded_edges = torch.zeros(max_edges, 2, device=consensus.device, dtype=torch.long)
        else:
            edge_count = min(edges.shape[0], max_edges)
            edge_mask[:edge_count] = True
            padded_edges = torch.zeros(max_edges, 2, device=consensus.device, dtype=torch.long)
            padded_edges[:edge_count] = edges[:edge_count]

        return _SingleChart(
            actions=selected_actions,
            coords=selected_coords,
            edges=padded_edges,
            weights=selected_weights,
            info_matrix=info,
            selected_indices=selected_idx,
            action_mask=action_mask,
            edge_mask=edge_mask,
        )

    def generate(self, chart_state: ChartState, consensus: torch.Tensor, disagreement: torch.Tensor) -> BICChart:
        if consensus.dim() == 1:
            consensus = consensus.unsqueeze(0)
        if disagreement.dim() == 1:
            disagreement = disagreement.unsqueeze(0)

        state_is_batched = chart_state.boundary_saliency.dim() > 1
        if not state_is_batched:
            chart_state = ChartState(
                anchor_action=chart_state.anchor_action.view(1, 1) if chart_state.anchor_action.dim() == 0 else (chart_state.anchor_action.unsqueeze(0) if chart_state.anchor_action.dim() == 1 else chart_state.anchor_action),
                metric_matrix=chart_state.metric_matrix.unsqueeze(0) if chart_state.metric_matrix.dim() == 2 else chart_state.metric_matrix,
                boundary_saliency=chart_state.boundary_saliency.unsqueeze(0),
                uncertainty=chart_state.uncertainty.unsqueeze(0),
                action_coords=chart_state.action_coords.unsqueeze(0),
                metadata=dict(chart_state.metadata),
            )

        batch_size = consensus.shape[0]
        single_charts: list[_SingleChart] = []
        for b in range(batch_size):
            single_state = ChartState(
                anchor_action=chart_state.anchor_action[b],
                metric_matrix=chart_state.metric_matrix[b],
                boundary_saliency=chart_state.boundary_saliency[b],
                uncertainty=chart_state.uncertainty[b],
                action_coords=chart_state.action_coords[b],
                metadata=dict(chart_state.metadata),
            )
            single_charts.append(self._generate_single(single_state, consensus[b], disagreement[b]))

        actions = torch.stack([c.actions for c in single_charts], dim=0)
        coords = torch.stack([c.coords for c in single_charts], dim=0)
        edges = torch.stack([c.edges for c in single_charts], dim=0)
        weights = torch.stack([c.weights for c in single_charts], dim=0)
        info = torch.stack([c.info_matrix for c in single_charts], dim=0)
        selected = torch.stack([c.selected_indices for c in single_charts], dim=0)
        action_mask = torch.stack([c.action_mask for c in single_charts], dim=0)
        edge_mask = torch.stack([c.edge_mask for c in single_charts], dim=0)
        metadata = {
            'action_mask': action_mask,
            'edge_mask': edge_mask,
            'chart_budget': self.chart_budget,
            'knn': self.knn,
            'mode': self.mode,
        }
        out = BICChart(
            actions=actions,
            coords=coords,
            edges=edges,
            weights=weights,
            info_matrix=info,
            selected_indices=selected,
            metadata=metadata,
        )
        if not state_is_batched:
            out.actions = out.actions[0]
            out.coords = out.coords[0]
            out.edges = out.edges[0]
            out.weights = out.weights[0]
            out.info_matrix = out.info_matrix[0]
            out.selected_indices = out.selected_indices[0]
            out.metadata = {
                **metadata,
                'action_mask': action_mask[0],
                'edge_mask': edge_mask[0],
            }
        return out
