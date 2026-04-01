"""PTV-Boundary geometry extraction from consensus scores and a chart."""
from __future__ import annotations

import torch

from ralagwm.data.batch import BICChart, RALAGGeometry


def _extract_single(consensus_scores: torch.Tensor, chart: BICChart, disagreement: torch.Tensor | None = None) -> RALAGGeometry:
    action_mask = chart.metadata.get('action_mask') if isinstance(chart.metadata, dict) else None
    edge_mask = chart.metadata.get('edge_mask') if isinstance(chart.metadata, dict) else None
    if action_mask is None:
        action_mask = torch.ones(chart.actions.shape[0], device=chart.actions.device, dtype=torch.bool)
    local_actions = chart.actions.long().clamp(min=0, max=consensus_scores.shape[0] - 1)
    local_scores = consensus_scores[local_actions]
    local_scores = local_scores * action_mask.float()
    denom = action_mask.float().sum().clamp_min(1.0)
    centered = local_scores - (local_scores.sum() / denom)
    centered = centered * action_mask.float()

    masked_scores = local_scores.clone()
    masked_scores[~action_mask] = -1e9
    top_vals, top_idx = torch.topk(masked_scores, k=min(2, masked_scores.shape[0]))
    margin = top_vals[0] - top_vals[1] if top_vals.numel() > 1 else torch.zeros((), device=local_scores.device)

    if chart.edges.numel() > 0:
        valid_edges = chart.edges[edge_mask] if edge_mask is not None else chart.edges
        if valid_edges.numel() > 0:
            src = valid_edges[:, 0]
            dst = valid_edges[:, 1]
            rel = chart.coords[dst] - chart.coords[src]
            metric = chart.info_matrix
            if metric.dim() == 3:
                metric = metric[0]
            gram = rel @ metric[: rel.shape[-1], : rel.shape[-1]]
            norm = torch.sqrt((gram * rel).sum(dim=-1).clamp_min(1e-6))
            edge_sens = (centered[dst] - centered[src]) / norm
        else:
            edge_sens = torch.zeros(0, device=local_scores.device, dtype=local_scores.dtype)
    else:
        edge_sens = torch.zeros(0, device=local_scores.device, dtype=local_scores.dtype)

    edge_var = edge_sens.var(unbiased=False) if edge_sens.numel() > 1 else torch.zeros((), device=local_scores.device)
    if disagreement is None:
        uncertainty_term = torch.zeros((), device=local_scores.device)
    else:
        uncertainty_term = disagreement[local_actions][action_mask].mean() if action_mask.any() else disagreement.mean()
    boundary_risk = torch.sigmoid(-margin) + 0.25 * edge_var + 0.10 * uncertainty_term
    boundary_risk = boundary_risk.clamp(0.0, 1.0)
    return RALAGGeometry(
        centered_scores=centered,
        margin=margin.unsqueeze(0),
        edge_sensitivity=edge_sens,
        top_action_index=top_idx[:1],
        boundary_risk=boundary_risk.unsqueeze(0),
        metadata={'action_mask': action_mask, 'edge_mask': edge_mask},
    )


def extract_ralag_geometry(
    consensus_scores: torch.Tensor,
    chart: BICChart,
    disagreement: torch.Tensor | None = None,
) -> RALAGGeometry:
    """Extract chart-local PTV-Boundary geometry.

    Supports both single charts and padded batched charts.
    """
    if consensus_scores.dim() == 1:
        return _extract_single(consensus_scores, chart, disagreement)

    outputs = []
    for b in range(consensus_scores.shape[0]):
        single_chart = BICChart(
            actions=chart.actions[b],
            coords=chart.coords[b],
            edges=chart.edges[b],
            weights=chart.weights[b],
            info_matrix=chart.info_matrix[b],
            selected_indices=chart.selected_indices[b],
            metadata={
                'action_mask': chart.metadata.get('action_mask', None)[b] if isinstance(chart.metadata.get('action_mask', None), torch.Tensor) else None,
                'edge_mask': chart.metadata.get('edge_mask', None)[b] if isinstance(chart.metadata.get('edge_mask', None), torch.Tensor) else None,
            },
        )
        dis = disagreement[b] if disagreement is not None and disagreement.dim() > 1 else disagreement
        outputs.append(_extract_single(consensus_scores[b], single_chart, dis))
    centered = torch.stack([o.centered_scores for o in outputs], dim=0)
    margin = torch.cat([o.margin for o in outputs], dim=0)
    max_edges = max(int(o.edge_sensitivity.numel()) for o in outputs) if outputs else 0
    edge_stack = []
    edge_mask = []
    for o in outputs:
        e = o.edge_sensitivity.view(-1)
        m = torch.zeros(max_edges, device=e.device, dtype=torch.bool)
        if e.numel() < max_edges:
            pad = torch.zeros(max_edges - e.numel(), device=e.device, dtype=e.dtype)
            e = torch.cat([e, pad], dim=0)
        m[: o.edge_sensitivity.numel()] = True
        edge_stack.append(e)
        edge_mask.append(m)
    top_idx = torch.cat([o.top_action_index for o in outputs], dim=0)
    risk = torch.cat([o.boundary_risk for o in outputs], dim=0)
    return RALAGGeometry(
        centered_scores=centered,
        margin=margin,
        edge_sensitivity=torch.stack(edge_stack, dim=0) if edge_stack else torch.zeros(0),
        top_action_index=top_idx,
        boundary_risk=risk,
        metadata={'edge_mask': torch.stack(edge_mask, dim=0) if edge_mask else None},
    )
