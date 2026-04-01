"""Feature construction for BIC-Chart."""
from __future__ import annotations

import torch

from ralagwm.typing import ChartState


def resolve_anchor_index_and_coord(chart_state: ChartState) -> tuple[int, torch.Tensor]:
    """Resolve the chart anchor into a pool index and pool coordinate.

    Discrete chart states encode the anchor as a normalized scalar in [0, 1].
    Continuous chart states encode the anchor as an action vector. This helper
    converts both representations into the shared candidate-pool coordinate
    system used by BIC-Chart.
    """
    coords = chart_state.action_coords
    if coords.dim() != 2:
        raise ValueError(f'Expected unbatched action coordinates, got shape {tuple(coords.shape)}')
    num_coords = int(coords.shape[0])
    if num_coords <= 0:
        raise ValueError('Chart state has no action coordinates.')

    anchor = chart_state.anchor_action
    action_type = str(getattr(chart_state, 'metadata', {}).get('action_type', '')).lower()

    if anchor.numel() <= 1:
        scalar = float(anchor.view(-1)[0].item())
        if action_type == 'discrete' or (0.0 <= scalar <= 1.0 + 1e-6):
            anchor_idx = int(round(scalar * max(num_coords - 1, 1)))
        else:
            anchor_idx = int(round(scalar))
        anchor_idx = max(0, min(anchor_idx, num_coords - 1))
        return anchor_idx, coords[anchor_idx]

    anchor_vec = anchor.view(-1).to(coords.device, dtype=coords.dtype)
    coord_dim = int(coords.shape[-1])
    if anchor_vec.numel() < coord_dim:
        pad = torch.zeros(coord_dim - anchor_vec.numel(), device=coords.device, dtype=coords.dtype)
        anchor_vec = torch.cat([anchor_vec, pad], dim=0)
    anchor_vec = anchor_vec[:coord_dim]
    distances = (coords - anchor_vec.unsqueeze(0)).pow(2).sum(dim=-1)
    anchor_idx = int(torch.argmin(distances).item())
    return anchor_idx, coords[anchor_idx]


def compute_bic_features(chart_state: ChartState, actions: torch.Tensor, feature_dim: int = 8) -> torch.Tensor:
    """Compute local geometry features :math:`\psi_t(a)`.

    The feature concatenates a normalized relative direction with a scalar
    boundary-position feature derived from boundary saliency.
    """
    coords = chart_state.action_coords
    anchor_idx, anchor_coord = resolve_anchor_index_and_coord(chart_state)
    selected = coords[actions.long()]
    rel = selected - anchor_coord.unsqueeze(0)

    metric = chart_state.metric_matrix
    if metric.dim() == 2:
        metric_diag = torch.diag(metric)
    else:
        metric_diag = torch.ones(selected.shape[-1], device=selected.device)
    metric_diag = metric_diag[: selected.shape[-1]]
    norm = torch.sqrt((rel.pow(2) * metric_diag.unsqueeze(0)).sum(dim=-1, keepdim=True) + 1e-6)
    direction = rel / norm.clamp_min(1e-6)

    anchor_saliency = chart_state.boundary_saliency[anchor_idx]
    rel_gap = (chart_state.boundary_saliency[actions.long()] - anchor_saliency).unsqueeze(-1)
    psi = torch.cat([direction, rel_gap], dim=-1)
    if psi.shape[-1] < feature_dim:
        pad = torch.zeros(psi.shape[0], feature_dim - psi.shape[-1], device=psi.device, dtype=psi.dtype)
        psi = torch.cat([psi, pad], dim=-1)
    return psi[:, :feature_dim]


def compute_boundary_weights(consensus_scores: torch.Tensor, disagreement: torch.Tensor, tau_delta: float = 0.25) -> torch.Tensor:
    """Compute boundary weights for the D-optimal objective.

    Higher weights are assigned to actions near the local boundary and to actions
    with higher audit disagreement.
    """
    top_score = consensus_scores.max()
    score_gap = (top_score - consensus_scores).abs()
    disagreement_scale = disagreement.mean().clamp_min(1e-6)
    weights = torch.exp(-score_gap / max(tau_delta, 1e-6)) * (1.0 + disagreement / disagreement_scale)
    weights = weights / weights.mean().clamp_min(1e-6)
    return weights
