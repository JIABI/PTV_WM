from __future__ import annotations

from typing import Any

import torch

from ralagwm.data.batch import AuditScores, ChartState


def _sinusoidal_coords(indices: torch.Tensor, coord_dim: int) -> torch.Tensor:
    coord_dim = max(int(coord_dim), 2)
    half = max(coord_dim // 2, 1)
    idx = indices.float().unsqueeze(-1)
    freq = torch.arange(half, device=indices.device, dtype=indices.dtype if indices.is_floating_point() else torch.float32)
    freq = freq.float() / max(half, 1)
    angles = idx / (10.0 ** freq.unsqueeze(0))
    coords = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if coords.shape[-1] < coord_dim:
        pad = torch.zeros(coords.shape[0], coord_dim - coords.shape[-1], device=coords.device, dtype=coords.dtype)
        coords = torch.cat([coords, pad], dim=-1)
    return coords[:, :coord_dim]


def _build_discrete_action_coords(pool_budget: int, coord_dim: int, device: torch.device) -> torch.Tensor:
    idx = torch.arange(pool_budget, device=device)
    return _sinusoidal_coords(idx, coord_dim=coord_dim)


def _build_continuous_action_coords(
    anchor_action: torch.Tensor,
    pool_budget: int,
    action_dim: int,
    mode: str,
) -> torch.Tensor:
    device = anchor_action.device
    action_dim = max(int(action_dim), 1)
    if anchor_action.dim() == 0:
        anchor_action = anchor_action.view(1)
    anchor_action = anchor_action.view(-1)[:action_dim]
    if anchor_action.numel() < action_dim:
        pad = torch.zeros(action_dim - anchor_action.numel(), device=device, dtype=anchor_action.dtype)
        anchor_action = torch.cat([anchor_action, pad], dim=0)

    coords = [anchor_action]
    if pool_budget <= 1:
        return torch.stack(coords, dim=0)

    if mode == 'highdim_continuous':
        for k in range(1, pool_budget):
            basis = torch.zeros(action_dim, device=device, dtype=anchor_action.dtype)
            basis[(k - 1) % action_dim] = 1.0
            scale = 0.10 + 0.02 * ((k - 1) // max(action_dim, 1))
            sign = -1.0 if (k % 2 == 0) else 1.0
            coords.append((anchor_action + sign * scale * basis).clamp(-1.0, 1.0))
    else:
        if action_dim == 1:
            offsets = torch.linspace(-0.5, 0.5, pool_budget, device=device, dtype=anchor_action.dtype)
            coords = [(anchor_action + off.view(1)).clamp(-1.0, 1.0) for off in offsets]
        else:
            steps = torch.linspace(-0.35, 0.35, max(pool_budget // 2 + 1, 2), device=device, dtype=anchor_action.dtype)
            for k in range(1, pool_budget):
                basis = torch.zeros(action_dim, device=device, dtype=anchor_action.dtype)
                basis[(k - 1) % action_dim] = 1.0
                scale = steps[min(k % steps.numel(), steps.numel() - 1)]
                sign = -1.0 if (k % 2 == 0) else 1.0
                coords.append((anchor_action + sign * scale * basis).clamp(-1.0, 1.0))
    if len(coords) < pool_budget:
        pad = coords[-1].clone()
        coords.extend([pad] * (pool_budget - len(coords)))
    return torch.stack(coords[:pool_budget], dim=0)


def build_chart_state_from_audit(
    audit_scores: AuditScores,
    action: torch.Tensor | None = None,
    action_type: str = 'discrete',
    action_dim: int = 1,
    pool_budget: int = 8,
    coord_dim: int | None = None,
    chart_mode: str = 'discrete',
) -> ChartState:
    consensus = audit_scores.consensus_scores.float()
    disagreement = audit_scores.disagreement.float()
    if consensus.dim() == 1:
        consensus = consensus.unsqueeze(0)
    if disagreement.dim() == 1:
        disagreement = disagreement.unsqueeze(0)
    batch_size, audit_dim = consensus.shape
    pool_budget = int(pool_budget)
    action_type = str(action_type)
    chart_mode = str(chart_mode)
    coord_dim = int(coord_dim or (8 if action_type == 'discrete' else action_dim))

    device = consensus.device

    if audit_dim < pool_budget:
        rep = (pool_budget + audit_dim - 1) // audit_dim
        consensus = consensus.repeat(1, rep)[:, :pool_budget]
        disagreement = disagreement.repeat(1, rep)[:, :pool_budget]
    elif audit_dim > pool_budget:
        consensus = consensus[:, :pool_budget]
        disagreement = disagreement[:, :pool_budget]

    if action_type == 'discrete':
        idx = torch.arange(pool_budget, device=device)
        base_coords = _build_discrete_action_coords(pool_budget=pool_budget, coord_dim=coord_dim, device=device)
        action_coords = base_coords.unsqueeze(0).repeat(batch_size, 1, 1)
        anchor_idx = torch.argmax(consensus, dim=-1).float().unsqueeze(-1)
        anchor_action = anchor_idx / max(pool_budget - 1, 1)
    else:
        if action is None:
            anchor_vec = torch.zeros(batch_size, action_dim, device=device)
        else:
            anchor_vec = action.float().view(batch_size, -1)
            if anchor_vec.shape[-1] < action_dim:
                pad = torch.zeros(batch_size, action_dim - anchor_vec.shape[-1], device=device, dtype=anchor_vec.dtype)
                anchor_vec = torch.cat([anchor_vec, pad], dim=-1)
            anchor_vec = anchor_vec[:, :action_dim]
        coords = []
        for b in range(batch_size):
            coords.append(_build_continuous_action_coords(anchor_vec[b], pool_budget=pool_budget, action_dim=action_dim, mode=chart_mode))
        action_coords = torch.stack(coords, dim=0)
        anchor_action = anchor_vec

    metric_matrix = torch.eye(coord_dim, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    metric_matrix = metric_matrix + 0.05 * torch.diag_embed(disagreement[:, :coord_dim].mean(dim=-1, keepdim=True).repeat(1, coord_dim))

    metadata: dict[str, Any] = {
        'coord_dim': coord_dim,
        'pool_budget': pool_budget,
        'action_type': action_type,
        'chart_mode': chart_mode,
        'action_dim': int(action_dim),
        'audit_dim': int(audit_dim),
        'target_chart_state': True,
    }
    return ChartState(
        anchor_action=anchor_action,
        metric_matrix=metric_matrix,
        boundary_saliency=consensus,
        uncertainty=disagreement.clamp_min(1e-6),
        action_coords=action_coords,
        metadata=metadata,
    )
