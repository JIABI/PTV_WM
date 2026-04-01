"""Audit training loop driven by replayed environment transitions."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ralagwm.data.replay import ReplayBuffer


def _prepare_obs(obs: torch.Tensor) -> torch.Tensor:
    if obs.dim() == 4:
        return obs.float()
    if obs.dim() > 2:
        return obs.reshape(obs.shape[0], -1).float()
    return obs.float()


def train_audit_epoch(model, optimizer: torch.optim.Optimizer, replay: ReplayBuffer, batch_size: int = 8, steps: int = 8, device: torch.device | str = 'cpu') -> dict[str, float]:
    total = 0.0
    for _ in range(int(steps)):
        batch = replay.sample_tensors(batch_size=batch_size, device=device)
        x = _prepare_obs(batch['obs'])
        audit_scores = model(x)
        actions = batch['actions']
        if actions.dim() > 1 and actions.shape[1] > 1:
            target = torch.tanh(actions.float())[:, : audit_scores.consensus_scores.shape[-1]]
            loss = F.mse_loss(audit_scores.consensus_scores[:, : target.shape[-1]], target)
        else:
            y = actions.reshape(-1).long().clamp(min=0, max=audit_scores.consensus_scores.shape[-1] - 1)
            loss = F.cross_entropy(audit_scores.consensus_scores, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return {'loss': total / max(int(steps), 1)}
