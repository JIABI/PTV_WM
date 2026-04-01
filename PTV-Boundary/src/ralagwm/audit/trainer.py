from __future__ import annotations

import torch
import torch.nn.functional as F

from ralagwm.data.replay import ReplayBuffer


def _prepare_obs(batch_obs: torch.Tensor) -> torch.Tensor:
    if batch_obs.dim() == 4:
        return batch_obs.float()
    if batch_obs.dim() > 2:
        return batch_obs.reshape(batch_obs.shape[0], -1).float()
    return batch_obs.float()


def train_audit_epoch(model, optimizer: torch.optim.Optimizer, replay: ReplayBuffer, batch_size: int = 8, steps: int = 8, device: torch.device | str = 'cpu') -> dict[str, float]:
    total = 0.0
    for _ in range(int(steps)):
        batch = replay.sample_tensors(batch_size=batch_size, device=device)
        x = _prepare_obs(batch['obs'])
        scores = model(x)
        actions = batch['actions']
        if actions.dim() > 1 and actions.shape[1] > 1:
            target = torch.tanh(actions.float())
            loss = F.mse_loss(scores.consensus_scores, target[:, : scores.consensus_scores.shape[-1]])
        else:
            target = actions.reshape(-1).long().clamp(min=0, max=scores.consensus_scores.shape[-1] - 1)
            loss = F.cross_entropy(scores.consensus_scores, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return {'loss': total / max(int(steps), 1)}
