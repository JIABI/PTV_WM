"""Collate for variable-agent batches."""
from typing import Sequence
import torch
from .batch_types import MultiAgentBatch

def _pad_agents(x: torch.Tensor, max_agents: int) -> torch.Tensor:
    if x.size(-2) == max_agents:
        return x
    pad_shape = list(x.shape)
    pad_shape[-2] = max_agents - x.size(-2)
    return torch.cat([x, torch.zeros(*pad_shape, dtype=x.dtype)], dim=-2)

def collate_multi_agent(batch: Sequence[MultiAgentBatch]) -> MultiAgentBatch:
    max_n = max(item.obs_hist.size(-2) for item in batch)
    obs = torch.stack([_pad_agents(x.obs_hist, max_n) for x in batch])
    fut = torch.stack([_pad_agents(x.fut_state, max_n) for x in batch])
    mask = torch.stack([_pad_agents(x.mask.unsqueeze(-1), max_n).squeeze(-1) for x in batch])

    def stack_opt(name: str):
        vals = [getattr(x, name) for x in batch]
        if any(v is None for v in vals):
            return None
        return torch.stack([_pad_agents(v, max_n) if v.ndim >= 3 else v for v in vals])

    return MultiAgentBatch(
        obs_hist=obs,
        state_hist=stack_opt("state_hist"),
        action_hist=stack_opt("action_hist"),
        event_hist=stack_opt("event_hist"),
        fut_obs=stack_opt("fut_obs"),
        fut_state=fut,
        mask=mask,
        meta={"max_agents": max_n},
    )
