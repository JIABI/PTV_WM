"""Batch contracts for all domains."""
from dataclasses import dataclass, field
from typing import Any
import torch

@dataclass
class MultiAgentBatch:
    """Unified multi-agent sequence batch.

    Shapes:
        obs_hist: [B,T,N,D_obs]
        state_hist: [B,T,N,D_state] optional
        action_hist: [B,T,N,D_act] optional
        event_hist: [B,T,D_event] optional
        fut_obs: [B,H,N,D_obs] optional
        fut_state: [B,H,N,D_state]
        mask: [B,T_or_H,N] (1 for valid agent/time)
    """
    obs_hist: torch.Tensor
    fut_state: torch.Tensor
    mask: torch.Tensor
    state_hist: torch.Tensor | None = None
    action_hist: torch.Tensor | None = None
    event_hist: torch.Tensor | None = None
    fut_obs: torch.Tensor | None = None
    meta: dict[str, Any] = field(default_factory=dict)
