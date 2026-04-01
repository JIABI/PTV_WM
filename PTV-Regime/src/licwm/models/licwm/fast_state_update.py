"""Fast state update supporting GRU/MLP variants."""
from __future__ import annotations
import torch
from torch import nn


class FastStateUpdate(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, msg_dim: int, h_dim: int, mode: str = "gru"):
        super().__init__()
        self.mode = mode
        self.obs_dim = obs_dim
        self.h_dim = h_dim
        self.msg_dim = msg_dim
        in_dim = obs_dim + action_dim + msg_dim
        if mode == "gru":
            self.core = nn.GRUCell(in_dim, h_dim)
        else:
            self.core = nn.Sequential(nn.Linear(in_dim + h_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim))
        self.pred_obs = nn.Linear(h_dim, obs_dim)

    def init_hidden(self, batch_size: int, num_agents: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size * num_agents, self.h_dim, device=device, dtype=dtype)

    def forward(self, h, x, a, msg_vec):
        inp = torch.cat([x, a, msg_vec], dim=-1)
        if self.mode == "gru":
            h_next = self.core(inp, h)
        else:
            h_next = self.core(torch.cat([h, inp], dim=-1))
        return h_next, self.pred_obs(h_next)
