"""Bounded residual audit channel.

This module is only intended for audit/ablation runs. It receives law-shaped
messages and adds a tightly bounded generic residual in the same message space,
without directly injecting climate into the fast hidden state.
"""
from __future__ import annotations
import torch
from torch import nn


class ResidualChannel(nn.Module):
    def __init__(self, c_dim: int, msg_dim: int, eps: float = 0.05):
        super().__init__()
        self.eps = eps
        self.residual = nn.Sequential(nn.Linear(c_dim, msg_dim), nn.Tanh())

    def forward(self, c_t: torch.Tensor, node_msg: torch.Tensor) -> torch.Tensor:
        # node_msg: [B,N,D_msg]
        res = self.residual(c_t).unsqueeze(1).expand_as(node_msg)
        return node_msg + self.eps * res
