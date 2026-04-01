"""Pooling utilities for fast-state system summaries q_t."""
import torch
from torch import nn

class SystemSummaryPool(nn.Module):
    def __init__(self, h_dim: int, q_dim: int):
        super().__init__()
        self.proj = nn.Linear(h_dim + 2, q_dim)

    def forward(self, h_nodes: torch.Tensor, x_nodes: torch.Tensor) -> torch.Tensor:
        # h_nodes: [B,N,H], x_nodes: [B,N,D]
        feats = torch.cat([h_nodes.mean(dim=1), x_nodes[..., :2].mean(dim=1)], dim=-1)
        return torch.tanh(self.proj(feats))
