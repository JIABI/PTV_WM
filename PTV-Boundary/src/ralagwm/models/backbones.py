"""Sequence backbones for temporal latent modeling."""

import torch
import torch.nn as nn


class RecurrentBackbone(nn.Module):
    """GRU backbone.

    Input:  [B, T, H] or [B, H]
    Output: [B, H]
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        y, _ = self.rnn(x)
        return y[:, -1]


class TinyTransformerBackbone(nn.Module):
    """Transformer-like placeholder for future stronger variants."""

    def __init__(self, hidden_dim: int = 64, nhead: int = 4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.encoder(x)[:, -1]


def build_backbone(kind: str = "gru", hidden_dim: int = 64, num_layers: int = 1) -> nn.Module:
    if kind == "transformer":
        return TinyTransformerBackbone(hidden_dim=hidden_dim)
    return RecurrentBackbone(hidden_dim=hidden_dim, num_layers=num_layers)
