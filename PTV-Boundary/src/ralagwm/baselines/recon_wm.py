from __future__ import annotations

import torch
import torch.nn as nn

from ralagwm.models.backbones import build_backbone
from ralagwm.models.encoders import build_encoder


class ReconWM(nn.Module):
    def __init__(self, obs_dim: int = 16, action_dim: int = 6, hidden_dim: int = 64, obs_type: str = 'proprio', image_size: int = 64, image_channels: int = 3):
        super().__init__()
        self.obs_type = str(obs_type)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.encoder = build_encoder(obs_type=obs_type, obs_dim=obs_dim, hidden_dim=hidden_dim, image_channels=image_channels, image_size=image_size)
        self.backbone = build_backbone(kind='gru', hidden_dim=hidden_dim)
        self.target_dim = obs_dim if obs_type != 'image' else image_channels * image_size * image_size
        self.head = nn.Linear(hidden_dim, self.target_dim)

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(self.encoder(obs))
        return {'prediction': self.head(h), 'logits': self.head(h)}
