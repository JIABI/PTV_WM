from __future__ import annotations

import torch
import torch.nn as nn


class BaseAuditHead(nn.Module):
    """Abstract audit head returning local action scores [B, A]."""

    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.num_actions = int(num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
