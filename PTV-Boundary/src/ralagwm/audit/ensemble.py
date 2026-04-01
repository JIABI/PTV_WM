"""Audit ensemble and robust consensus wrapper."""
from __future__ import annotations

import torch
import torch.nn as nn

from ralagwm.typing import AuditScores
from .consensus import build_audit_scores


class AuditEnsemble(nn.Module):
    def __init__(self, heads: list[nn.Module], trim_ratio: float = 0.25):
        super().__init__()
        self.heads = nn.ModuleList(heads)
        self.trim_ratio = float(trim_ratio)

    def forward(self, x: torch.Tensor) -> AuditScores:
        raw = torch.stack([head(x) for head in self.heads], dim=0)
        return build_audit_scores(raw, trim_ratio=self.trim_ratio)

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
