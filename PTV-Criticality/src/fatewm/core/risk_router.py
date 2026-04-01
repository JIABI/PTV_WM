"""Budgeted risk router for RRRM-style horizon allocation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .allocation import budgeted_allocation


@dataclass
class RouterConfig:
    z_feat_dim: int = 4
    time_emb_dim: int = 16
    hidden: int = 128
    n_experts: int = 4
    entropy_beta: float = 0.02
    budget_B: float = 8.0
    dual_lr: float = 1e-3
    selection_temperature: float = 0.5
    selection_kind: str = "sparsemax"  # {sparsemax, sigmoid}


class RiskRouter(nn.Module):
    def __init__(self, cfg: RouterConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.z_feat_dim + cfg.time_emb_dim
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden), nn.ReLU(),
            nn.Linear(cfg.hidden, cfg.hidden), nn.ReLU(),
        )
        self.importance_head = nn.Linear(cfg.hidden, 1)
        self.fate_head = nn.Linear(cfg.hidden, cfg.n_experts)
        self.register_buffer("lambda_dual", torch.tensor(0.0))

    def forward(self, z_feat: torch.Tensor, t_emb: torch.Tensor):
        """Return per-sample horizon logits and fate probabilities.

        Args:
            z_feat: [B, z_feat_dim]
            t_emb: [B, J, time_emb_dim]
        Returns:
            logits: [B, J]
            fate_probs: [B, J, K]
            reg: scalar entropy regularizer
        """
        B, J = t_emb.shape[:2]
        z = z_feat.unsqueeze(1).expand(B, J, z_feat.shape[-1])
        x = torch.cat([z, t_emb], dim=-1)
        h = self.backbone(x)
        logits = self.importance_head(h).squeeze(-1)
        fate_logits = self.fate_head(h)
        fate_probs = torch.softmax(fate_logits, dim=-1)
        ent = -(fate_probs * torch.log(fate_probs.clamp_min(1e-8))).sum(dim=-1).mean()
        reg = float(self.cfg.entropy_beta) * ent
        return logits, fate_probs, reg

    def allocate(self, logits: torch.Tensor) -> torch.Tensor:
        return budgeted_allocation(
            logits,
            self.cfg.budget_B,
            temperature=self.cfg.selection_temperature,
            kind=self.cfg.selection_kind,
            lambda_dual=self.lambda_dual if self.cfg.selection_kind == "sigmoid" else None,
            stop_grad=False,
        )

    @torch.no_grad()
    def dual_update(self, mean_budget_violation: float):
        new_val = float(self.lambda_dual.item()) + float(self.cfg.dual_lr) * float(mean_budget_violation)
        self.lambda_dual.fill_(max(0.0, new_val))
