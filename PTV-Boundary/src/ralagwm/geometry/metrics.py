"""Metrics for geometry quality and control diagnostics."""
from __future__ import annotations

import torch

from ralagwm.typing import RALAGGeometry


def top1_disagreement(pred: RALAGGeometry, target: RALAGGeometry) -> torch.Tensor:
    return (pred.top_action_index != target.top_action_index).float().mean()


def boundary_risk_brier(pred: RALAGGeometry, target: RALAGGeometry) -> torch.Tensor:
    return (pred.boundary_risk - target.boundary_risk).pow(2).mean()
