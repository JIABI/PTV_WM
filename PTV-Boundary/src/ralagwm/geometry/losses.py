"""Losses for PTV-Boundary geometry supervision."""
from __future__ import annotations

import torch

from ralagwm.typing import RALAGGeometry


def _match_1d(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = a.reshape(-1)
    b = b.reshape(-1)
    n = min(a.numel(), b.numel())
    return a[:n], b[:n]


def score_field_error(pred: RALAGGeometry, target: RALAGGeometry) -> torch.Tensor:
    a, b = _match_1d(pred.centered_scores, target.centered_scores)
    return (a - b).pow(2).mean() if a.numel() else torch.zeros(1, device=pred.margin.device, dtype=pred.margin.dtype).squeeze(0)


def margin_error(pred: RALAGGeometry, target: RALAGGeometry) -> torch.Tensor:
    a, b = _match_1d(pred.margin, target.margin)
    return (a - b).pow(2).mean() if a.numel() else torch.zeros(1, device=pred.margin.device, dtype=pred.margin.dtype).squeeze(0)


def edge_sensitivity_error(pred: RALAGGeometry, target: RALAGGeometry) -> torch.Tensor:
    a, b = _match_1d(pred.edge_sensitivity, target.edge_sensitivity)
    return (a - b).pow(2).mean() if a.numel() else torch.zeros(1, device=pred.margin.device, dtype=pred.margin.dtype).squeeze(0)


def geometry_distance(pred: RALAGGeometry, target: RALAGGeometry, alpha_s: float = 1.0, alpha_m: float = 1.0, alpha_k: float = 1.0) -> torch.Tensor:
    return alpha_s * score_field_error(pred, target) + alpha_m * margin_error(pred, target) + alpha_k * edge_sensitivity_error(pred, target)
