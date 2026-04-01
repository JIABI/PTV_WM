"""Boundary-identifiability objective."""
from __future__ import annotations

import torch

from ralagwm.utils.math_ops import safe_logdet


class BICObjective:
    """Boundary-weighted D-optimal design objective."""

    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim

    def information_matrix(self, psi: torch.Tensor, weights: torch.Tensor, indices: list[int]) -> torch.Tensor:
        info = torch.eye(self.feature_dim, device=psi.device, dtype=psi.dtype)
        for idx in indices:
            vec = psi[idx].unsqueeze(1)
            info = info + weights[idx] * (vec @ vec.T)
        return info

    def score(self, psi: torch.Tensor, weights: torch.Tensor, indices: list[int]) -> torch.Tensor:
        return safe_logdet(self.information_matrix(psi, weights, indices))
