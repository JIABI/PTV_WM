"""Map climate state to bounded law-state channels."""
from __future__ import annotations
import torch
from torch import nn

SEMANTIC_CHANNELS = ["attraction", "repulsion", "alignment", "goal", "risk"]


class LawStateHead(nn.Module):
    def __init__(
        self,
        c_dim: int,
        channels: list[str] | None = None,
        rho_bounds: tuple[float, float] = (0.2, 2.0),
        beta_bounds: tuple[float, float] = (0.1, 2.0),
        tau_bounds: tuple[float, float] = (-1.0, 1.0),
    ):
        super().__init__()
        self.channels = channels or SEMANTIC_CHANNELS
        self.r = len(self.channels)
        self.rho_bounds = rho_bounds
        self.beta_bounds = beta_bounds
        self.tau_bounds = tau_bounds
        self.rho = nn.Linear(c_dim, 1)
        self.beta = nn.Linear(c_dim, self.r)
        self.tau = nn.Linear(c_dim, self.r)

    def _bound(self, raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return lo + (hi - lo) * torch.sigmoid(raw)

    def forward(self, c_t: torch.Tensor) -> dict[str, torch.Tensor]:
        rho = self._bound(self.rho(c_t), *self.rho_bounds)
        beta = self._bound(self.beta(c_t), *self.beta_bounds)
        tau = self._bound(self.tau(c_t), *self.tau_bounds)
        return {"rho": rho, "beta": beta, "tau": tau, "channels": self.channels}
