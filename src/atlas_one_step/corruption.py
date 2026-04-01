from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class DiffusionLikeCorruption:
    t_min: float = 0.05
    t_max: float = 0.95
    num_time_samples: int = 0
    eps: float = 1e-6

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.num_time_samples and self.num_time_samples > 1:
            t = torch.randint(0, self.num_time_samples, (batch_size,), device=device, dtype=torch.long)
            t = t.float() / float(self.num_time_samples - 1)
        else:
            t = torch.rand(batch_size, device=device)
        return self.t_min + (self.t_max - self.t_min) * t

    def alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.cos(0.5 * torch.pi * t)
        sigma = torch.sin(0.5 * torch.pi * t)
        return alpha, sigma

    def sample_xt(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if eps is None:
            eps = torch.randn_like(x0)
        alpha, sigma = self.alpha_sigma(t)
        while alpha.ndim < x0.ndim:
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        xt = alpha * x0 + sigma * eps
        return xt, eps

    def primitives(self, x0: torch.Tensor, xt: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        alpha, sigma = self.alpha_sigma(t)
        while alpha.ndim < x0.ndim:
            alpha = alpha.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        u_t = alpha * eps - sigma * x0  # v-pred-like primitive
        r_t = x0 - xt
        return {"x0": x0, "u_t": u_t, "r_t": r_t, "eps": eps, "xt": xt, "alpha": alpha, "sigma": sigma, "t_scalar": t}
