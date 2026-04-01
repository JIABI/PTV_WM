from __future__ import annotations

import torch
import torch.nn as nn


class GeometryBottleneck(nn.Module):
    """
    Simple variational bottleneck for RALAG-WM.

    Args:
        input_dim: input feature dimension
        latent_dim: bottleneck latent dimension
        hidden_dim: hidden layer width; if None, defaults to input_dim
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        input_dim = int(input_dim)
        latent_dim = int(latent_dim)
        hidden_dim = int(hidden_dim) if hidden_dim is not None else input_dim

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Input:
            x: [B, input_dim]

        Returns:
            z: [B, latent_dim]
            stats: {"kl": scalar tensor, "mu": [B, latent_dim], "logvar": [B, latent_dim]}
        """
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)

        h = self.net(x.float())
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(min=-10.0, max=10.0)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        kl = 0.5 * torch.mean(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)

        return z, {
            "kl": kl,
            "mu": mu,
            "logvar": logvar,
        }