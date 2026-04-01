"""Semantic channelized prototype bank.

The bank is time-shared and independent of the climate. Climate only modulates the
resulting channel activations through the law-state in message passing.
"""
from __future__ import annotations
import torch
from torch import nn
from .basis_families import radial_basis, threshold_basis, directional_basis


class ChannelPrototypeBank(nn.Module):
    def __init__(self, n_channels: int, n_prototypes: int, mode: str = "mixed"):
        super().__init__()
        self.n_channels = n_channels
        self.n_prototypes = n_prototypes
        self.mode = mode
        self.centers = nn.Parameter(torch.rand(n_channels, n_prototypes))
        self.widths = nn.Parameter(torch.ones(n_channels, n_prototypes))
        self.thresholds = nn.Parameter(torch.rand(n_channels, n_prototypes))
        self.directions = nn.Parameter(torch.randn(n_channels, n_prototypes, 2))
        # time-shared channel-internal mixture weights
        self.mix = nn.Parameter(torch.zeros(n_channels, n_prototypes))

    def forward(self, phi_ij: torch.Tensor, d_ij: torch.Tensor) -> dict[str, torch.Tensor]:
        rb = radial_basis(d_ij, self.centers, self.widths)
        tb = threshold_basis(d_ij, self.thresholds)
        db = directional_basis(phi_ij[..., :2], self.directions)
        if self.mode == "radial":
            proto = rb
        elif self.mode == "threshold":
            proto = tb
        elif self.mode == "directional":
            proto = db
        else:
            proto = 0.4 * rb + 0.3 * tb + 0.3 * db
        mix = torch.softmax(self.mix, dim=-1)[None, None, None, :, :]
        u_channel = (mix * proto).sum(dim=-1)
        return {"proto_raw": proto, "u_channel": u_channel, "mix": mix.squeeze(0).squeeze(0).squeeze(0)}
