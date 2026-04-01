"""Channelized law-shaped message passing."""
from __future__ import annotations
import torch
from torch import nn


class MessagePassing(nn.Module):
    def __init__(self, n_channels: int, msg_dim: int):
        super().__init__()
        self.n_channels = n_channels
        self.channel_to_vec = nn.Linear(n_channels, msg_dim)
        self.radius_sharpness = 6.0
        self.gate_sharpness = 4.0

    def forward(self, phi_ij: torch.Tensor, d_ij: torch.Tensor, law_state: dict, u_channel: torch.Tensor):
        """Args:
            phi_ij: [B,N,N,D]
            d_ij: [B,N,N]
            law_state['rho']: [B,1]
            law_state['beta']: [B,C]
            law_state['tau']: [B,C]
            u_channel: [B,N,N,C]
        Returns:
            node_vec: [B,N,msg_dim]
            aux dict with channel/gate statistics
        """
        rho, beta, tau = law_state["rho"], law_state["beta"], law_state["tau"]
        chi = torch.sigmoid(self.radius_sharpness * (rho[:, None, None, :] - d_ij.unsqueeze(-1)))
        # gate score uses learned prototype response magnitude rather than raw distance only.
        gamma = torch.sigmoid(self.gate_sharpness * (u_channel - tau[:, None, None, :]))
        msg_ch = chi * gamma * beta[:, None, None, :] * u_channel
        msg_ch = msg_ch * (1.0 - torch.eye(d_ij.shape[-1], device=d_ij.device)[None, :, :, None])
        node_ch = msg_ch.sum(dim=2)
        node_vec = self.channel_to_vec(node_ch)
        aux = {
            "chi_mean": chi.mean(dim=(1, 2)),
            "gamma_mean": gamma.mean(dim=(1, 2)),
            "msg_norm": node_vec.norm(dim=-1).mean(dim=1),
            "channel_activation": node_ch.mean(dim=1),
            "u_channel": u_channel.mean(dim=(1, 2)),
        }
        return node_vec, aux
