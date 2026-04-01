"""Slow climate dynamics + optional jump."""
from __future__ import annotations
import torch
from torch import nn


class ClimateTransition(nn.Module):
    def __init__(self, c_dim: int, q_dim: int, event_dim: int, omega_max: float = 0.2, mode: str = "full", use_event_token: bool = True):
        super().__init__()
        self.mode = mode
        self.use_event_token = use_event_token
        self.omega_max = omega_max
        cond_dim = q_dim + (event_dim if use_event_token else 0)
        self.g_phi = nn.Sequential(nn.Linear(c_dim + q_dim, c_dim), nn.Tanh())
        self.w_omega = nn.Linear(q_dim, c_dim)
        self.w_eta = nn.Linear(cond_dim, c_dim)
        self.u_delta = nn.Linear(c_dim + cond_dim, c_dim)
        self.A = nn.Parameter(torch.ones(c_dim))
        self.event_dim = event_dim

    def forward(self, c_t, q_t, event_t=None, dt=1.0):
        cbar = self.g_phi(torch.cat([c_t, q_t], dim=-1))
        omega = self.omega_max * torch.sigmoid(self.w_omega(q_t))
        c_slow = c_t if self.mode == "no_slow" else c_t + dt * omega * (cbar - c_t)
        if self.mode == "no_jump":
            eta = torch.zeros_like(c_t)
            delta = torch.zeros_like(c_t)
            c_next = c_slow
        else:
            if self.use_event_token:
                if event_t is None:
                    event_t = q_t.new_zeros((q_t.shape[0], self.event_dim))
                cond = torch.cat([q_t, event_t], dim=-1)
            else:
                cond = q_t
            eta = torch.sigmoid(self.w_eta(cond))
            delta = self.A * torch.tanh(self.u_delta(torch.cat([c_t, cond], dim=-1)))
            c_next = c_slow + eta * delta
        return {"c_next": c_next, "c_slow": c_slow, "eta": eta, "delta_c": delta, "omega": omega, "cbar": cbar}
