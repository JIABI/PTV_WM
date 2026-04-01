"""PTV-Regime main assembly."""
from __future__ import annotations
import torch
from torch import nn
from .climate_encoder import ClimateEncoder
from .law_state_head import LawStateHead, SEMANTIC_CHANNELS
from .prototype_bank import PrototypeBank
from .message_passing import MessagePassing
from .fast_state_update import FastStateUpdate
from .climate_transition import ClimateTransition
from .residual_channel import ResidualChannel
from .outputs import LICWMOutputs
from ..common.pooling import SystemSummaryPool


class LICWorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, event_dim, h_fast, c_dim, num_prototypes, omega_max,
                 use_event_token=True, transition_mode="full", stochastic_climate=False,
                 enable_residual_channel=False, residual_eps=0.05, fast_mode="gru"):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.c_dim = c_dim
        self.channels = len(SEMANTIC_CHANNELS)
        self.climate_encoder = ClimateEncoder(obs_dim, action_dim, event_dim, h_fast, c_dim, stochastic_climate)
        self.law_head = LawStateHead(c_dim, SEMANTIC_CHANNELS)
        self.bank = PrototypeBank(self.channels, num_prototypes)
        self.message = MessagePassing(self.channels, h_fast)
        self.fast = FastStateUpdate(obs_dim, action_dim, h_fast, h_fast, mode=fast_mode)
        self.pool = SystemSummaryPool(h_fast, h_fast)
        self.transition = ClimateTransition(c_dim, h_fast, event_dim, omega_max, mode=transition_mode, use_event_token=use_event_token)
        self.residual = ResidualChannel(c_dim, h_fast, residual_eps) if enable_residual_channel else None
        self.event_dim = event_dim

    def forward(self, obs_hist, action_hist=None, event_hist=None, horizon=None, teacher_forcing=True,
                climate_oracle=None, law_oracle=None):
        b, t, n, d = obs_hist.shape
        device, dtype = obs_hist.device, obs_hist.dtype
        h = self.fast.init_hidden(b, n, device, dtype)
        enc = self.climate_encoder(obs_hist, action_hist, event_hist)
        c = climate_oracle if climate_oracle is not None else enc["c"]
        horizon = horizon or t - 1
        x = obs_hist[:, 0]
        preds, fast_traj, c_traj, eta_traj = [], [], [], []
        laws = {"rho": [], "beta": [], "tau": []}
        aux = {"msg": [], "omega": [], "delta_c": [], "c_slow": [], "cbar": [], "chi_mean": [], "gamma_mean": [], "msg_norm": [], "channel_activation": [], "u_channel": []}
        for s in range(horizon):
            if teacher_forcing and s < t:
                x = obs_hist[:, s]
            if action_hist is not None and action_hist.numel() > 0 and s < action_hist.shape[1]:
                a = action_hist[:, s]
            else:
                a = x.new_zeros((b, n, self.action_dim))
            if event_hist is not None and s < event_hist.shape[1]:
                e = event_hist[:, s]
            else:
                e = x.new_zeros((b, self.event_dim))
            law = law_oracle if law_oracle is not None else self.law_head(c)
            phi = x.unsqueeze(2) - x.unsqueeze(1)
            dij = torch.norm(phi[..., :2], dim=-1)
            bank = self.bank(phi, dij)
            msg, msg_aux = self.message(phi, dij, law, bank["u_channel"])
            if self.residual is not None:
                msg = self.residual(c, msg)
            h, x_pred = self.fast(h, x.reshape(b * n, d), a.reshape(b * n, self.action_dim), msg.reshape(b * n, -1))
            x = x_pred.reshape(b, n, d)
            q = self.pool(h.reshape(b, n, -1), x)
            trans = self.transition(c, q, e)
            c = trans["c_next"]
            preds.append(x)
            fast_traj.append(h.reshape(b, n, -1))
            c_traj.append(c)
            eta_traj.append(trans["eta"])
            aux["msg"].append(msg)
            aux["omega"].append(trans["omega"])
            aux["delta_c"].append(trans["delta_c"])
            aux["c_slow"].append(trans["c_slow"])
            aux["cbar"].append(trans["cbar"])
            for k, v in msg_aux.items():
                aux[k].append(v)
            for k in ["rho", "beta", "tau"]:
                laws[k].append(law[k])
        return LICWMOutputs(
            pred_states=torch.stack(preds, dim=1),
            pred_obs=torch.stack(preds, dim=1),
            fast_hidden=torch.stack(fast_traj, dim=1),
            climate_states=torch.stack(c_traj, dim=1),
            jump_gates=torch.stack(eta_traj, dim=1),
            law_states={k: torch.stack(v, dim=1) for k, v in laws.items()},
            aux={k: torch.stack(v, dim=1) for k, v in aux.items()},
        )
