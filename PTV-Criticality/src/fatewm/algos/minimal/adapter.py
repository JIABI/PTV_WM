import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fatewm.algos.common.interfaces import AlgoAdapter, AlgoOutputs
from fatewm.core.objectives import (
    compute_attention_only_loss,
    compute_fatewm_loss,
    compute_freq_heuristic_loss,
    compute_ms_jepa_uniform_loss,
    compute_rrrm_loss,
    compute_uniform_loss,
)


class MinimalAdapter(AlgoAdapter):
    """Compact self-contained world model used by the PTV-Criticality reference code.

    It is intentionally lightweight so the full training/evaluation stack remains
    runnable without external RL frameworks, while still exposing the interfaces
    required by the paper: encode, score, latent rollout, reward prediction, and
    action selection for both discrete and continuous environments.
    """

    def __init__(self, obs_shape, action_dim: int, is_discrete: bool, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        self.latent_dim = latent_dim

        if isinstance(obs_shape, int) or (hasattr(obs_shape, "__len__") and len(obs_shape) == 1):
            in_dim = obs_shape if isinstance(obs_shape, int) else obs_shape[0]
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self._is_image = False
        else:
            c, h, w = obs_shape
            self.encoder = nn.Sequential(
                nn.Conv2d(c, 32, 4, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
                nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, latent_dim),
            )
            self._is_image = True

        act_in_dim = action_dim
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + act_in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        out_dim = action_dim if is_discrete else action_dim * 2
        self.policy = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.reward = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.slow_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def is_model_based(self) -> bool:
        return True

    def encode(self, obs: torch.Tensor):
        if self._is_image:
            if obs.ndim == 3:
                obs = obs.unsqueeze(0)
            if obs.ndim == 4 and obs.shape[1] not in (1, 3, 4, 12) and obs.shape[-1] in (1, 3):
                obs = obs.permute(0, 3, 1, 2).contiguous()
        else:
            if obs.ndim == 1:
                obs = obs.unsqueeze(0)
        return self.encoder(obs)

    def scores(self, z):
        return self.policy(z)

    def act(self, obs: np.ndarray, eval_mode: bool = False):
        device = next(self.parameters()).device
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        if obs_t.ndim in (1, 3):
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            z = self.encode(obs_t)
            s = self.scores(z)[0]
        if self.is_discrete:
            if (not eval_mode) and (np.random.rand() < 0.1):
                return int(np.random.randint(self.action_dim))
            return int(torch.argmax(s).item())
        half = s.shape[-1] // 2
        mean = s[:half]
        logstd = torch.clamp(s[half:], -5.0, 2.0)
        std = torch.exp(logstd)
        a = mean if eval_mode else (mean + std * torch.randn_like(std))
        return torch.tanh(a).detach().cpu().numpy().astype(np.float32)

    def predict(self, z, actions, delta: int):
        if self.is_discrete:
            if actions.dim() == 2:
                a_seq = F.one_hot(actions, num_classes=self.action_dim).float()
            else:
                a_seq = actions
        else:
            a_seq = actions
        zt = z
        for i in range(int(delta)):
            a = a_seq[:, i]
            x = torch.cat([zt, a], dim=-1)
            zt = self.dynamics(x)
        return zt

    def update(self, batch, deltas, method_cfg, fate_estimator=None, att_gate=None, env_cfg=None, teacher_algo=None) -> AlgoOutputs:
        device = next(self.parameters()).device
        if len(batch) == 5:
            obs_seq, act_seq, rew_seq, done_seq, extra_seq = batch
        else:
            obs_seq, act_seq, rew_seq, done_seq = batch
            extra_seq = None

        obs_seq = torch.tensor(obs_seq, dtype=torch.float32, device=device)
        act_seq = torch.tensor(act_seq, dtype=torch.long if self.is_discrete else torch.float32, device=device)
        rew_seq = torch.tensor(rew_seq, dtype=torch.float32, device=device)
        done_seq = torch.tensor(done_seq, dtype=torch.float32, device=device)
        batch_tensors = (obs_seq, act_seq, rew_seq, done_seq, extra_seq)

        method_name = str(method_cfg.get("name", "ptv_criticality"))
        if method_name in ("ptv_criticality", "rrrm", "fatewm"):
            loss, logs = compute_rrrm_loss(self, fate_estimator, batch_tensors, deltas, method_cfg, device, env_cfg=env_cfg, teacher_algo=teacher_algo)
        elif method_name == "uniform":
            loss, logs = compute_uniform_loss(self, batch_tensors, deltas, method_cfg, device)
        elif method_name == "freq_heuristic":
            loss, logs = compute_freq_heuristic_loss(self, batch_tensors, deltas, method_cfg, device)
        elif method_name == "attention_only":
            if att_gate is None:
                raise ValueError("attention_only requires att_gate.")
            loss, logs = compute_attention_only_loss(self, att_gate, batch_tensors, deltas, method_cfg, device)
        elif method_name == "ms_jepa_uniform":
            loss, logs = compute_ms_jepa_uniform_loss(self, batch_tensors, deltas, method_cfg, device)
        else:
            loss, logs = compute_fatewm_loss(self, fate_estimator, batch_tensors, deltas, method_cfg, device, env_cfg=env_cfg, teacher_algo=teacher_algo)

        # Auxiliary one-step reward prediction.
        try:
            z0 = self.encode(obs_seq[:, 0])
            a0 = act_seq[:, 0:1] if self.is_discrete else act_seq[:, 0:1, :]
            z1 = self.predict(z0, a0, delta=1)
            reward_loss = torch.mean((self.reward(z1) - rew_seq[:, 0:1]) ** 2)
            loss = loss + float(method_cfg.get("lambda_reward", 0.1)) * reward_loss
            logs["reward_loss"] = float(reward_loss.detach().cpu())
        except Exception:
            pass

        # Auxiliary one-step slow-head supervision for toy-style interfaces.
        try:
            if obs_seq.ndim == 3 and obs_seq.shape[-1] % 2 == 0:
                slow_t1 = obs_seq[:, 1, obs_seq.shape[-1] // 2 :]
                slow_norm_t1 = torch.norm(slow_t1, dim=-1, keepdim=True)
                z0 = self.encode(obs_seq[:, 0])
                a0 = act_seq[:, 0:1] if self.is_discrete else act_seq[:, 0:1, :]
                z1 = self.predict(z0, a0, delta=1)
                slow_loss = torch.mean((self.slow_head(z1) - slow_norm_t1) ** 2)
                loss = loss + float(method_cfg.get("lambda_slow", 0.2)) * slow_loss
                logs["slow_loss"] = float(slow_loss.detach().cpu())
        except Exception:
            pass

        logs["loss_total"] = float(loss.detach().cpu())
        return AlgoOutputs(loss=loss, logs=logs)
