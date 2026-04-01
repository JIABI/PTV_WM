from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class ShieldState:
    risks: list
    prev_action: int = 0
    prev_value: Optional[float] = None
    n_decisions: int = 0
    n_holds: int = 0

    @staticmethod
    def new():
        return ShieldState(risks=[], prev_action=0, prev_value=None, n_decisions=0, n_holds=0)

    def update(self, r: float):
        self.risks.append(float(r))


@torch.no_grad()
def predict_next_slow_norm(algo, obs: np.ndarray, action: int, device: torch.device) -> float:
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    if obs_t.ndim in (1, 3):
        obs_t = obs_t.unsqueeze(0)
    z0 = algo.encode(obs_t)
    a_t = torch.tensor([[int(action)]], dtype=torch.long, device=device)
    z1 = algo.predict(z0, a_t, delta=1)
    if hasattr(algo, "slow_head"):
        return float(algo.slow_head(z1).squeeze().detach().cpu())
    return float(torch.norm(z1, dim=-1).item())


@torch.no_grad()
def fixed_score_decision(algo, obs: np.ndarray, device: torch.device) -> int:
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    if obs_t.ndim in (1, 3):
        obs_t = obs_t.unsqueeze(0)
    z = algo.encode(obs_t)
    scores = algo.scores(z)[0]
    return int(torch.argmax(scores).item())


@torch.no_grad()
def fixed_score_with_shield(
    algo,
    obs: np.ndarray,
    device: torch.device,
    event_threshold: float,
    alpha: float,
    fallback_action: int,
    shield_state: Optional[ShieldState] = None,
) -> int:
    a = fixed_score_decision(algo, obs, device)
    risk = predict_next_slow_norm(algo, obs, a, device) / (float(event_threshold) + 1e-6)
    if shield_state is not None:
        shield_state.update(float(risk))
    return int(fallback_action) if risk > float(alpha) else int(a)


@torch.no_grad()
def min_slow_decision(
    algo,
    obs: np.ndarray,
    device: torch.device,
    hysteresis_margin: float = 0.0,
    hysteresis_state: Optional[ShieldState] = None,
) -> int:
    A = int(getattr(algo, "action_dim", 3))
    best_a, best_v = 0, 1e18
    for a in range(A):
        v = predict_next_slow_norm(algo, obs, a, device)
        if v < best_v:
            best_a, best_v = a, v

    if hysteresis_state is not None and float(hysteresis_margin) > 0.0:
        prev_a = int(hysteresis_state.prev_action)
        prev_v = hysteresis_state.prev_value
        if prev_v is None:
            prev_v = predict_next_slow_norm(algo, obs, prev_a, device)
        improvement = float(prev_v) - float(best_v)
        hysteresis_state.n_decisions += 1
        if best_a != prev_a and improvement < float(hysteresis_margin):
            hysteresis_state.n_holds += 1
            chosen_a, chosen_v = prev_a, float(prev_v)
        else:
            chosen_a, chosen_v = int(best_a), float(best_v)
        hysteresis_state.prev_action = int(chosen_a)
        hysteresis_state.prev_value = float(chosen_v)
        return int(chosen_a)

    if hysteresis_state is not None:
        hysteresis_state.prev_action = int(best_a)
        hysteresis_state.prev_value = float(best_v)
        hysteresis_state.n_decisions += 1
    return int(best_a)


@torch.no_grad()
def min_slow_with_shield(
    algo,
    obs: np.ndarray,
    device: torch.device,
    event_threshold: float,
    alpha: float,
    fallback_action: int,
    shield_state: Optional[ShieldState] = None,
    hysteresis_margin: float = 0.0,
) -> int:
    a = min_slow_decision(
        algo,
        obs,
        device,
        hysteresis_margin=float(hysteresis_margin),
        hysteresis_state=shield_state,
    )
    slow_hat = predict_next_slow_norm(algo, obs, a, device)
    risk = float(slow_hat / (float(event_threshold) + 1e-6))
    if shield_state is not None:
        shield_state.update(risk)
    return int(fallback_action) if risk > float(alpha) else int(a)
