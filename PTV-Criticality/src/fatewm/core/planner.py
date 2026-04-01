from __future__ import annotations
import itertools
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch

from fatewm.core.utils import stopgrad


@torch.no_grad()
def plan_discrete_mpc(
    algo,
    obs: np.ndarray,
    deltas: list[int],
    fate_estimator,
    cfg,
    device: torch.device,
) -> int:
    """Simple MPC for discrete action spaces (toy).
    Uses the learned latent dynamics + learned reward head (if present) to score action sequences.

    Score(seq) = sum_{k=0..H-1} gamma^k * r_hat(z_k) - risk_weight * risk_hat(z_k)
    risk_hat uses FateEstimator on instantaneous features (a0,v,state,delta,logdelta) for delta=1.
    """
    H = int(cfg.planner.horizon)
    A = int(algo.action_dim)
    gamma = float(cfg.planner.discount)
    risk_w = float(cfg.planner.risk_weight)
    use_fate = bool(cfg.planner.use_fate_risk) and (fate_estimator is not None)

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    if obs_t.ndim == 1:
        obs_t = obs_t.unsqueeze(0)
    z0 = algo.encode(obs_t)  # [1, latent]

    # enumerate all sequences for toy: A^H (H=5, A=3 => 243)
    # if A^H too large, sample randomly
    total = A ** H
    n_candidates = int(getattr(cfg.planner, "n_candidates", total))
    if n_candidates >= total:
        seqs = list(itertools.product(range(A), repeat=H))
    else:
        rng = np.random.default_rng(int(cfg.seed) + 12345)
        seqs = [tuple(rng.integers(0, A, size=H).tolist()) for _ in range(n_candidates)]

    best_score = -1e18
    best_a0 = 0

    for seq in seqs:
        z = z0
        score = 0.0
        for k, a in enumerate(seq):
            # predict next latent (delta=1 step)
            a_t = torch.tensor([[a]], dtype=torch.long, device=device)  # [1,1]
            z = algo.predict(z, a_t, delta=1)

            # reward prediction: prefer reward_head if exists, else heuristic from latent norm
            if hasattr(algo, "reward"):
                r_hat = float(algo.reward(z).squeeze().detach().cpu())
            else:
                r_hat = float((-0.1 * torch.norm(z, dim=-1)).squeeze().detach().cpu())

            step_score = (gamma ** k) * r_hat

            if use_fate:
                # features: a0=|r|, v=margin proxy, state summary, delta, logdelta
                scores = algo.scores(z)  # [1,A]
                top2 = torch.topk(scores, k=2, dim=-1).values
                margin = (top2[:, 0] - top2[:, 1]).abs().unsqueeze(-1)  # bigger margin => safer
                # vulnerability proxy inversely related to margin
                v = 1.0 / (margin + 1e-6)

                a0 = torch.tensor([[abs(r_hat)]], device=device, dtype=torch.float32)
                state = z.mean(dim=-1, keepdim=True)
                delta = torch.tensor([[1.0]], device=device)
                logdelta = torch.tensor([[0.0]], device=device)
                phi = torch.cat([a0, v, state, delta, logdelta], dim=-1)
                I_hat = float(fate_estimator(phi).squeeze().detach().cpu())
                step_score -= risk_w * (gamma ** k) * I_hat

            score += step_score

        if score > best_score:
            best_score = score
            best_a0 = int(seq[0])

    return best_a0
