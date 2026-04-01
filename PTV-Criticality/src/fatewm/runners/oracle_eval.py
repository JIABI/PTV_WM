from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
import torch

from fatewm.core.decision_interface import fixed_score_decision, fixed_score_with_shield


def oracle_rollout_toy(
    cfg,
    env,
    algo,
    device: torch.device,
    fate_estimator,
    deltas: List[int],
) -> Dict[str, Any]:
    """Oracle substitution for Toy:
    - policy logic fixed (score-based argmax)
    - oracle provides TRUE risk from env state (slow_norm / threshold) for shielding decision
    - this isolates drift due to risk mis-estimation vs interface logic.
    """
    obs = env.reset(seed=int(cfg.seed) + 424242)
    done = False
    ep_rew = []
    ep_failed = False

    tau = float(cfg.interface.shield.tau)
    fallback = int(cfg.interface.shield.fallback_action)

    while not done:
        # fixed score-based action candidate
        a = fixed_score_decision(algo, obs, device)

        if bool(cfg.interface.shield.enabled) and bool(cfg.interface.oracle.use_true_risk):
            # compute TRUE risk based on slow norm
            thr = float(cfg.env.event_threshold)
            slow = obs[obs.shape[0]//2:]
            true_risk = float(np.linalg.norm(slow) / (thr + 1e-6))
            if true_risk > tau:
                a = fallback

        obs, r, done, info = env.step(a)
        ep_rew.append(r)
        if info.get("failure", False):
            ep_failed = True

    return {
        "oracle_return": float(np.sum(ep_rew)),
        "oracle_failed": int(ep_failed),
    }
