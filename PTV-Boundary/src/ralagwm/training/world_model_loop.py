"""Environment-driven world-model training loop for RALAG-WM."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ralagwm.audit.ensemble import AuditEnsemble
from ralagwm.audit.heads import build_audit_head_for_obs
from ralagwm.data.replay import ReplayBuffer
from ralagwm.envs.base import BaseEnvAdapter
from ralagwm.training.loops import build_ralag_batch_from_replay, collect_rollout, select_action_from_model, train_ralag_step


@dataclass
class EpochSummary:
    collect_mean_return: float
    train_loss: float
    pred_geom: float
    refined_geom: float
    zeta: float
    kl: float
    replay_size: int


def build_default_audit_ensemble(obs_shape: tuple[int, ...] | int, action_dim: int, num_audits: int = 4, hidden: int = 64) -> AuditEnsemble:
    if isinstance(obs_shape, int):
        obs_shape = (obs_shape,)
    heads = [build_audit_head_for_obs(obs_shape=obs_shape, num_actions=action_dim) for _ in range(num_audits)]
    ensemble = AuditEnsemble(heads=heads, trim_ratio=0.25)
    ensemble.eval()
    return ensemble


def warmstart_replay(env: BaseEnvAdapter, replay: ReplayBuffer, episodes: int = 2, max_steps: int | None = None, seed: int | None = None) -> dict[str, float]:
    return collect_rollout(env=env, replay=replay, policy_fn=lambda obs: env.sample_random_action(), episodes=episodes, max_steps=max_steps, seed=seed)


def collect_model_rollout(env: BaseEnvAdapter, replay: ReplayBuffer, model, episodes: int = 1, max_steps: int | None = None, device: torch.device | str = 'cpu', seed: int | None = None) -> dict[str, float]:
    return collect_rollout(env=env, replay=replay, policy_fn=lambda obs: select_action_from_model(model, obs, env, device=device)[0], episodes=episodes, max_steps=max_steps, seed=seed)


def evaluate_deploy_policy(env: BaseEnvAdapter, model, episodes: int = 2, max_steps: int | None = None, device: torch.device | str = 'cpu', seed: int | None = None) -> dict[str, float]:
    returns: list[float] = []
    lengths: list[int] = []
    flip = 0.0
    max_steps = int(max_steps or env.spec.max_episode_steps)
    for ep in range(int(episodes)):
        obs, _ = env.reset(seed=None if seed is None else seed + ep)
        prev_action = None
        prev_prev_action = None
        ep_return = 0.0
        for t in range(max_steps):
            action, _ = select_action_from_model(model, obs, env, device=device)
            step = env.step(action)
            ep_return += float(step.reward)
            if prev_prev_action is not None and prev_action is not None:
                if env.spec.action_type == 'discrete':
                    if int(action) != int(prev_action) and int(action) == int(prev_prev_action):
                        flip += 1.0
                else:
                    a = np.asarray(action, dtype=np.float32)
                    b = np.asarray(prev_action, dtype=np.float32)
                    c = np.asarray(prev_prev_action, dtype=np.float32)
                    if np.linalg.norm(a - b) > 1e-3 and np.linalg.norm(a - c) < 1e-3:
                        flip += 1.0
            prev_prev_action = prev_action
            prev_action = action
            obs = step.observation
            if step.done:
                lengths.append(t + 1)
                break
        else:
            lengths.append(max_steps)
        returns.append(ep_return)
    arr = np.asarray(returns, dtype=np.float32)
    cvar = float(np.mean(np.sort(arr)[: max(1, len(arr) // 10)])) if len(arr) else 0.0
    return {
        'episodes': float(len(returns)),
        'mean_return': float(np.mean(arr)) if len(arr) else 0.0,
        'median_return': float(np.median(arr)) if len(arr) else 0.0,
        'cvar10': cvar,
        'mean_length': float(np.mean(lengths)) if lengths else 0.0,
        'flip_proxy': float(flip),
    }


def train_world_model_epoch(model, optimizer: torch.optim.Optimizer, env: BaseEnvAdapter, replay: ReplayBuffer, audit_ensemble: AuditEnsemble, batch_size: int = 8, gradient_steps: int = 8, collection_episodes: int = 1, max_steps: int | None = None, kl_weight: float = 0.01, device: torch.device | str = 'cpu', seed: int | None = None) -> EpochSummary:
    collect_metrics = collect_model_rollout(env, replay, model, episodes=collection_episodes, max_steps=max_steps, device=device, seed=seed)
    metric_sums = {'loss': 0.0, 'pred_geom': 0.0, 'refined_geom': 0.0, 'zeta': 0.0, 'kl': 0.0}
    for _ in range(int(gradient_steps)):
        batch = build_ralag_batch_from_replay(replay, model, audit_ensemble, batch_size=batch_size, device=device)
        metrics = train_ralag_step(model, batch=batch, optimizer=optimizer, kl_weight=kl_weight)
        for k in metric_sums:
            metric_sums[k] += float(metrics.get(k, 0.0))
    denom = max(int(gradient_steps), 1)
    return EpochSummary(
        collect_mean_return=float(collect_metrics['mean_return']),
        train_loss=metric_sums['loss'] / denom,
        pred_geom=metric_sums['pred_geom'] / denom,
        refined_geom=metric_sums['refined_geom'] / denom,
        zeta=metric_sums['zeta'] / denom,
        kl=metric_sums['kl'] / denom,
        replay_size=len(replay),
    )
