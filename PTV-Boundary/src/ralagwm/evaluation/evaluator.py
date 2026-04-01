"""Model loading, environment rosters, and rollout evaluation helpers."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ralagwm.audit.ensemble import AuditEnsemble
from ralagwm.audit.heads import build_audit_head_for_obs
from ralagwm.baselines import PolicyWM, RankWM, ReconWM, ValueWM
from ralagwm.chart.manual import ManualChartGenerator
from ralagwm.data.replay import ReplayBuffer
from ralagwm.envs import make_env
from ralagwm.geometry.losses import geometry_distance
from ralagwm.models.ralag_wm import RALAGWM
from ralagwm.training.checkpointing import load_checkpoint
from ralagwm.training.loops import build_ralag_batch_from_replay, collect_rollout, select_action_from_baseline, select_action_from_model
from ralagwm.utils.io import load_yaml, resolve_path

BASELINES = {
    'recon_wm': ReconWM,
    'value_wm': ValueWM,
    'policy_wm': PolicyWM,
    'rank_wm': RankWM,
}


def _deepcopy_cfg(cfg: Any) -> Any:
    return deepcopy(cfg)


def _set_attr(cfg: Any, key: str, value: Any) -> None:
    if isinstance(cfg, dict):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def _get_attr(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def infer_obs_dim(env_cfg: Any, env: Any | None = None) -> int:
    obs_type = str(_get_attr(env_cfg, 'obs_type', 'proprio'))
    if env is not None:
        spec = getattr(env, 'spec', None)
        obs_shape = tuple(getattr(spec, 'observation_shape', ()) or ()) if spec is not None else ()
        if obs_shape:
            if obs_type == 'image':
                prod = 1
                for v in obs_shape:
                    prod *= int(v)
                return int(prod)
            if len(obs_shape) == 1:
                return int(obs_shape[0])
            prod = 1
            for v in obs_shape:
                prod *= int(v)
            return int(prod)
    if obs_type == 'image':
        size = int(_get_attr(env_cfg, 'obs_size', _get_attr(env_cfg, 'obs_dim', 84)))
        channels = 1 if bool(_get_attr(env_cfg, 'grayscale_obs', False)) else 3
        frame_stack = int(_get_attr(env_cfg, 'frame_stack', 1))
        return channels * frame_stack * size * size
    return int(_get_attr(env_cfg, 'obs_dim', 16))


def load_manifest(manifest_path: str | Path | None) -> dict[str, Any] | None:
    if manifest_path is None:
        return None
    path = resolve_path(manifest_path)
    if not path.exists():
        return None
    return load_yaml(path)


def default_domain_roster() -> list[dict[str, Any]]:
    return [
        {'name': 'atari100k', 'env_id': 'ALE/Pong-v5', 'label': 'Atari-Pong', 'group': 'visual_discrete', 'random_score': -20.7, 'human_score': 14.6},
        {'name': 'atari100k', 'env_id': 'ALE/Breakout-v5', 'label': 'Atari-Breakout', 'group': 'visual_discrete', 'random_score': 1.7, 'human_score': 30.5},
        {'name': 'dmc_vision', 'domain_name': 'cartpole', 'task_name': 'swingup', 'label': 'DMC-Vision-Cartpole', 'group': 'visual_continuous'},
        {'name': 'dmc_vision', 'domain_name': 'walker', 'task_name': 'walk', 'label': 'DMC-Vision-Walker', 'group': 'visual_continuous'},
        {'name': 'dmc_proprio', 'domain_name': 'cartpole', 'task_name': 'swingup', 'label': 'DMC-Proprio-Cartpole', 'group': 'nonimage_continuous'},
        {'name': 'dmc_proprio', 'domain_name': 'cheetah', 'task_name': 'run', 'label': 'DMC-Proprio-Cheetah', 'group': 'nonimage_continuous'},
        {'name': 'crafter', 'env_id': 'CrafterReward-v1', 'label': 'Crafter', 'group': 'open_world'},
        {'name': 'highdim_continuous', 'env_id': 'myoFingerReachFixed-v0', 'label': 'MyoFingerReach', 'group': 'highdim_continuous'},
        {'name': 'procgen', 'env_id': 'procgen:procgen-coinrun-v0', 'label': 'Procgen-CoinRun', 'group': 'generalization'},
    ]


def resolve_domain_roster(cfg: Any) -> list[dict[str, Any]]:
    if str(_get_attr(_get_attr(cfg, 'env', {}), 'name', '')) == 'dummy':
        return [{'name': 'dummy', 'label': 'Dummy', 'group': 'dummy'}]
    manifest_path = _get_attr(_get_attr(cfg, 'experiment', {}), 'manifest_path', None)
    manifest = load_manifest(manifest_path)
    if manifest and manifest.get('domains'):
        return manifest['domains']
    return default_domain_roster()


def _load_named_env_defaults(name: str) -> dict[str, Any]:
    path = resolve_path(f"configs/env/{name}.yaml")
    if not path.exists():
        return {}
    payload = load_yaml(path)
    return payload if isinstance(payload, dict) else {}


def env_cfg_from_domain(base_cfg: Any, domain: dict[str, Any]) -> Any:
    cfg = _deepcopy_cfg(base_cfg)
    env = _get_attr(cfg, 'env')

    domain_name = str(domain.get('name', _get_attr(env, 'name', 'dummy')))
    defaults = _load_named_env_defaults(domain_name)
    for k, v in defaults.items():
        _set_attr(env, k, v)

    for k, v in domain.items():
        _set_attr(env, k, v)

    name = str(_get_attr(env, 'name', domain_name))
    if name == 'dmc_vision':
        _set_attr(env, 'obs_type', 'image')
        _set_attr(env, 'action_type', 'continuous')
        _set_attr(env, 'from_pixels', True)
    elif name == 'dmc_proprio':
        _set_attr(env, 'obs_type', 'proprio')
        _set_attr(env, 'action_type', 'continuous')
        _set_attr(env, 'from_pixels', False)
    elif name in {'atari100k', 'crafter', 'procgen'}:
        _set_attr(env, 'obs_type', 'image')
        _set_attr(env, 'action_type', 'discrete')
    elif name == 'highdim_continuous':
        _set_attr(env, 'obs_type', 'proprio')
        _set_attr(env, 'action_type', 'continuous')
    elif name == 'dummy':
        pass
    return cfg


def build_model_from_cfg(cfg, ckpt_path: str | None = None, device: torch.device | str = 'cpu'):
    env = make_env(cfg.env)
    obs_shape = env.spec.observation_shape
    obs_type = env.spec.obs_type
    action_dim = int(env.spec.action_dim)
    if obs_type == 'image':
        if len(obs_shape) == 4 and obs_shape[-1] in (1,3,4):
            image_size = int(obs_shape[1])
            image_channels = int(obs_shape[0] * obs_shape[-1])
        elif len(obs_shape) == 3 and obs_shape[-1] in (1,3,4):
            image_size = int(obs_shape[0])
            image_channels = int(obs_shape[-1])
        elif len(obs_shape) == 3:
            image_size = int(obs_shape[1])
            image_channels = int(obs_shape[0])
        else:
            image_size = int(_get_attr(cfg.env, 'obs_size', 64))
            image_channels = int(_get_attr(cfg.env, 'image_channels', 3))
    else:
        image_size = int(_get_attr(cfg.env, 'obs_size', 64))
        image_channels = 1
    obs_dim = infer_obs_dim(cfg.env, env)
    baseline_name = _get_attr(cfg, 'baseline', None)
    if baseline_name in BASELINES and ckpt_path and Path(ckpt_path).name != 'ralag_wm.pt':
        model = BASELINES[baseline_name](obs_dim=obs_dim, action_dim=action_dim, hidden_dim=int(cfg.model.encoder.hidden_dim), obs_type=obs_type, image_size=image_size, image_channels=image_channels).to(device)
        if ckpt_path and Path(ckpt_path).exists():
            load_checkpoint(ckpt_path, model, strict=False)
        env.close()
        return model, False
    model = RALAGWM(
        obs_type=obs_type,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=int(cfg.model.encoder.hidden_dim),
        latent_dim=int(cfg.model.bottleneck.latent_dim),
        backbone_kind='gru',
        deploy_kind=str(cfg.deploy.kind),
        image_size=image_size,
        image_channels=image_channels,
        chart_mode='discrete' if env.spec.action_type == 'discrete' else ('highdim_continuous' if action_dim > 16 else 'continuous'),
        chart_budget=int(_get_attr(cfg.chart, 'chart_budget', min(8, max(2, action_dim)))),
        pool_budget=int(_get_attr(cfg.chart, 'pool_budget', max(16, action_dim))),
    ).to(device)
    if ckpt_path and Path(ckpt_path).exists():
        load_checkpoint(ckpt_path, model, strict=False)
    env.close()
    return model, True


def build_audit_from_cfg(cfg: Any, device: torch.device | str = 'cpu', single: bool = False) -> AuditEnsemble:
    env = make_env(cfg.env)
    obs_shape = env.spec.observation_shape
    if str(getattr(env.spec, 'action_type', 'discrete')) == 'discrete':
        num_actions = int(env.spec.action_dim)
    else:
        num_actions = int(_get_attr(cfg.chart, 'pool_budget', max(16, int(env.spec.action_dim))))
    num = 1 if single else int(_get_attr(cfg.audit, 'num_audits', 4))
    heads = [build_audit_head_for_obs(obs_shape=obs_shape, num_actions=num_actions) for _ in range(num)]
    ensemble = AuditEnsemble(heads=heads, trim_ratio=float(_get_attr(cfg.audit, 'trim_ratio', 0.25))).to(device)
    ckpt = resolve_path('outputs/checkpoints/audit_ensemble.pt')
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device)
        state = payload.get('model') if isinstance(payload, dict) else None
        if state:
            try:
                ensemble.load_state_dict(state, strict=False)
            except Exception:
                pass
    ensemble.eval()
    env.close()
    return ensemble


def make_policy(model, is_ralag: bool, env, device: torch.device | str = 'cpu'):
    if is_ralag:
        return lambda obs: select_action_from_model(model, obs, env, device=device)[0]
    return lambda obs: select_action_from_baseline(model, obs, env, device=device)


def rollout_policy(env, policy_fn: Callable, episodes: int = 2, max_steps: int | None = None, seed: int | None = None) -> dict[str, float]:
    returns: list[float] = []
    lengths: list[int] = []
    flip = 0.0
    chattering = 0.0
    successes: list[float] = []
    achievement_counts: list[float] = []
    max_steps = int(max_steps or env.spec.max_episode_steps)
    for ep in range(int(episodes)):
        obs, _ = env.reset(seed=None if seed is None else seed + ep)
        prev_action = None
        prev_prev_action = None
        ep_return = 0.0
        ep_achievements = 0.0
        ep_success = 0.0
        for t in range(max_steps):
            action = policy_fn(obs)
            step = env.step(action)
            ep_return += float(step.reward)
            info = step.info or {}
            ach = info.get('achievements')
            if isinstance(ach, dict):
                ep_achievements = max(ep_achievements, float(sum(float(v) for v in ach.values())))
            elif isinstance(ach, (int, float)):
                ep_achievements = max(ep_achievements, float(ach))
            if info.get('success') is True:
                ep_success = 1.0
            elif isinstance(info.get('is_success'), (int, float, bool)):
                ep_success = max(ep_success, float(info.get('is_success')))
            if prev_prev_action is not None and prev_action is not None:
                if env.spec.action_type == 'discrete':
                    a = int(action); b = int(prev_action); c = int(prev_prev_action)
                    if a != b and a == c:
                        flip += 1.0
                        chattering += 1.0
                else:
                    a = np.asarray(action, dtype=np.float32)
                    b = np.asarray(prev_action, dtype=np.float32)
                    c = np.asarray(prev_prev_action, dtype=np.float32)
                    if np.linalg.norm(a - b) > 1e-3 and np.linalg.norm(a - c) < 1e-3:
                        flip += 1.0
                        chattering += 1.0
            prev_prev_action = prev_action
            prev_action = action
            obs = step.observation
            if step.done:
                lengths.append(t + 1)
                break
        else:
            lengths.append(max_steps)
        returns.append(ep_return)
        successes.append(ep_success)
        achievement_counts.append(ep_achievements)
    arr = np.asarray(returns, dtype=np.float32)
    cvar = float(np.mean(np.sort(arr)[: max(1, len(arr) // 10)])) if len(arr) else 0.0
    return {
        'episodes': int(episodes),
        'episode_returns': [float(v) for v in arr.tolist()],
        'episode_lengths': [int(v) for v in lengths],
        'mean_return': float(np.mean(arr)) if len(arr) else 0.0,
        'median_return': float(np.median(arr)) if len(arr) else 0.0,
        'mean_length': float(np.mean(lengths)) if lengths else 0.0,
        'flip_proxy': float(flip),
        'chattering_proxy': float(chattering),
        'cvar10': cvar,
        'success_rate': float(np.mean(successes)) if successes else 0.0,
        'achievement_score': float(np.mean(achievement_counts)) if achievement_counts else 0.0,
    }


def collect_analysis_batch(cfg: Any, model, audit, device: torch.device | str = 'cpu', batch_size: int = 8):
    env = make_env(cfg.env)
    replay = ReplayBuffer(capacity=max(64, batch_size * 4))
    collect_rollout(env, replay, policy_fn=lambda obs: env.sample_random_action(), episodes=1, max_steps=min(64, env.spec.max_episode_steps), seed=int(_get_attr(cfg, 'seed', 0)))
    batch = build_ralag_batch_from_replay(replay, model, audit, batch_size=min(batch_size, len(replay)), device=device)
    env.close()
    return batch


def build_oracle_geometry_from_batch(batch):
    return batch.geometry_target


def compute_future_fidelity_proxy(model, batch, baseline_name: str | None = None) -> float:
    with torch.no_grad():
        if baseline_name is not None and hasattr(model, 'forward') and not hasattr(model, 'deploy_head'):
            pred = model(batch.obs)['prediction'].reshape(batch.obs.shape[0], -1)
            target = batch.next_obs.reshape(batch.next_obs.shape[0], -1)
            width = min(pred.shape[-1], target.shape[-1])
            return float((pred[:, :width] - target[:, :width]).pow(2).mean().item())
        out = model(batch.obs)
        if 'pred_next_obs' in out.auxiliary:
            pred = out.auxiliary['pred_next_obs'].reshape(batch.obs.shape[0], -1)
            target = batch.next_obs.reshape(batch.next_obs.shape[0], -1)
            width = min(pred.shape[-1], target.shape[-1])
            return float((pred[:, :width] - target[:, :width]).pow(2).mean().item())
        return float(out.losses.get('kl', torch.zeros(1, device=batch.obs.device)).mean().item())


def compute_geometry_summary(model, batch) -> dict[str, float]:
    with torch.no_grad():
        out = model(batch.obs)
        pred = out.refined_geometry
        tgt = batch.geometry_target if getattr(batch, 'geometry_target', None) is not None else pred
        pe = pred.edge_sensitivity.reshape(pred.edge_sensitivity.shape[0], -1) if pred.edge_sensitivity.dim() > 1 else pred.edge_sensitivity.reshape(1, -1)
        te = tgt.edge_sensitivity.reshape(tgt.edge_sensitivity.shape[0], -1) if tgt.edge_sensitivity.dim() > 1 else tgt.edge_sensitivity.reshape(1, -1)
        min_edge = min(pe.shape[-1], te.shape[-1])
        return {
            'refined_geom_error': float(geometry_distance(pred, tgt).item()),
            'margin_error': float((pred.margin - tgt.margin).abs().mean().item()),
            'edge_sensitivity_error': float((pe[:, :min_edge] - te[:, :min_edge]).abs().mean().item()) if min_edge > 0 else 0.0,
            'top1_disagreement': float((pred.top_action_index != tgt.top_action_index).float().mean().item()),
            'boundary_risk_brier': float((pred.boundary_risk - tgt.boundary_risk).pow(2).mean().item()),
        }
