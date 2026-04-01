from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F

from ralagwm.chart.state import build_chart_state_from_audit
from ralagwm.data.batch import RALAGBatch, Transition
from ralagwm.geometry.extractor import extract_ralag_geometry
from ralagwm.geometry.losses import edge_sensitivity_error, geometry_distance, margin_error, score_field_error


def _as_tensor(x: Any, device: str | torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, device=device)


def _obs_type_from_model(model: Any) -> str:
    if hasattr(model, 'obs_type'):
        return str(model.obs_type)
    if hasattr(model, 'encoder'):
        name = model.encoder.__class__.__name__.lower()
        if 'image' in name or 'conv' in name:
            return 'image'
    return 'proprio'


def _prepare_obs_for_model(obs: torch.Tensor, obs_type: str) -> torch.Tensor:
    if obs_type == 'image':
        return obs.float()
    if obs.dim() > 2:
        return obs.view(obs.shape[0], -1).float()
    return obs.float()


def _prepare_single_obs_for_model(obs: Any, obs_type: str, device: str | torch.device) -> torch.Tensor:
    obs_t = _as_tensor(obs, device)
    if obs_type == 'image':
        if obs_t.dim() in (3, 4):
            obs_t = obs_t.unsqueeze(0)
        return obs_t.float()
    if obs_t.dim() == 1:
        obs_t = obs_t.unsqueeze(0)
    elif obs_t.dim() > 2:
        obs_t = obs_t.view(1, -1)
    return obs_t.float()


def _extract_logits_like(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    for key in ('deploy_logits', 'policy_logits', 'logits', 'scores', 'consensus_scores', 'prediction'):
        if hasattr(outputs, key):
            value = getattr(outputs, key)
            if value is not None:
                return value
        if isinstance(outputs, dict) and key in outputs:
            return outputs[key]
    if hasattr(outputs, 'pred_geometry') and outputs.pred_geometry is not None:
        return outputs.pred_geometry.centered_scores
    if hasattr(outputs, 'refined_geometry') and outputs.refined_geometry is not None:
        return outputs.refined_geometry.centered_scores
    raise ValueError('Could not extract action scores/logits from model outputs.')


def _env_random_action(env: Any):
    if hasattr(env, 'sample_random_action'):
        return env.sample_random_action()
    if hasattr(env, 'action_space'):
        return env.action_space.sample()
    raise AttributeError('Environment has no random action sampler.')


def _get_env_num_actions(env: Any) -> int | None:
    if hasattr(env, 'action_space') and hasattr(env.action_space, 'n'):
        try:
            return int(env.action_space.n)
        except Exception:
            pass
    spec = getattr(env, 'spec', None)
    if spec is not None:
        for key in ('num_actions', 'action_dim'):
            if hasattr(spec, key):
                try:
                    value = int(getattr(spec, key))
                    if value > 0:
                        return value
                except Exception:
                    pass
    wrapped = getattr(env, 'env', None)
    if wrapped is not None and wrapped is not env:
        return _get_env_num_actions(wrapped)
    return None


def _select_valid_discrete_action(scores: torch.Tensor, env: Any):
    num_actions = _get_env_num_actions(env)
    if num_actions is None or num_actions <= 0:
        return _env_random_action(env)
    vec = scores[0] if scores.dim() > 1 else scores
    vec = vec.view(-1)
    if vec.numel() != num_actions:
        return _env_random_action(env)
    action = int(torch.argmax(vec).item())
    return max(0, min(action, num_actions - 1))


def _replay_add(replay: Any, transition: Transition) -> None:
    if hasattr(replay, 'add'):
        replay.add(transition); return
    if hasattr(replay, 'push'):
        replay.push(transition); return
    if hasattr(replay, 'append'):
        replay.append(transition); return
    raise AttributeError('ReplayBuffer has no add/push/append method.')


@torch.no_grad()
def select_action_from_model(model: Any, obs: Any, env: Any, device: str | torch.device = 'cpu'):
    obs_type = _obs_type_from_model(model)
    obs_t = _prepare_single_obs_for_model(obs, obs_type=obs_type, device=device)
    outputs = model(obs_t)
    action_type = str(getattr(getattr(env, 'spec', None), 'action_type', 'discrete'))

    if action_type == 'discrete':
        logits = _extract_logits_like(outputs)
        action = _select_valid_discrete_action(logits, env)
        return action, outputs

    if hasattr(outputs, 'selected_action') and outputs.selected_action is not None:
        act = outputs.selected_action[0] if isinstance(outputs.selected_action, torch.Tensor) and outputs.selected_action.dim() > 1 else outputs.selected_action
        if isinstance(act, torch.Tensor):
            return act.detach().cpu().numpy(), outputs
        return act, outputs

    logits = _extract_logits_like(outputs)
    action_dim = int(getattr(getattr(env, 'spec', None), 'action_dim', 1))
    vec = logits[0] if logits.dim() > 1 else logits
    vec = vec.view(-1)
    if vec.numel() == action_dim:
        return vec.detach().cpu().numpy(), outputs
    return _env_random_action(env), outputs


@torch.no_grad()
def select_action_from_baseline(model: Any, obs: Any, env: Any, device: str | torch.device = 'cpu'):
    obs_type = _obs_type_from_model(model)
    obs_t = _prepare_single_obs_for_model(obs, obs_type=obs_type, device=device)
    outputs = model(obs_t)
    logits = _extract_logits_like(outputs)
    action_type = str(getattr(getattr(env, 'spec', None), 'action_type', 'discrete'))
    if action_type == 'discrete':
        return _select_valid_discrete_action(logits, env)
    action_dim = int(getattr(getattr(env, 'spec', None), 'action_dim', 1))
    vec = logits[0] if logits.dim() > 1 else logits
    vec = vec.view(-1)
    if vec.numel() == action_dim:
        return vec.detach().cpu().numpy()
    return _env_random_action(env)


def collect_rollout(env: Any, replay: Any, policy_fn: Callable[[Any], Any], episodes: int, max_steps: int, seed: int = 0) -> dict[str, float]:
    returns, lengths = [], []
    for ep in range(int(episodes)):
        obs, _ = env.reset(seed=int(seed) + ep)
        done = False
        truncated = False
        ep_ret = 0.0
        ep_len = 0
        while not done and not truncated and ep_len < int(max_steps):
            action = policy_fn(obs)
            step_out = env.step(action)
            if hasattr(step_out, 'observation'):
                next_obs, reward, done, info = step_out.observation, step_out.reward, step_out.done, step_out.info
                truncated = getattr(step_out, 'truncated', False)
            elif len(step_out) == 5:
                next_obs, reward, done, truncated, info = step_out
            else:
                next_obs, reward, done, info = step_out
                truncated = False
            _replay_add(replay, Transition(obs=torch.as_tensor(obs), action=torch.as_tensor(action), reward=torch.as_tensor(float(reward)), next_obs=torch.as_tensor(next_obs), done=torch.as_tensor(bool(done or truncated)), info=info if isinstance(info, dict) else {}))
            ep_ret += float(reward)
            ep_len += 1
            obs = next_obs
        returns.append(ep_ret)
        lengths.append(ep_len)
    return {'episodes': float(len(returns)), 'mean_return': float(sum(returns) / max(len(returns), 1)), 'mean_length': float(sum(lengths) / max(len(lengths), 1)), 'replay_size': float(len(replay))}


@torch.no_grad()
def _score_actions_with_audit(audit_ensemble: Any, x: torch.Tensor):
    return audit_ensemble(x.float())


def build_ralag_batch_from_replay(replay: Any, model: Any, audit_ensemble: Any, batch_size: int, device: str | torch.device = 'cpu') -> RALAGBatch:
    batch = replay.sample_tensors(batch_size=int(batch_size), device=device)
    raw_obs, raw_next_obs = batch['obs'], batch['next_obs']
    obs_type = _obs_type_from_model(model)
    obs_for_model = _prepare_obs_for_model(raw_obs, obs_type=obs_type)
    next_obs_for_model = _prepare_obs_for_model(raw_next_obs, obs_type=obs_type)
    audit_input = raw_next_obs.float() if obs_type == 'image' else next_obs_for_model.float()
    audit_scores = _score_actions_with_audit(audit_ensemble, audit_input)
    dones = batch.get('dones', torch.zeros(raw_obs.shape[0], device=device))

    chart_state = build_chart_state_from_audit(
        audit_scores=audit_scores,
        action=batch['actions'],
        action_type=getattr(model, 'action_type', 'discrete'),
        action_dim=int(getattr(model, 'action_dim', 1)),
        pool_budget=int(getattr(model, 'pool_budget', audit_scores.consensus_scores.shape[-1])),
        coord_dim=int(getattr(model, 'coord_dim', max(int(getattr(model, 'action_dim', 1)), 1))),
        chart_mode=str(getattr(model, 'chart_mode', 'discrete')),
    )
    chart = model.chart_generator.generate(chart_state, chart_state.boundary_saliency, chart_state.uncertainty)
    geometry_target = extract_ralag_geometry(chart_state.boundary_saliency, chart, chart_state.uncertainty)

    return RALAGBatch(
        obs=obs_for_model,
        action=batch['actions'],
        next_obs=next_obs_for_model,
        next_action=None,
        chart_state=chart_state,
        chart=chart,
        geometry_target=geometry_target,
        done=dones.view(-1),
        reward=batch['rewards'].view(-1),
        audit_scores=audit_scores,
        metadata={'obs_type': obs_type},
    )


def _match_tensor(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = a.float().view(a.shape[0], -1)
    b = b.float().view(b.shape[0], -1)
    width = min(a.shape[-1], b.shape[-1])
    return a[:, :width], b[:, :width]


def _chart_state_loss(pred_state, target_state) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    anchor_pred, anchor_tgt = _match_tensor(pred_state.anchor_action, target_state.anchor_action)
    metric_pred, metric_tgt = _match_tensor(pred_state.metric_matrix, target_state.metric_matrix)
    sal_pred, sal_tgt = _match_tensor(pred_state.boundary_saliency, target_state.boundary_saliency)
    unc_pred, unc_tgt = _match_tensor(pred_state.uncertainty, target_state.uncertainty)
    coord_pred, coord_tgt = _match_tensor(pred_state.action_coords, target_state.action_coords)
    losses = {
        'anchor': F.mse_loss(anchor_pred, anchor_tgt),
        'metric': F.mse_loss(metric_pred, metric_tgt),
        'saliency': F.mse_loss(sal_pred, sal_tgt),
        'uncertainty': F.mse_loss(unc_pred, unc_tgt),
        'coords': F.mse_loss(coord_pred, coord_tgt),
    }
    total = losses['anchor'] + losses['metric'] + losses['saliency'] + losses['uncertainty'] + losses['coords']
    return total, losses


def _multi_step_proxy_loss(outputs: Any, batch: RALAGBatch) -> torch.Tensor:
    if 'pred_next_obs' not in outputs.auxiliary:
        return torch.zeros((), device=batch.obs.device)
    pred = outputs.auxiliary['pred_next_obs'].float().view(batch.obs.shape[0], -1)
    target = batch.next_obs.float().view(batch.obs.shape[0], -1)
    width = min(pred.shape[-1], target.shape[-1])
    return F.mse_loss(pred[:, :width], target[:, :width])


def _geometry_loss_from_batch(outputs: Any, batch: RALAGBatch) -> tuple[torch.Tensor, dict[str, float]]:
    pred_geom = outputs.pred_geometry
    refined_geom = outputs.refined_geometry if outputs.refined_geometry is not None else pred_geom
    target_geom = batch.geometry_target
    geom_loss = geometry_distance(pred_geom, target_geom)
    refine_loss = geometry_distance(refined_geom, target_geom)
    zeta_loss, zeta_terms = _chart_state_loss(outputs.pred_chart_state, batch.chart_state)
    next_obs_aux = _multi_step_proxy_loss(outputs, batch)
    refine_gate = outputs.refinement_gate.float() if outputs.refinement_gate is not None else torch.zeros_like(target_geom.margin)
    oracle_trigger = ((target_geom.margin < getattr(outputs, 'margin_threshold', 0.2)) | (target_geom.boundary_risk > 0.6)).float() if target_geom is not None else torch.zeros_like(refine_gate)
    gate_loss = F.binary_cross_entropy(refine_gate.clamp(0.0, 1.0), oracle_trigger)
    total = geom_loss + refine_loss + zeta_loss + 0.25 * next_obs_aux + 0.10 * gate_loss
    return total, {
        'pred_geom': float(geom_loss.item()),
        'refined_geom': float(refine_loss.item()),
        'zeta': float(zeta_loss.item()),
        'zeta_anchor': float(zeta_terms['anchor'].item()),
        'zeta_metric': float(zeta_terms['metric'].item()),
        'zeta_saliency': float(zeta_terms['saliency'].item()),
        'zeta_uncertainty': float(zeta_terms['uncertainty'].item()),
        'zeta_coords': float(zeta_terms['coords'].item()),
        'score_error': float(score_field_error(refined_geom, target_geom).item()),
        'margin_error': float(margin_error(refined_geom, target_geom).item()),
        'edge_error': float(edge_sensitivity_error(refined_geom, target_geom).item()),
        'multi': float(next_obs_aux.item()),
        'gate': float(gate_loss.item()),
        'kl': float(outputs.losses.get('kl', torch.zeros((), device=batch.obs.device)).mean().item()) if hasattr(outputs, 'losses') else 0.0,
    }


def train_ralag_step(model: Any, batch: RALAGBatch, optimizer: torch.optim.Optimizer, kl_weight: float = 1e-3) -> dict[str, float]:
    optimizer.zero_grad()
    outputs = model(batch)
    loss, stats = _geometry_loss_from_batch(outputs, batch)
    if hasattr(outputs, 'losses') and isinstance(outputs.losses, dict):
        if 'kl' in outputs.losses and torch.is_tensor(outputs.losses['kl']):
            loss = loss + float(kl_weight) * outputs.losses['kl'].mean()
    loss.backward()
    optimizer.step()
    stats['loss'] = float(loss.item())
    return stats


@torch.no_grad()
def evaluate_dummy(env: Any, model: Any, episodes: int = 2, max_steps: int = 50, device: str | torch.device = 'cpu', seed: int = 0) -> dict[str, float]:
    class _TmpReplay(list):
        def add(self, x):
            self.append(x)
    replay = _TmpReplay()
    return collect_rollout(env=env, replay=replay, policy_fn=lambda obs: select_action_from_model(model, obs, env, device=device)[0], episodes=int(episodes), max_steps=int(max_steps), seed=int(seed))


def train_baseline_step(model: Any, obs: torch.Tensor, optimizer: torch.optim.Optimizer, target: torch.Tensor) -> dict[str, float]:
    optimizer.zero_grad()
    outputs = model(obs)
    pred = _extract_logits_like(outputs)
    if target.dtype in (torch.int32, torch.int64) and pred.dim() >= 2:
        loss = F.cross_entropy(pred, target.long())
    else:
        pred = pred.float().view(pred.shape[0], -1)
        target = target.float().view(target.shape[0], -1)
        min_dim = min(pred.shape[1], target.shape[1])
        loss = F.mse_loss(pred[:, :min_dim], target[:, :min_dim])
    loss.backward()
    optimizer.step()
    return {'loss': float(loss.item())}
