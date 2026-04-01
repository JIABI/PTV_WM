from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from fatewm.core.allocation import budgeted_allocation, topk_binary_mask
from fatewm.core.constraints import emb_constraint
from fatewm.core.risk_functional import (
    RiskSurrogateConfig,
    cvar_tail,
    decision_risk,
    fourier_time_embedding,
    sample_dmc_candidates,
    toy_oracle_action_costs,
)
from fatewm.core.utils import stopgrad


def z_features(algo, z0: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    scores0 = algo.scores(z0)
    top2 = torch.topk(scores0, k=min(2, scores0.shape[-1]), dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).abs() if top2.shape[-1] == 2 else top2[:, 0].abs()
    vuln = 1.0 / (margin.unsqueeze(-1) + eps)
    zn = torch.norm(z0, dim=-1, keepdim=True)
    zm = z0.mean(dim=-1, keepdim=True)
    zs = z0.std(dim=-1, keepdim=True)
    return torch.cat([zn, vuln, zm, zs], dim=-1)


def _log_tau_embed(deltas: Sequence[int], batch_size: int, device: torch.device, n_freq: int) -> torch.Tensor:
    tau = torch.tensor(list(deltas), dtype=torch.float32, device=device).view(1, -1, 1)
    log_tau = torch.log(tau.clamp_min(1.0))
    emb = fourier_time_embedding(log_tau, n_freq=n_freq)
    return emb.expand(batch_size, -1, -1)


def _pairwise_ranking_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # pred,target: [J]
    diff_p = pred.unsqueeze(0) - pred.unsqueeze(1)
    diff_t = target.unsqueeze(0) - target.unsqueeze(1)
    sign = torch.sign(diff_t)
    mask = sign.ne(0).float()
    loss = F.softplus(-sign * diff_p) * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def _fate_targets_from_mediator(psi: torch.Tensor) -> torch.Tensor:
    """Construct soft 4-way fate targets from horizon-wise mediator traces.

    psi: [B, J]
    Returns: [B, J, 4] corresponding to [dissipate, transport, amplify, persist].
    """
    B, J = psi.shape
    if J == 1:
        out = torch.zeros((B, J, 4), device=psi.device, dtype=psi.dtype)
        out[..., 3] = 1.0
        return out

    pad_l = psi[:, :1]
    pad_r = psi[:, -1:]
    x = torch.cat([pad_l, psi, pad_r], dim=1)
    g = 0.5 * (x[:, 2:] - x[:, :-2])
    c = x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]
    p = torch.abs(x[:, 2:] - x[:, 1:-1])

    s_dis = F.softplus(-g)
    s_tr = F.softplus(torch.abs(c))
    s_amp = F.softplus(g)
    s_per = psi * torch.exp(-torch.abs(c)) * F.softplus(1.0 - p)
    scores = torch.stack([s_dis, s_tr, s_amp, s_per], dim=-1)
    return torch.softmax(scores, dim=-1).detach()


def _rollout_energy_discrete(algo, obs0: torch.Tensor, actions: torch.Tensor, horizons: Sequence[int], use_slow: bool) -> torch.Tensor:
    """Return model energies [B, J, A] for repeated discrete actions."""
    B = obs0.shape[0]
    A = int(actions.numel())
    J = len(horizons)
    z0 = algo.encode(obs0)
    energies = torch.zeros((B, J, A), device=obs0.device)
    for a_idx, a in enumerate(actions.tolist()):
        z = z0
        acc = torch.zeros((B,), device=obs0.device)
        act = torch.full((B, 1), int(a), dtype=torch.long, device=obs0.device)
        max_h = int(max(horizons))
        cache = {}
        for step in range(1, max_h + 1):
            z = algo.predict(z, act, delta=1)
            step_cost = algo.slow_head(z).squeeze(-1) if use_slow else (-algo.reward(z).squeeze(-1))
            acc = acc + step_cost
            cache[step] = acc.clone()
        for j, h in enumerate(horizons):
            energies[:, j, a_idx] = cache[int(h)]
    return energies


def _teacher_energy_discrete(teacher_algo, obs0: torch.Tensor, actions: torch.Tensor, horizons: Sequence[int], use_slow: bool) -> torch.Tensor:
    with torch.no_grad():
        return _rollout_energy_discrete(teacher_algo, obs0, actions, horizons, use_slow=use_slow)


def _model_energy_continuous_constant(algo, obs0: torch.Tensor, candidates: torch.Tensor, horizons: Sequence[int]) -> torch.Tensor:
    """Return model energies [J, K] for a single sample and candidate set."""
    device = obs0.device
    if obs0.ndim == 1:
        obs0 = obs0.unsqueeze(0)
    z0 = algo.encode(obs0)
    K = int(candidates.shape[0])
    A = int(candidates.shape[1])
    z = z0.expand(K, -1)
    act = candidates.to(device).reshape(K, A).unsqueeze(1)
    max_h = int(max(horizons))
    acc = torch.zeros((K,), device=device)
    cache = {}
    for step in range(1, max_h + 1):
        z = algo.predict(z, act, delta=1)
        acc = acc + (-algo.reward(z).squeeze(-1))
        cache[step] = acc.clone()
    return torch.stack([cache[int(h)] for h in horizons], dim=0)


def _extract_extra_reference(extra_seq, horizons: Sequence[int], device: torch.device):
    if extra_seq is None:
        return None, None
    ref_by_sample = []
    cand_by_sample = []
    for b in range(len(extra_seq)):
        ex0 = extra_seq[b][0]
        if ex0 is None or not isinstance(ex0, dict):
            ref_by_sample.append(None)
            cand_by_sample.append(None)
            continue
        cands = ex0.get("candidates", None)
        ref = ex0.get("oracle_costs_by_tau", None)
        if ref is None and ex0.get("oracle_costs", None) is not None:
            base = torch.tensor(ex0["oracle_costs"], dtype=torch.float32, device=device)
            ref = torch.stack([base for _ in horizons], dim=0)
        elif ref is not None:
            ref = torch.tensor(ref, dtype=torch.float32, device=device)
        if cands is not None:
            cands = torch.tensor(cands, dtype=torch.float32 if np.asarray(cands).dtype.kind == 'f' else torch.long, device=device)
        ref_by_sample.append(ref)
        cand_by_sample.append(cands)
    return ref_by_sample, cand_by_sample


def _router_supervision(logits: torch.Tensor, risk_per_h: torch.Tensor, budget: int) -> tuple[torch.Tensor, float]:
    logits_mean = logits.mean(dim=0)
    risk_norm = risk_per_h / (risk_per_h.sum() + 1e-8)
    rank_loss = _pairwise_ranking_loss(logits_mean, risk_per_h.detach())
    pred_membership = torch.sigmoid(logits_mean)
    true_membership = topk_binary_mask(risk_per_h.detach(), int(budget))
    topk_loss = F.binary_cross_entropy(pred_membership, true_membership)
    pred_top = torch.topk(logits_mean, k=min(int(budget), logits_mean.numel())).indices
    true_top = torch.topk(risk_per_h.detach(), k=min(int(budget), risk_per_h.numel())).indices
    hit = len(set(pred_top.tolist()) & set(true_top.tolist())) / max(1, min(int(budget), logits_mean.numel()))
    tail_align = F.mse_loss(torch.softmax(logits_mean, dim=-1), risk_norm.detach())
    return rank_loss + topk_loss + tail_align, float(hit)


def _compute_rrrm_core(
    algo,
    router,
    obs_seq,
    act_seq,
    rew_seq,
    extra_seq,
    deltas,
    method_cfg,
    device,
    env_cfg=None,
    teacher_algo=None,
):
    obs0 = obs_seq[:, 0]
    z0 = algo.encode(obs0)
    B = obs0.shape[0]
    J = len(deltas)
    env_name = str(getattr(env_cfg, "name", "")) if env_cfg is not None else ""

    # ---------- Reference and model energies ----------
    use_slow_energy = env_name == "toy"
    ref_energy = None
    model_energy = None

    if env_name == "toy":
        A = int(getattr(algo, "action_dim", 3))
        actions = torch.arange(A, device=device, dtype=torch.long)
        obs_dim = int(getattr(env_cfg, "obs_dim", obs0.shape[-1]))
        slow_decay = float(getattr(env_cfg, "slow_decay", 0.995))
        fast_decay = float(getattr(env_cfg, "fast_decay", 0.6))
        event_threshold = float(getattr(env_cfg, "event_threshold", 1.5))
        fail_pen = float(method_cfg.get("oracle_failure_penalty", 5.0))
        ref_energy = torch.stack([
            toy_oracle_action_costs(
                obs0,
                action_dim=A,
                obs_dim=obs_dim,
                slow_decay=slow_decay,
                fast_decay=fast_decay,
                event_threshold=event_threshold,
                horizon=int(h),
                failure_penalty=fail_pen,
            )
            for h in deltas
        ], dim=1)
        model_energy = _rollout_energy_discrete(algo, obs0, actions, deltas, use_slow=True)

    elif env_name in ("procgen", "atari100k"):
        A = int(getattr(algo, "action_dim", 2))
        actions = torch.arange(A, device=device, dtype=torch.long)
        model_energy = _rollout_energy_discrete(algo, obs0, actions, deltas, use_slow=False)
        ref_by_sample, _ = _extract_extra_reference(extra_seq, deltas, device)
        if any(r is not None for r in ref_by_sample):
            ref_energy = torch.zeros_like(model_energy)
            for b in range(B):
                if ref_by_sample[b] is not None:
                    ref_energy[b] = ref_by_sample[b]
                elif teacher_algo is not None:
                    ref_energy[b:b+1] = _teacher_energy_discrete(teacher_algo, obs0[b:b+1], actions, deltas, use_slow=False)
                else:
                    ref_energy[b:b+1] = stopgrad(model_energy[b:b+1])
        elif teacher_algo is not None:
            ref_energy = _teacher_energy_discrete(teacher_algo, obs0, actions, deltas, use_slow=False)
        else:
            ref_energy = stopgrad(model_energy)

    elif env_name == "dmc":
        ref_by_sample, cand_by_sample = _extract_extra_reference(extra_seq, deltas, device)
        if not any(c is not None for c in cand_by_sample):
            # fallback: self-sampled candidates around zero / actor action
            cand_by_sample = []
            low = torch.tensor(env_cfg.action_bounds[0], dtype=torch.float32, device=device) if hasattr(env_cfg, 'action_bounds') else None
            high = torch.tensor(env_cfg.action_bounds[1], dtype=torch.float32, device=device) if hasattr(env_cfg, 'action_bounds') else None
            for b in range(B):
                prev = torch.zeros((algo.action_dim,), device=device)
                actor_action = None
                try:
                    aa = algo.act(obs0[b].detach().cpu().numpy(), eval_mode=True)
                    actor_action = torch.tensor(aa, dtype=torch.float32, device=device)
                except Exception:
                    pass
                if low is None or high is None:
                    low = -torch.ones((algo.action_dim,), device=device)
                    high = torch.ones((algo.action_dim,), device=device)
                cand_by_sample.append(sample_dmc_candidates(prev, num_candidates=int(method_cfg.get('num_candidates', 32)), sigma=float(method_cfg.get('candidate_sigma', 0.3)), action_low=low, action_high=high, actor_action=actor_action))
        maxK = max(int(c.shape[0]) for c in cand_by_sample if c is not None)
        model_energy = torch.full((B, J, maxK), 1e6, device=device)
        ref_energy = torch.full((B, J, maxK), 1e6, device=device)
        for b in range(B):
            cands = cand_by_sample[b]
            if cands is None:
                continue
            me = _model_energy_continuous_constant(algo, obs0[b:b+1], cands.float(), deltas)
            K = int(cands.shape[0])
            model_energy[b, :, :K] = me
            if ref_by_sample[b] is not None:
                re = ref_by_sample[b]
                if re.ndim == 1:
                    re = re.unsqueeze(0).expand(J, -1)
                ref_energy[b, :, :K] = re[:, :K]
            elif teacher_algo is not None:
                te = _model_energy_continuous_constant(teacher_algo, obs0[b:b+1], cands.float(), deltas)
                ref_energy[b, :, :K] = te
            else:
                ref_energy[b, :, :K] = stopgrad(me)
    else:
        # generic discrete fallback
        A = int(getattr(algo, 'action_dim', 2))
        actions = torch.arange(A, device=device, dtype=torch.long)
        model_energy = _rollout_energy_discrete(algo, obs0, actions, deltas, use_slow=False)
        ref_energy = _teacher_energy_discrete(teacher_algo, obs0, actions, deltas, use_slow=False) if teacher_algo is not None else stopgrad(model_energy)

    # ---------- Decision-criticality mediator ----------
    surrogate_cfg = RiskSurrogateConfig(
        kind=str(method_cfg.get("mediator", "margin")),
        tau_s=float(method_cfg.get("tau_s", 1.0)),
        kappa=float(method_cfg.get("tau_m", method_cfg.get("kappa", 0.25))),
    )
    psi = []
    for j in range(J):
        psi_j = decision_risk(-ref_energy[:, j, :], -model_energy[:, j, :], surrogate_cfg)
        psi.append(psi_j)
    psi = torch.stack(psi, dim=1)  # [B,J]
    risk_per_h = torch.stack([cvar_tail(psi[:, j], alpha=float(method_cfg.get("risk_alpha", 0.9))) for j in range(J)], dim=0)

    # ---------- Router / allocation ----------
    budget_B = int(method_cfg.get("B", min(8, J)))
    if router is not None:
        t_emb = _log_tau_embed(deltas, B, device, n_freq=int(method_cfg.get("time_emb_freq", 8)))
        logits, fate_probs, router_reg = router(z_features(algo, z0), t_emb)
        w = router.allocate(logits)
        router_loss, topb_hit = _router_supervision(logits, risk_per_h, budget_B)
        fate_targets = _fate_targets_from_mediator(psi)
        fate_loss = -(fate_targets * torch.log(fate_probs.clamp_min(1e-8))).sum(dim=-1).mean()
    else:
        logits = risk_per_h.unsqueeze(0).expand(B, -1)
        w = budgeted_allocation(logits, budget_B, temperature=float(method_cfg.get("selection_temperature", 0.5)), kind="sparsemax")
        router_reg = torch.tensor(0.0, device=device)
        router_loss = torch.tensor(0.0, device=device)
        fate_loss = torch.tensor(0.0, device=device)
        topb_hit = 1.0

    strong_loss = (stopgrad(w) * psi).sum(dim=-1).mean()
    weak_loss = psi.mean()

    dyn_loss = torch.tensor(0.0, device=device)
    for d in deltas:
        z_pred = algo.predict(z0, act_seq, delta=int(d))
        z_ref = algo.encode(obs_seq[:, int(d)]) if obs_seq.shape[1] > int(d) else stopgrad(z_pred)
        dyn_loss = dyn_loss + emb_constraint(z_pred, z_ref)
    dyn_loss = dyn_loss / float(max(1, len(deltas)))

    total = (
        float(method_cfg.get("lambda_strong", 1.0)) * strong_loss
        + float(method_cfg.get("lambda_weak", 0.1)) * weak_loss
        + float(method_cfg.get("lambda_dyn", 1.0)) * dyn_loss
        + float(method_cfg.get("lambda_router", 0.25)) * router_loss
        + float(method_cfg.get("lambda_fate", 0.05)) * fate_loss
        + router_reg
    )
    logs = {
        "loss_total": float(total.detach().cpu()),
        "loss_strong": float(strong_loss.detach().cpu()),
        "loss_weak": float(weak_loss.detach().cpu()),
        "loss_dyn": float(dyn_loss.detach().cpu()),
        "loss_router": float(router_loss.detach().cpu()),
        "loss_fate": float(fate_loss.detach().cpu()),
        "tail_psi": float(cvar_tail(psi.reshape(-1), alpha=float(method_cfg.get('risk_alpha', 0.9))).detach().cpu()),
        "topB_hit": float(topb_hit),
        "mean_sum_w": float(w.sum(dim=-1).mean().detach().cpu()),
        "budget_B": float(budget_B),
        "lambda_dual": float(getattr(router, 'lambda_dual', torch.tensor(0.0)).detach().cpu()) if router is not None else 0.0,
    }
    for j, d in enumerate(deltas):
        logs[f"risk_tau_{int(d)}"] = float(risk_per_h[j].detach().cpu())
        logs[f"alloc_tau_{int(d)}"] = float(w[:, j].mean().detach().cpu())
    return total, logs


def compute_rrrm_loss(algo, router, batch, deltas, method_cfg, device, env_cfg=None, teacher_algo=None):
    obs_seq, act_seq, rew_seq, done_seq, extra_seq = batch
    return _compute_rrrm_core(
        algo,
        router,
        obs_seq,
        act_seq,
        rew_seq,
        extra_seq,
        deltas,
        method_cfg,
        device,
        env_cfg=env_cfg,
        teacher_algo=teacher_algo,
    )


def compute_fatewm_loss(algo, router, batch, deltas, method_cfg, device, env_cfg=None, teacher_algo=None):
    # backward-compatible alias: the repository now uses the RRRM objective as
    # the main paper-aligned method, while preserving the older method name.
    return compute_rrrm_loss(algo, router, batch, deltas, method_cfg, device, env_cfg=env_cfg, teacher_algo=teacher_algo)


def compute_uniform_loss(algo, batch, deltas, method_cfg, device):
    obs_seq, act_seq, rew_seq, done_seq = batch[:4]
    z0 = algo.encode(obs_seq[:, 0])
    loss = torch.tensor(0.0, device=device)
    for d in deltas:
        z_pred = algo.predict(z0, act_seq, delta=int(d))
        z_ref = algo.encode(obs_seq[:, int(d)]) if obs_seq.shape[1] > int(d) else stopgrad(z_pred)
        loss = loss + emb_constraint(z_pred, z_ref)
    loss = loss / float(len(deltas))
    return loss, {"loss_total": float(loss.detach().cpu())}


def compute_freq_heuristic_loss(algo, batch, deltas, method_cfg, device):
    alpha = float(method_cfg.get("alpha", 0.05))
    weights = np.array([math.exp(-alpha * float(d)) for d in deltas], dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)
    obs_seq, act_seq, rew_seq, done_seq = batch[:4]
    z0 = algo.encode(obs_seq[:, 0])
    loss = torch.tensor(0.0, device=device)
    for w, d in zip(weights, deltas):
        z_pred = algo.predict(z0, act_seq, delta=int(d))
        z_ref = algo.encode(obs_seq[:, int(d)]) if obs_seq.shape[1] > int(d) else stopgrad(z_pred)
        loss = loss + float(w) * emb_constraint(z_pred, z_ref)
    return loss, {"loss_total": float(loss.detach().cpu())}


def compute_attention_only_loss(algo, gate, batch, deltas, method_cfg, device):
    obs_seq, act_seq, rew_seq, done_seq = batch[:4]
    z0 = algo.encode(obs_seq[:, 0])
    scores0 = algo.scores(z0)
    top2 = torch.topk(scores0, k=min(2, scores0.shape[-1]), dim=-1).values
    margin0 = (top2[:, 0] - top2[:, 1]).abs() if top2.shape[-1] == 2 else top2[:, 0].abs()
    v = 1.0 / (margin0.unsqueeze(-1) + 1e-6)
    a0 = torch.norm(z0, dim=-1, keepdim=True)
    state = z0.mean(dim=-1, keepdim=True)
    phi = torch.cat([a0, v, state], dim=-1)
    w = gate(phi, temperature=float(method_cfg.get("temperature", 1.0)))
    loss = torch.tensor(0.0, device=device)
    for j, d in enumerate(deltas):
        z_pred = algo.predict(z0, act_seq, delta=int(d))
        z_ref = algo.encode(obs_seq[:, int(d)]) if obs_seq.shape[1] > int(d) else stopgrad(z_pred)
        loss = loss + w[:, j].mean() * emb_constraint(z_pred, z_ref)
    return loss, {"loss_total": float(loss.detach().cpu())}


def compute_ms_jepa_uniform_loss(algo, batch, deltas, method_cfg, device):
    return compute_uniform_loss(algo, batch, deltas, method_cfg, device)
