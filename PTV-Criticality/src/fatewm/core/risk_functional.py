"""Risk-functional components for NeurIPS-style budgeted multi-time learning.

This module implements:
  - continuous (normalized) time embedding
  - decision boundary risk surrogates (listwise CE and margin risk)
  - simple risk-shape proxies used by mixture-of-risk-experts
  - CVaR tail functional

Design goal: keep every piece auditable and differentiable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def fourier_time_embedding(log_tau: torch.Tensor, n_freq: int = 8) -> torch.Tensor:
    """Fourier features for a (log) continuous time coordinate.

    Args:
        log_tau: [..., 1] tensor
        n_freq: number of frequencies
    Returns:
        [..., 2*n_freq] embedding
    """
    # Frequencies spaced exponentially for multi-scale coverage.
    freqs = torch.exp(
        torch.linspace(0.0, math.log(2.0 ** (n_freq - 1)), n_freq, device=log_tau.device)
    )
    x = log_tau * freqs.view(*([1] * (log_tau.ndim - 1)), -1)
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


def _top2_margin(scores: torch.Tensor) -> torch.Tensor:
    """Return margin between top-1 and runner-up along last dim. Shape [..., 1]."""
    k = min(2, scores.shape[-1])
    topk = torch.topk(scores, k=k, dim=-1).values
    if k == 1:
        m = topk[..., 0]
    else:
        m = (topk[..., 0] - topk[..., 1]).abs()
    return m.unsqueeze(-1)


def decision_risk_listwise(scores_ref: torch.Tensor, scores_hat: torch.Tensor, tau_s: float = 1.0) -> torch.Tensor:
    """Listwise ranking risk surrogate.

    CE( softmax(s_ref/tau_s), softmax(s_hat/tau_s) ).
    Lower is better (higher ranking consistency).
    Returns per-sample risk: [B]
    """
    p = torch.softmax(scores_ref / float(tau_s), dim=-1).detach()  # reference distribution (stop-grad)
    logq = torch.log_softmax(scores_hat / float(tau_s), dim=-1)
    return -(p * logq).sum(dim=-1)


def decision_risk_margin(scores_ref: torch.Tensor, scores_hat: torch.Tensor, kappa: float = 0.25) -> torch.Tensor:
    """Margin-based flip risk surrogate.

    sigma((m_ref - m_hat)/kappa), where m is top1-runnerup margin.
    Returns per-sample risk: [B]
    """
    m_ref = _top2_margin(scores_ref).detach()
    m_hat = _top2_margin(scores_hat)
    x = (m_ref - m_hat) / float(kappa)
    return torch.sigmoid(x).squeeze(-1)


def transport_proxy(z_ref: torch.Tensor, z_hat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """A simple "support-shift" proxy for transporting fate.

    Uses cosine distance between feature-magnitude distributions.
    Returns per-sample: [B]
    """
    p = torch.abs(z_ref)
    q = torch.abs(z_hat)
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    q = q / (q.sum(dim=-1, keepdim=True) + eps)
    cos = (p * q).sum(dim=-1) / (torch.norm(p, dim=-1) * torch.norm(q, dim=-1) + eps)
    return (1.0 - cos).clamp_min(0.0)


def cvar_tail(x: torch.Tensor, alpha: float = 0.8) -> torch.Tensor:
    """CVaR_alpha of x along the batch dimension.

    x: [B] or [B, ...]
    Returns: scalar if x is [B], else [...]
    """
    if x.ndim == 1:
        B = x.shape[0]
        k = max(1, int(math.ceil((1.0 - float(alpha)) * B)))
        vals, _ = torch.topk(x, k=k, largest=True)
        return vals.mean()
    B = x.shape[0]
    rest = x.shape[1:]
    x2 = x.reshape(B, -1)
    k = max(1, int(math.ceil((1.0 - float(alpha)) * B)))
    vals, _ = torch.topk(x2, k=k, largest=True, dim=0)
    return vals.mean(dim=0).reshape(*rest)


@dataclass
class RiskSurrogateConfig:
    kind: str = "listwise"  # "listwise" or "margin"
    tau_s: float = 1.0
    kappa: float = 0.25


def decision_risk(scores_ref: torch.Tensor, scores_hat: torch.Tensor, cfg: RiskSurrogateConfig) -> torch.Tensor:
    if cfg.kind == "margin":
        return decision_risk_margin(scores_ref, scores_hat, kappa=cfg.kappa)
    return decision_risk_listwise(scores_ref, scores_hat, tau_s=cfg.tau_s)


# ---------------------------
# Oracle references (ToyEnv)
# ---------------------------

@torch.no_grad()
def toy_oracle_next_slow_norm(
    obs: torch.Tensor,
    action: torch.Tensor,
    *,
    obs_dim: int,
    noise_std: float,
    slow_decay: float,
    fast_decay: float,
) -> torch.Tensor:
    """Deterministic one-step slow-norm oracle for ToyEnv.

    This matches ToyEnv.step() with noise set to 0. Used to build an auditable
    action ranking reference s*(a) without relying on any learned head.

    Args:
        obs: [B,obs_dim]
        action: [B] int
    Returns:
        slow_norm_next: [B]
    """
    assert obs.shape[-1] == obs_dim
    half = obs_dim // 2
    x_fast = obs[:, :half]
    x_slow = obs[:, half:]
    a = action.to(obs.device).float()

    # Same action-effect mapping as ToyEnv.
    a_eff_fast = (a - 1.0) * 0.15
    a_eff_slow = (1.0 - (a - 1.0).abs()) * 0.05

    x_fast_next = fast_decay * x_fast + a_eff_fast.unsqueeze(-1)
    coupling = x_fast_next.abs().mean(dim=-1, keepdim=True)
    x_slow_next = slow_decay * x_slow + 0.02 * coupling + a_eff_slow.unsqueeze(-1)
    return torch.norm(x_slow_next, dim=-1)


@torch.no_grad()
def toy_oracle_action_costs(
    obs: torch.Tensor,
    *,
    action_dim: int,
    obs_dim: int,
    slow_decay: float,
    fast_decay: float,
    event_threshold: float,
    horizon: int = 10,
    failure_penalty: float = 5.0,
) -> torch.Tensor:
    """Oracle multi-step cost J*(a) for each candidate action in ToyEnv.

    We unroll a *deterministic* horizon with constant action a. This is cheap
    (A<=5, H<=20) and produces a strong, auditable ranking reference.

    Cost = sum_k slow_norm_k + failure_penalty * I(any slow_norm_k > threshold).

    Returns:
        costs: [B, action_dim]
    """
    B = obs.shape[0]
    device = obs.device
    costs = torch.zeros((B, action_dim), device=device)
    for a in range(action_dim):
        x = obs.clone()
        fail = torch.zeros((B,), device=device)
        acc = torch.zeros((B,), device=device)
        a_t = torch.full((B,), float(a), device=device)
        for _ in range(int(horizon)):
            sn = toy_oracle_next_slow_norm(
                x,
                a_t,
                obs_dim=obs_dim,
                noise_std=0.0,
                slow_decay=slow_decay,
                fast_decay=fast_decay,
            )
            acc = acc + sn
            fail = torch.maximum(fail, (sn > float(event_threshold)).float())
            # Update full state deterministically.
            half = obs_dim // 2
            x_fast = x[:, :half]
            x_slow = x[:, half:]
            a_eff_fast = (a_t - 1.0) * 0.15
            a_eff_slow = (1.0 - (a_t - 1.0).abs()) * 0.05
            x_fast_next = fast_decay * x_fast + a_eff_fast.unsqueeze(-1)
            coupling = x_fast_next.abs().mean(dim=-1, keepdim=True)
            x_slow_next = slow_decay * x_slow + 0.02 * coupling + a_eff_slow.unsqueeze(-1)
            x = torch.cat([x_fast_next, x_slow_next], dim=-1)
        costs[:, a] = acc + float(failure_penalty) * fail
    return costs


# ---------------------------
# Oracle references (DMCEnv)
# ---------------------------

def sample_dmc_candidates(
    prev_action: torch.Tensor,
    *,
    num_candidates: int,
    sigma: float,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    include_zero: bool = True,
    actor_action: torch.Tensor | None = None,
    sigma_large: float | None = None,
    frac_large: float = 0.25,
    include_uniform: bool = True,
) -> torch.Tensor:
    """Sample a candidate action set for continuous-control tasks.

    Candidate set includes:
      - previous action (continuity anchor)
      - optional actor action (proposal anchor)
      - optional zero action (conservative anchor)
      - a mixture of local and larger perturbations (multi-scale coverage)
      - optional uniform random actions (coverage fallback)

    This supports a unified energy interface across tasks without manual
    discrete/continuous switching.
    """
    A = int(prev_action.shape[-1])
    K = int(num_candidates)
    out = [prev_action]
    if actor_action is not None:
        out.append(actor_action)
    if include_zero:
        out.append(torch.zeros_like(prev_action))

    n_rem = max(0, K - len(out))
    n_uni = int(round(0.25 * n_rem)) if include_uniform else 0
    n_noise = max(0, n_rem - n_uni)

    n_large = 0
    if (sigma_large is not None) and (n_noise > 0):
        n_large = int(round(float(frac_large) * n_noise))
    n_small = max(0, n_noise - n_large)

    center = actor_action if actor_action is not None else prev_action

    if n_small > 0:
        eps = float(sigma) * torch.randn((n_small, A), device=prev_action.device, dtype=prev_action.dtype)
        cand = center.unsqueeze(0) + eps
        cand = torch.max(torch.min(cand, action_high), action_low)
        out.append(cand)

    if n_large > 0:
        eps = float(sigma_large) * torch.randn((n_large, A), device=prev_action.device, dtype=prev_action.dtype)
        cand = center.unsqueeze(0) + eps
        cand = torch.max(torch.min(cand, action_high), action_low)
        out.append(cand)

    if n_uni > 0:
        uni = torch.rand((n_uni, A), device=prev_action.device, dtype=prev_action.dtype)
        cand = action_low + (action_high - action_low) * uni
        out.append(cand)

    candidates = torch.cat([x.unsqueeze(0) if x.ndim == 1 else x for x in out], dim=0)
    return candidates[:K]


def refine_action_by_energy_gd(
    algo,
    obs: torch.Tensor,
    action_init: torch.Tensor,
    *,
    device: torch.device,
    steps: int = 2,
    step_size: float = 0.1,
    action_low: torch.Tensor | None = None,
    action_high: torch.Tensor | None = None,
    grad_clip: float = 10.0,
) -> tuple[torch.Tensor, float]:
    """Refine a continuous action by gradient descent on the learned energy.

    Energy is defined as negative predicted reward for the next-step latent.
    Refinement is used only when the candidate ranking is near-tied.
    """
    if obs.ndim == 1:
        obs = obs.unsqueeze(0)
    obs = obs.to(device=device, dtype=torch.float32)
    a = action_init.to(device=device, dtype=torch.float32).clone().detach().requires_grad_(True)

    if action_low is not None:
        action_low = action_low.to(device=device, dtype=torch.float32)
    if action_high is not None:
        action_high = action_high.to(device=device, dtype=torch.float32)

    for _ in range(int(max(1, steps))):
        z0 = algo.encode(obs)
        z1 = algo.predict(z0, a.view(1, 1, -1), delta=1)
        r_hat = algo.reward(z1).view(())
        energy = -r_hat
        (g,) = torch.autograd.grad(energy, a, retain_graph=False, create_graph=False)
        if grad_clip is not None and float(grad_clip) > 0:
            g = torch.clamp(g, -float(grad_clip), float(grad_clip))
        a = (a - float(step_size) * g).detach().requires_grad_(True)
        if action_low is not None and action_high is not None:
            a = torch.max(torch.min(a, action_high), action_low).detach().requires_grad_(True)

    with torch.no_grad():
        z0 = algo.encode(obs)
        z1 = algo.predict(z0, a.view(1, 1, -1), delta=1)
        e = float((-algo.reward(z1).view(())).detach().cpu())
    return a.detach(), e


def refine_action_by_energy_mirror(
    algo,
    obs: torch.Tensor,
    candidates: torch.Tensor,
    *,
    device: torch.device,
    energies: torch.Tensor | None = None,
    steps: int = 1,
    eta: float = 10.0,
    temperature: float = 1.0,
    action_low: torch.Tensor | None = None,
    action_high: torch.Tensor | None = None,
    resample: bool = False,
    resample_num: int = 0,
    resample_sigma: float = 0.1,
) -> tuple[torch.Tensor, float]:
    """Refine a continuous action by mirror-descent / exponentiated reweighting.

    This is a differentiable alternative to action-gradient descent refinement.
    We refine in *distribution space* over a candidate set rather than taking
    gradients w.r.t. the action itself. A single mirror step corresponds to
    an exponentiated-weights update:

        q_{k+1}(a) ∝ q_k(a) * exp(-eta * E(a))

    where E(a) is the learned energy. We output the refined action as the
    weighted mean of candidates.

    Practical notes:
      - More stable early in training than ∇_a E, since it only needs
        relative energy ordering.
      - Differentiable w.r.t. model parameters θ (through E_θ), while
        avoiding potentially noisy ∇_a E.
    """
    if obs.ndim == 1:
        obs = obs.unsqueeze(0)
    obs = obs.to(device=device, dtype=torch.float32)
    cand = candidates.to(device=device, dtype=torch.float32)
    if cand.ndim != 2:
        cand = cand.view(cand.shape[0], -1)

    if action_low is not None:
        action_low = action_low.to(device=device, dtype=torch.float32)
    if action_high is not None:
        action_high = action_high.to(device=device, dtype=torch.float32)

    def _energy_for_actions(a_mat: torch.Tensor) -> torch.Tensor:
        z0 = algo.encode(obs)  # [1, latent]
        Kc = int(a_mat.shape[0])
        z0_rep = z0.expand(Kc, -1)
        z1 = algo.predict(z0_rep, a_mat.view(Kc, 1, -1), delta=1)
        r_hat = algo.reward(z1).view(Kc)
        return -r_hat

    if energies is None:
        E = _energy_for_actions(cand)
    else:
        E = energies.to(device=device, dtype=torch.float32).view(-1)
        if E.shape[0] != cand.shape[0]:
            E = _energy_for_actions(cand)

    a_mean = cand[torch.argmin(E)].detach()
    for _ in range(int(max(1, steps))):
        scale = float(eta) / max(float(temperature), 1e-6)
        w = torch.softmax((-scale) * E, dim=0)  # [K]
        a_mean = (w.unsqueeze(-1) * cand).sum(dim=0)
        if action_low is not None and action_high is not None:
            a_mean = torch.max(torch.min(a_mean, action_high), action_low)

        if bool(resample) and int(resample_num) > 0:
            A = int(cand.shape[1])
            eps = float(resample_sigma) * torch.randn((int(resample_num), A), device=device, dtype=torch.float32)
            new_cand = a_mean.unsqueeze(0) + eps
            if action_low is not None and action_high is not None:
                new_cand = torch.max(torch.min(new_cand, action_high), action_low)
            Kkeep = max(1, min(int(cand.shape[0]), int(cand.shape[0]) - int(resample_num)))
            keep_idx = torch.topk(-E, k=Kkeep, largest=True).indices
            cand = torch.cat([a_mean.unsqueeze(0), cand[keep_idx], new_cand], dim=0)
            E = _energy_for_actions(cand)

    with torch.no_grad():
        e_ref = float(_energy_for_actions(a_mean.view(1, -1)).view(()).detach().cpu())
    return a_mean.detach(), e_ref


def dmc_oracle_action_costs(
    env,
    candidates: torch.Tensor,
    *,
    horizon: int = 1,
    failure_penalty: float = 0.0,
    bad_event_fn=None,
) -> torch.Tensor:
    """Compute an auditable oracle cost for each candidate action.

    We snapshot the simulator physics state, then for each candidate action we
    restore the state and simulate a short horizon with constant action.

    Default horizon=1 keeps this practical for online collection.

    Returns:
        costs: [K]
    """
    # best-effort: requires DMCEnv.get_state/set_state.
    state0 = env.get_state()
    t0 = getattr(env, "t", 0)
    costs = []
    with torch.no_grad():
        for a in candidates:
            env.set_state(state0)
            env.t = t0
            total = 0.0
            bad = 0.0
            for _ in range(int(horizon)):
                _, r, done, _info = env.step(a.detach().cpu().numpy())
                total += float(-r)
                if bad_event_fn is not None:
                    bad = max(bad, float(bool(bad_event_fn(env))))
                if done:
                    break
            total += float(failure_penalty) * bad
            costs.append(total)
    # restore
    env.set_state(state0)
    env.t = t0
    return torch.tensor(costs, dtype=torch.float32)


def procgen_oracle_action_costs(
    env,
    action_dim: int,
    *,
    horizon: int = 1,
    failure_penalty: float = 0.0,
    bad_event_fn=None,
) -> torch.Tensor:
    """Auditable oracle costs for Procgen (discrete actions).

    We snapshot env state via ProcgenEnv.get_state/set_state. For each action a,
    restore the snapshot and simulate a short horizon with constant action.

    Returns:
        costs: [A]
    """
    # NOTE: Procgen rewards (e.g., coinrun) are often sparse; constant-action
    # short-horizon rollouts can produce nearly-uniform costs, yielding a weak
    # reference distribution. We therefore support a simple MC continuation:
    # execute the candidate action once, then follow a random policy for the
    # remaining horizon. This preserves auditability while producing a more
    # discriminative p*(a|x) early in training.

    state0 = env.get_state()
    t0 = getattr(env, "t", 0)

    # Optional knobs (may be injected by the train loop).
    oracle_particles = int(getattr(env, "oracle_particles", 1))
    oracle_continuation = str(getattr(env, "oracle_continuation", "constant"))

    costs = []
    with torch.no_grad():
        for a in range(int(action_dim)):
            particle_costs = []
            for _p in range(max(1, oracle_particles)):
                env.set_state(state0)
                env.t = t0
                total = 0.0
                bad = 0.0

                # First step: apply the candidate action.
                _o, r, done, _info = env.step(int(a))
                total += float(-r)
                if bad_event_fn is not None:
                    bad = max(bad, float(bool(bad_event_fn(env))))

                # Remaining steps: either repeat (constant) or random continuation.
                for _k in range(int(horizon) - 1):
                    if done:
                        break
                    if oracle_continuation == "random_after_first":
                        aa = int(np.random.randint(int(action_dim)))
                    else:
                        aa = int(a)
                    _o, r, done, _info = env.step(int(aa))
                    total += float(-r)
                    if bad_event_fn is not None:
                        bad = max(bad, float(bool(bad_event_fn(env))))

                total += float(failure_penalty) * bad
                particle_costs.append(total)

            # Conservative (optimistic) oracle: take best particle cost.
            costs.append(float(np.min(particle_costs)) if particle_costs else 0.0)

    env.set_state(state0)
    env.t = t0
    return torch.tensor(costs, dtype=torch.float32)
