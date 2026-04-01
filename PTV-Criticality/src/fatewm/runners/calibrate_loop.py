"""Low-frequency calibration loop for FateClassifier (4-class fate)."""
import numpy as np
import torch
import torch.nn.functional as F

from fatewm.core.metrics import spearman_corr
from fatewm.core.constraints import soft_jaccard


def _latent_membership(z: torch.Tensor, theta: float = 0.0, tau: float = 0.5):
    return torch.sigmoid((z - theta) / max(tau, 1e-6))


@torch.no_grad()
def _fate_spectrum(algo, z0, act_seq, K: int, eta: float, theta: float, tau: float):
    """Compute gain, transport, dissipation, amplitude for a perturbation injected at z0."""
    device = z0.device
    B = z0.shape[0]
    eps = 0.05
    dz0 = torch.randn_like(z0)
    dz0 = eps * dz0 / (torch.norm(dz0, dim=-1, keepdim=True) + 1e-8)

    z_base = z0
    z_pert = z0 + dz0

    A0 = torch.norm(z_pert - z_base, dim=-1) + 1e-8
    A_sum = torch.zeros_like(A0)
    trans_sum = torch.zeros_like(A0)

    for k in range(K):
        if act_seq is None:
            a = torch.zeros((B, 1), dtype=torch.long, device=device)
        else:
            a = act_seq[:, k:k+1] if act_seq.shape[1] > k else act_seq[:, 0:1]

        z_base = algo.predict(z_base, a, delta=1)
        z_pert = algo.predict(z_pert, a, delta=1)

        Ak = torch.norm(z_pert - z_base, dim=-1)
        A_sum = A_sum + (eta ** k) * Ak

        pk = _latent_membership(z_base, theta, tau)
        qk = _latent_membership(z_pert, theta, tau)
        jac = soft_jaccard(pk, qk)
        trans_sum = trans_sum + (eta ** k) * (1.0 - jac) * torch.ones_like(A0)

    gain = (A_sum / A0)            # >1 amplification, <1 dissipation
    dissipation = torch.clamp(1.0 - gain, min=0.0)
    transport = trans_sum / (K + 1e-6)
    amplitude = A0
    return gain, transport, dissipation, amplitude


def _fate_label(gain, transport, amplitude, eps_g=0.05, tau_s=0.25, tau_a=0.08):
    """Return int labels in {0,1,2,3} for each sample."""
    lbl = torch.zeros_like(gain, dtype=torch.long)  # default dissipate

    # large amplitude
    lbl = torch.where(amplitude > tau_a, torch.full_like(lbl, 3), lbl)

    # amplify
    lbl = torch.where((gain > (1.0 + eps_g)) & (amplitude <= tau_a), torch.full_like(lbl, 2), lbl)

    # transport: near-neutral gain and high transport
    neutral = (gain >= (1.0 - eps_g)) & (gain <= (1.0 + eps_g))
    lbl = torch.where(neutral & (transport > tau_s) & (amplitude <= tau_a), torch.full_like(lbl, 1), lbl)
    return lbl


def calibrate_if_needed(cfg, components, rb, step: int, deltas):
    if cfg.method.name != "fatewm":
        return
    if not cfg.method.calibration.enabled:
        return
    fe = components.get("fate_estimator", None)
    if fe is None:
        return
    if len(rb) < int(cfg.method.calibration.batch):
        return

    device = components["device"]
    algo = components["algo"]
    batch = rb.sample(int(cfg.method.calibration.batch))
    obs_seq, act_seq, rew_seq, done_seq = batch

    obs_seq = torch.tensor(obs_seq, dtype=torch.float32, device=device)
    act_seq = torch.tensor(act_seq, dtype=torch.long, device=device)

    z0 = algo.encode(obs_seq[:, 0])

    K = int(getattr(cfg.fate, "K", 10))
    eta = float(getattr(cfg.fate, "eta", 0.9))
    theta = float(getattr(cfg.fate, "transport_theta", 0.0))
    tau = float(getattr(cfg.fate, "transport_tau", 0.5))

    gain, transport, dissipation, amplitude = _fate_spectrum(algo, z0, act_seq, K=K, eta=eta, theta=theta, tau=tau)
    y = _fate_label(gain, transport, amplitude)

    # Build phi (7-dim action-sensitive, delta=1)
    with torch.no_grad():
        scores0 = algo.scores(z0)
        top2 = torch.topk(scores0, k=min(2, scores0.shape[-1]), dim=-1).values
        margin0 = (top2[:, 0] - top2[:, 1]).abs() if top2.shape[-1] == 2 else top2[:, 0].abs()
        v = 1.0 / (margin0.unsqueeze(-1) + 1e-6)
        a0 = torch.norm(z0, dim=-1, keepdim=True)
        state = z0.mean(dim=-1, keepdim=True)

        a_first = act_seq[:, 0:1]
        z1 = algo.predict(z0, a_first, delta=1)
        scores1 = algo.scores(z1)
        top2_1 = torch.topk(scores1, k=min(2, scores1.shape[-1]), dim=-1).values
        margin1 = (top2_1[:, 0] - top2_1[:, 1]).abs() if top2_1.shape[-1] == 2 else top2_1[:, 0].abs()
        d_margin = (margin1 - margin0).unsqueeze(-1)
        d_state = (torch.norm(z1, dim=-1, keepdim=True) - torch.norm(z0, dim=-1, keepdim=True))

        delta = torch.ones((z0.shape[0], 1), device=device)
        logdelta = torch.zeros_like(delta)
        phi = torch.cat([a0, v, state, delta, logdelta, d_margin, d_state], dim=-1)

    logits = fe(phi)
    loss = F.cross_entropy(logits, y)

    opt = components.get("fate_optimizer", components.get("optimizer", None))
    if opt is None:
        return
    opt.zero_grad()
    loss.backward()
    opt.step()

    with torch.no_grad():
        p = torch.softmax(logits, dim=-1)
        I = p[:, 2] + 0.5 * p[:, 1]  # amplify + 0.5 transport
        rho = spearman_corr(I.detach().cpu().numpy().reshape(-1), gain.detach().cpu().numpy().reshape(-1))

    print(f"[calibrate@{step}] fate_ce={float(loss.detach().cpu()):.4f} spearman={rho:.3f} "
          f"gain={float(gain.mean().detach().cpu()):.3f} trans={float(transport.mean().detach().cpu()):.3f}")
