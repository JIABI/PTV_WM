import numpy as np
import torch

def episode_return(rews):
    return float(np.sum(rews))

def one_step_mse(pred, target):
    return float(torch.mean((pred - target) ** 2).detach().cpu())

def spearman_corr(x, y):
    x = np.asarray(x); y = np.asarray(y)
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    denom = (np.sqrt((rx**2).sum()) * np.sqrt((ry**2).sum()) + 1e-12)
    return float((rx*ry).sum() / denom)

def tail_mean(x, q=0.9):
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    thr = np.quantile(x, q)
    return float(x[x >= thr].mean()) if np.any(x >= thr) else float(x.mean())

def drift_metric(states, idx_slice=None):
    # drift proxy: mean L2 norm over selected slice of state (e.g., slow part)
    s = np.asarray(states)
    if s.ndim == 1:
        s = s.reshape(1, -1)
    if idx_slice is not None:
        s = s[:, idx_slice]
    return float(np.mean(np.linalg.norm(s, axis=-1)))

def switch_rate(actions):
    """Any action-change rate: P[a_t != a_{t-1}]."""
    a = np.asarray(actions).astype(int)
    if a.size < 2:
        return 0.0
    return float(np.mean(a[1:] != a[:-1]))


def aba_rate(actions):
    """Strict ABA (ping-pong) rate: P[a_t == a_{t-2} != a_{t-1}]."""
    a = np.asarray(actions).astype(int)
    if a.size < 3:
        return 0.0
    aba = 0
    for i in range(2, a.size):
        if a[i] == a[i - 2] and a[i] != a[i - 1]:
            aba += 1
    return float(aba / max(1, a.size - 2))


# Backwards-compatible alias: older logs used "ping_pong" for ABA.
def ping_pong_rate(actions):
    return aba_rate(actions)
