from __future__ import annotations

import numpy as np
import torch


def mse_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((x - y) ** 2).flatten(1).mean(dim=1)


def mae_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).abs().flatten(1).mean(dim=1)


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(mse.clamp_min(1e-8))


def feature_fd(real: torch.Tensor, fake: torch.Tensor) -> float:
    real_f = torch.nn.functional.adaptive_avg_pool2d(real, (4, 4)).flatten(1)
    fake_f = torch.nn.functional.adaptive_avg_pool2d(fake, (4, 4)).flatten(1)
    mu_r, mu_f = real_f.mean(0), fake_f.mean(0)
    var_r, var_f = real_f.var(0, unbiased=False), fake_f.var(0, unbiased=False)
    mean_term = (mu_r - mu_f).pow(2).mean()
    var_term = (torch.sqrt(var_r + 1e-8) - torch.sqrt(var_f + 1e-8)).pow(2).mean()
    return float((mean_term + var_term).item())


def summarize_tail(errors: torch.Tensor, collapse_threshold: float, percentiles: list[int]) -> dict[str, float]:
    err_np = errors.detach().cpu().numpy()
    out = {
        'worst_k_score': float(np.sort(err_np)[-max(1, len(err_np) // 10):].mean()),
        'rare_failure_rate': float((err_np > collapse_threshold).mean()),
        'mean': float(err_np.mean()),
        'std': float(err_np.std()),
    }
    out['cv'] = float(out['std'] / max(out['mean'], 1e-8))
    for p in percentiles:
        out[f'percentile_{p}'] = float(np.percentile(err_np, p))
    if 50 in percentiles and 90 in percentiles:
        out['tail_gap_90_50'] = float(out['percentile_90'] - out['percentile_50'])
    return out
