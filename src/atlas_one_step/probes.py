from __future__ import annotations

import numpy as np
import torch


def _embed_pixel(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.adaptive_avg_pool2d(x, (8, 8)).flatten(1)


def _embed_perceptual(x: torch.Tensor) -> torch.Tensor:
    pooled = torch.nn.functional.avg_pool2d(x, 2, stride=2)
    return torch.cat([pooled.flatten(1), x.mean(dim=(-1, -2))], dim=1)


def _embed_ssl(x: torch.Tensor) -> torch.Tensor:
    g = x.mean(dim=(-1, -2))
    v = x.var(dim=(-1, -2))
    return torch.cat([g, v], dim=1)


def _nn_distance(target_feats: torch.Tensor, support_feats: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(target_feats, support_feats)
    return d.min(dim=1).values.mean()


def support_deviation(target: torch.Tensor, x0: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        pix = _nn_distance(_embed_pixel(target), _embed_pixel(x0))
        perc = _nn_distance(_embed_perceptual(target), _embed_perceptual(x0))
        ssl = _nn_distance(_embed_ssl(target), _embed_ssl(x0))
    return {
        'support_pix': float(pix.item()),
        'support_perc': float(perc.item()),
        'support_ssl': float(ssl.item()),
        'support_deviation': float((pix + perc + ssl).div(3.0).item()),
    }


def normal_burden(target: torch.Tensor, x0: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        feats = _embed_ssl(x0)
        feats = feats - feats.mean(0, keepdim=True)
        q = min(4, feats.shape[0], feats.shape[1])
        _, _, V = torch.pca_lowrank(feats, q=q)
        basis = V[:, : max(1, min(2, V.shape[1]))]
        delta = _embed_ssl(target) - _embed_ssl(x0)
        proj = delta @ basis @ basis.T
        tan = (proj.pow(2).sum(dim=1) / (delta.pow(2).sum(dim=1) + 1e-8)).mean()
        nor = 1.0 - tan
    return {'rho_tan': float(tan.item()), 'rho_nor': float(nor.item()), 'normal_burden': float(nor.item())}


def covariance_conditioning(target: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        feats = _embed_perceptual(target)
        feats = feats - feats.mean(0, keepdim=True)
        cov = feats.T @ feats / max(feats.shape[0] - 1, 1)
        eig = torch.linalg.eigvalsh(cov + 1e-5 * torch.eye(cov.shape[0], device=cov.device)).real.clamp_min(1e-8)
        cond = eig.max() / eig.min()
        anisotropy = eig.max() / eig.mean()
    return {
        'jacobian_norm_proxy': float(eig.max().item()),
        'anisotropy': float(anisotropy.item()),
        'conditioning': float(cond.item()),
        'covariance_conditioning': float(cond.item()),
    }


def relative_shift_and_sensitivity(pred: torch.Tensor, target: torch.Tensor, x0: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        delta = (pred - target).flatten(1)
        base = (target - x0).flatten(1)
        shift = delta.norm(dim=1).mean() / (base.norm(dim=1).mean() + 1e-8)
        sens = delta.abs().mean() / (target.flatten(1).abs().mean() + 1e-8)
    return {'relative_shift': float(shift.item()), 'prediction_sensitivity': float(sens.item())}


def pathology_score(probes: dict[str, float]) -> float:
    return float(
        0.22 * probes.get('support_deviation', probes.get('support_pix', 0.0))
        + 0.20 * probes.get('normal_burden', probes.get('rho_nor', 0.0))
        + 0.18 * np.log1p(probes.get('covariance_conditioning', probes.get('conditioning', 0.0)))
        + 0.20 * probes.get('relative_shift', 0.0)
        + 0.12 * probes.get('prediction_sensitivity', 0.0)
        + 0.08 * np.log1p(probes.get('early_grad_var', 0.0))
    )
