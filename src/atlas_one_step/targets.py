from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math
import numpy as np
import torch


@dataclass
class TargetSpec:
    family: str
    params: dict[str, Any]

    def complexity(self) -> float:
        if self.family == 'scheduled':
            return float(len(self.params.get('ax', [])) + len(self.params.get('bu', [])) + len(self.params.get('cr', [])))
        return float(len(self.params))


def _poly_basis(t: torch.Tensor, order: int) -> list[torch.Tensor]:
    return [t ** i for i in range(order)]


def _scheduled_coeffs(t: torch.Tensor, coeffs: list[float]) -> torch.Tensor:
    basis = _poly_basis(t, len(coeffs))
    out = torch.zeros_like(t)
    for c, b in zip(coeffs, basis):
        out = out + float(c) * b
    return out


def construct_target(spec: TargetSpec, primitives: dict[str, torch.Tensor]) -> torch.Tensor:
    fam = spec.family
    x0, u, r, eps = primitives['x0'], primitives['u_t'], primitives['r_t'], primitives['eps']
    batch_size = x0.shape[0]
    t = primitives.get('t_scalar', None)
    if t is None:
        alpha = primitives['alpha'].reshape(batch_size)
        t = (2 / math.pi) * torch.arccos(alpha.clamp(-1 + 1e-6, 1 - 1e-6))

    if fam == 'line_x0_u':
        a = float(spec.params['alpha'])
        y = a * x0 + (1 - a) * u
    elif fam == 'line_x0_r':
        a = float(spec.params['alpha'])
        y = a * x0 + (1 - a) * r
    elif fam == 'line_x0_eps':
        a = float(spec.params['alpha'])
        y = a * x0 + (1 - a) * eps
    elif fam == 'simplex':
        a = float(spec.params['alpha'])
        b = float(spec.params['beta'])
        c = float(spec.params['gamma'])
        y = a * x0 + b * u + c * r
    elif fam == 'scheduled':
        ax = _scheduled_coeffs(t, list(spec.params['ax']))
        bu = _scheduled_coeffs(t, list(spec.params['bu']))
        cr = _scheduled_coeffs(t, list(spec.params['cr']))
        for coeff_name, coeff in [('ax', ax), ('bu', bu), ('cr', cr)]:
            while coeff.ndim < x0.ndim:
                coeff = coeff.unsqueeze(-1)
            if coeff_name == 'ax':
                ax = coeff
            elif coeff_name == 'bu':
                bu = coeff
            else:
                cr = coeff
        y = ax * x0 + bu * u + cr * r
    else:
        raise ValueError(f'Unsupported target family: {fam}')

    scale = float(spec.params.get('scale', 1.0))
    bias = float(spec.params.get('bias', 0.0))
    y = scale * y + bias
    if not torch.isfinite(y).all():
        raise RuntimeError('Constructed target contains NaN/Inf')
    return y


def reconstruct_x0_from_target(spec: TargetSpec, target: torch.Tensor, primitives: dict[str, torch.Tensor]) -> torch.Tensor:
    alpha = primitives['alpha']
    sigma = primitives['sigma']
    xt = primitives['xt']
    scale = float(spec.params.get('scale', 1.0))
    bias = float(spec.params.get('bias', 0.0))
    y = (target - bias) / max(scale, 1e-6)

    def coeffs_line_x0_u(a: float):
        c_x0 = a - (1 - a) / sigma
        c_xt = (1 - a) * alpha / sigma
        return c_x0, c_xt

    def coeffs_line_x0_r(a: float):
        c_x0 = 1.0
        c_xt = -(1 - a)
        return c_x0, c_xt

    def coeffs_line_x0_eps(a: float):
        c_x0 = a - (1 - a) * alpha / sigma
        c_xt = (1 - a) / sigma
        return c_x0, c_xt

    fam = spec.family
    if fam == 'line_x0_u':
        c_x0, c_xt = coeffs_line_x0_u(float(spec.params['alpha']))
    elif fam == 'line_x0_r':
        c_x0, c_xt = coeffs_line_x0_r(float(spec.params['alpha']))
    elif fam == 'line_x0_eps':
        c_x0, c_xt = coeffs_line_x0_eps(float(spec.params['alpha']))
    elif fam == 'simplex':
        a = float(spec.params['alpha'])
        b = float(spec.params['beta'])
        c = float(spec.params['gamma'])
        c_x0 = a - b / sigma + c
        c_xt = b * alpha / sigma - c
    elif fam == 'scheduled':
        batch_size = xt.shape[0]
        t = primitives.get('t_scalar', None)
        if t is None:
            alpha_scalar = alpha.reshape(batch_size)
            t = (2 / math.pi) * torch.arccos(alpha_scalar.clamp(-1 + 1e-6, 1 - 1e-6))
        ax = _scheduled_coeffs(t, list(spec.params['ax']))
        bu = _scheduled_coeffs(t, list(spec.params['bu']))
        cr = _scheduled_coeffs(t, list(spec.params['cr']))
        while ax.ndim < xt.ndim:
            ax = ax.unsqueeze(-1)
            bu = bu.unsqueeze(-1)
            cr = cr.unsqueeze(-1)
        c_x0 = ax - bu / sigma + cr
        c_xt = bu * alpha / sigma - cr
    else:
        raise ValueError(f'Unsupported target family: {fam}')

    return (y - c_xt * xt) / (c_x0 + 1e-6)


def sample_target_specs(
    family: str,
    num_points: int,
    schedule_basis_order: int = 3,
    alpha_min: float = 0.05,
    alpha_max: float = 0.95,
    custom_alphas: list[float] | None = None,
) -> list[TargetSpec]:
    specs: list[TargetSpec] = []
    if family in {'line_x0_u', 'line_x0_r', 'line_x0_eps'}:
        alphas = custom_alphas if custom_alphas is not None else np.linspace(alpha_min, alpha_max, num_points)
        for alpha in alphas:
            specs.append(TargetSpec(family=family, params={'alpha': float(alpha)}))
    elif family == 'simplex':
        vals = np.linspace(0.0, 1.0, max(3, int(math.sqrt(num_points)) + 1))
        for a in vals:
            for b in vals:
                c = 1.0 - a - b
                if c < -1e-8:
                    continue
                specs.append(TargetSpec(family='simplex', params={'alpha': float(a), 'beta': float(b), 'gamma': float(max(c, 0.0))}))
        specs = specs[:max(num_points, min(len(specs), num_points))]
    elif family == 'scheduled':
        rng = np.random.default_rng(0)
        for _ in range(num_points):
            ax = rng.uniform(0.2, 1.0, size=schedule_basis_order).tolist()
            bu = rng.uniform(-0.5, 0.5, size=schedule_basis_order).tolist()
            cr = rng.uniform(-0.5, 0.5, size=schedule_basis_order).tolist()
            specs.append(TargetSpec(family='scheduled', params={'ax': ax, 'bu': bu, 'cr': cr}))
    else:
        raise ValueError(f'Unsupported family for sampling: {family}')
    return specs


def spec_to_dict(spec: TargetSpec) -> dict[str, Any]:
    return {'family': spec.family, 'params': spec.params}


def spec_from_dict(d: dict[str, Any]) -> TargetSpec:
    return TargetSpec(family=d['family'], params=d['params'])
