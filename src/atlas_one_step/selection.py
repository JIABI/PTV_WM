from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd

from .targets import TargetSpec, sample_target_specs, spec_to_dict
from .utils import save_json


def _as_list(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def semantic_gap(candidate: TargetSpec, loss_target: TargetSpec) -> float:
    if candidate.family != loss_target.family:
        # bounded but non-zero mismatch when families differ
        return 1.0 + 0.05 * abs(candidate.complexity() - loss_target.complexity())

    gap = 0.0
    keys = set(candidate.params) | set(loss_target.params)
    for k in keys:
        a = _as_list(candidate.params.get(k, 0.0))
        b = _as_list(loss_target.params.get(k, 0.0))
        m = max(len(a), len(b))
        a = a + [0.0] * (m - len(a))
        b = b + [0.0] * (m - len(b))
        gap += float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
    return float(gap)


def _scheduled_stats(spec: TargetSpec) -> dict[str, float]:
    ax = np.asarray(_as_list(spec.params.get('ax', [0.0])), dtype=float)
    bu = np.asarray(_as_list(spec.params.get('bu', [0.0])), dtype=float)
    cr = np.asarray(_as_list(spec.params.get('cr', [0.0])), dtype=float)
    coeff = np.stack([ax, bu, cr], axis=0)
    smooth = float(np.abs(np.diff(coeff, axis=1)).mean()) if coeff.shape[1] > 1 else 0.0
    x0_share = float(np.clip(ax.mean(), 0.0, 1.0))
    ut_share = float(np.clip(np.abs(bu).mean(), 0.0, 1.5))
    rt_share = float(np.clip(np.abs(cr).mean(), 0.0, 1.5))
    return {
        'smooth': smooth,
        'x0_share': x0_share,
        'ut_share': ut_share,
        'rt_share': rt_share,
    }


def _pseudo_diag_features(spec: TargetSpec, diag_cols: list[str]) -> dict[str, float]:
    fam = spec.family
    if fam == 'line_x0_u':
        alpha = float(spec.params.get('alpha', 0.5))
        # U-shaped brittleness around intermediate alpha.
        hump = float(np.exp(-((alpha - 0.55) ** 2) / 0.025))
        base = 0.18 + 0.55 * hump + 0.07 * (1.0 - alpha)
        rho = 0.20 + 0.65 * hump
        cond = 1.2 + 4.0 * hump
        rel = 0.10 + 0.60 * hump
        sens = 0.10 + 0.55 * hump
        early = 0.15 + 0.50 * hump
    elif fam == 'line_x0_r':
        alpha = float(spec.params.get('alpha', 0.5))
        hump = float(np.exp(-((alpha - 0.50) ** 2) / 0.035))
        base = 0.16 + 0.40 * hump
        rho = 0.18 + 0.45 * hump
        cond = 1.1 + 2.8 * hump
        rel = 0.08 + 0.40 * hump
        sens = 0.08 + 0.35 * hump
        early = 0.12 + 0.30 * hump
    elif fam == 'line_x0_eps':
        alpha = float(spec.params.get('alpha', 0.5))
        hump = float(np.exp(-((alpha - 0.45) ** 2) / 0.06))
        base = 0.35 + 0.40 * hump + 0.10 * (1.0 - alpha)
        rho = 0.30 + 0.35 * hump
        cond = 1.6 + 3.0 * hump
        rel = 0.18 + 0.45 * hump
        sens = 0.20 + 0.40 * hump
        early = 0.18 + 0.35 * hump
    elif fam == 'simplex':
        a = float(spec.params.get('alpha', 0.33))
        b = float(spec.params.get('beta', 0.33))
        c = float(spec.params.get('gamma', 0.34))
        interior = 1.0 - max(a, b, c)
        base = 0.18 + 0.55 * interior + 0.25 * c
        rho = 0.16 + 0.70 * interior + 0.15 * c
        cond = 1.1 + 4.0 * interior
        rel = 0.10 + 0.55 * interior
        sens = 0.10 + 0.45 * interior
        early = 0.12 + 0.35 * interior
    elif fam == 'scheduled':
        stats = _scheduled_stats(spec)
        rt_share = stats['rt_share']
        ut_share = stats['ut_share']
        smooth = stats['smooth']
        x0_share = stats['x0_share']
        # smooth x0->ut schedules with limited rt pulse are easiest.
        base = 0.16 + 0.20 * smooth + 0.18 * rt_share + 0.04 * max(0.0, 0.45 - x0_share)
        rho = 0.14 + 0.18 * smooth + 0.32 * rt_share + 0.06 * max(0.0, ut_share - 0.9)
        cond = 1.1 + 1.8 * smooth + 1.2 * rt_share
        rel = 0.08 + 0.22 * smooth + 0.18 * rt_share
        sens = 0.08 + 0.18 * smooth + 0.20 * rt_share
        early = 0.10 + 0.16 * smooth + 0.16 * rt_share
    else:
        base, rho, cond, rel, sens, early = 0.5, 0.5, 2.0, 0.5, 0.5, 0.5

    pathology = 0.22 * base + 0.20 * rho + 0.18 * np.log1p(cond) + 0.20 * rel + 0.12 * sens + 0.08 * np.log1p(early)
    pseudo = {
        'pathology.support_pix': float(base),
        'pathology.support_perc': float(base),
        'pathology.support_ssl': float(base),
        'pathology.support_deviation': float(base),
        'pathology.rho_nor': float(rho),
        'pathology.normal_burden': float(rho),
        'pathology.conditioning': float(cond),
        'pathology.covariance_conditioning': float(cond),
        'pathology.relative_shift': float(rel),
        'pathology.prediction_sensitivity': float(sens),
        'pathology.pathology_score': float(pathology),
        'grad_var': float(early),
    }
    return {c: float(pseudo.get(c, 0.0)) for c in diag_cols}


def rank_candidates(
    candidates: list[TargetSpec],
    surrogate_path: str | Path,
    loss_target: TargetSpec,
    semantic_gap_weight: float,
    complexity_weight: float,
) -> list[dict[str, Any]]:
    artifact = joblib.load(surrogate_path)
    clf = artifact['clf_diag']
    reg = artifact.get('reg_diag')
    diag_cols = artifact['diag_cols']
    rows = []
    for spec in candidates:
        X = pd.DataFrame([_pseudo_diag_features(spec, diag_cols)])
        proba = clf.predict_proba(X)[0]
        classes = list(clf.classes_)
        trainable_prob = float(proba[classes.index('trainable')]) if 'trainable' in classes else 0.0
        rescue_prob = float(proba[classes.index('rescue')]) if 'rescue' in classes else 0.0
        failure_prob = float(proba[classes.index('failure')]) if 'failure' in classes else 0.0
        predicted_pathology = float(reg.predict(X)[0]) if reg is not None else float(X['pathology.pathology_score'].iloc[0])
        gap = float(semantic_gap(spec, loss_target))
        comp = float(spec.complexity())
        objective = predicted_pathology + semantic_gap_weight * gap + complexity_weight * comp
        rows.append({
            'spec': spec_to_dict(spec),
            'objective': float(objective),
            'predicted_pathology': predicted_pathology,
            'semantic_gap': gap,
            'complexity_penalty': comp,
            'trainable_prob': trainable_prob,
            'rescue_prob': rescue_prob,
            'failure_prob': failure_prob,
            'phase': 'trainable' if trainable_prob >= max(rescue_prob, failure_prob) else ('rescue' if rescue_prob >= failure_prob else 'failure'),
            'pseudo_diagnostics': X.iloc[0].to_dict(),
        })
    rows = sorted(rows, key=lambda x: x['objective'])
    return rows


def select_target(
    surrogate_path: str | Path,
    output_path: str | Path,
    family: str,
    num_points: int,
    loss_target: TargetSpec,
    semantic_gap_weight: float = 0.5,
    complexity_weight: float = 0.05,
    schedule_basis_order: int = 3,
    alpha_min: float = 0.05,
    alpha_max: float = 0.95,
    custom_alphas: list[float] | None = None,
) -> dict[str, Any]:
    candidates = sample_target_specs(
        family,
        num_points,
        schedule_basis_order=schedule_basis_order,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        custom_alphas=custom_alphas,
    )
    ranked = rank_candidates(candidates, surrogate_path, loss_target, semantic_gap_weight, complexity_weight)
    selected = {
        'loss_target': spec_to_dict(loss_target),
        'weights': {
            'semantic_gap_weight': float(semantic_gap_weight),
            'complexity_weight': float(complexity_weight),
        },
        'selected': ranked[0],
        'topk': ranked[:10],
    }
    save_json(selected, output_path)
    return selected
