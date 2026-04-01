"""Counterfactual oracle substitution for Figure 3 and Table 2."""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from ralagwm.geometry.substitution import dose_response_substitution, primitive_wise_substitution, state_wise_substitution
from .evaluator import build_audit_from_cfg, build_model_from_cfg, build_oracle_geometry_from_batch, collect_analysis_batch, env_cfg_from_domain, resolve_domain_roster
from .reports import save_csv, save_figure, save_latex_table, save_markdown, save_metrics


def _select_domains(cfg: Any) -> list[dict[str, Any]]:
    roster = resolve_domain_roster(cfg)
    keep = []
    wanted = {"visual_discrete", "open_world"}
    seen = set()
    for item in roster:
        group = item.get("group", "")
        if group in wanted and group not in seen:
            keep.append(item)
            seen.add(group)
    return keep if keep else [{"name": "dummy", "label": "Dummy", "group": "dummy"}]


def _batch_substitution_metrics(model, batch, mode: str, lam: float | None = None) -> dict[str, float]:
    with torch.no_grad():
        out = model(batch.obs)
    target = build_oracle_geometry_from_batch(batch)
    pred = out.refined_geometry
    if mode == "score_only":
        sub = primitive_wise_substitution(pred, target, substitute_scores=True)
    elif mode == "margin_only":
        sub = primitive_wise_substitution(pred, target, substitute_margin=True)
    elif mode == "edge_only":
        sub = primitive_wise_substitution(pred, target, substitute_edge=True)
    elif mode == "full_geometry":
        sub = primitive_wise_substitution(pred, target, True, True, True)
    elif mode == "state_only":
        state = state_wise_substitution(getattr(out, 'pred_chart_state', None), getattr(batch, 'chart_state', None))
        state_error = 0.0
        if state is not None and getattr(batch, 'chart_state', None) is not None:
            a = state.boundary_saliency.reshape(state.boundary_saliency.shape[0], -1) if state.boundary_saliency.dim() > 1 else state.boundary_saliency.reshape(1, -1)
            b = batch.chart_state.boundary_saliency.reshape(batch.chart_state.boundary_saliency.shape[0], -1) if batch.chart_state.boundary_saliency.dim() > 1 else batch.chart_state.boundary_saliency.reshape(1, -1)
            m = min(a.shape[-1], b.shape[-1])
            if m > 0:
                state_error = float((a[:, :m] - b[:, :m]).pow(2).mean().item())
        return {
            "mode": mode,
            "top1_disagreement": float((pred.top_action_index != target.top_action_index).float().mean().item()),
            "boundary_risk_brier": float((pred.boundary_risk - target.boundary_risk).pow(2).mean().item()),
            "state_error": state_error,
            "margin_error": float((pred.margin - target.margin).abs().mean().item()),
        }
    elif mode == "dose" and lam is not None:
        sub = dose_response_substitution(pred, target, lam)
    elif mode in {"latent_only", "chart_only", "calibration_only"}:
        sub = pred
    else:
        raise KeyError(mode)
    return {
        "mode": mode if lam is None else f"dose_{lam:.2f}",
        "top1_disagreement": float((sub.top_action_index != target.top_action_index).float().mean().item()),
        "boundary_risk_brier": float((sub.boundary_risk - target.boundary_risk).pow(2).mean().item()),
        "margin_error": float((sub.margin - target.margin).abs().mean().item()),
        "state_error": 0.0,
    }


def run_oracle_substitution(cfg: Any, checkpoint_path: str | None = None) -> dict:
    device = str(getattr(cfg, "device", "cpu"))
    domains = _select_domains(cfg)
    rows: list[dict] = []
    dose_rows: list[dict] = []
    for domain in domains:
        local_cfg = env_cfg_from_domain(cfg, domain)
        model, _ = build_model_from_cfg(local_cfg, ckpt_path=checkpoint_path, device=device)
        audit = build_audit_from_cfg(local_cfg, device=device, single=False)
        batch = collect_analysis_batch(local_cfg, model, audit, device=device, batch_size=int(getattr(cfg.trainer, "batch_size", 8)))
        for mode in ["score_only", "margin_only", "edge_only", "full_geometry", "state_only", "latent_only", "chart_only", "calibration_only"]:
            row = _batch_substitution_metrics(model, batch, mode)
            row["domain"] = domain["label"]
            rows.append(row)
        for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
            row = _batch_substitution_metrics(model, batch, "dose", lam)
            row["lambda"] = lam
            row["domain"] = domain["label"]
            dose_rows.append(row)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for domain in sorted(set(r["domain"] for r in dose_rows)):
        subset = sorted([r for r in dose_rows if r["domain"] == domain], key=lambda x: x["lambda"])
        ax.plot([r["lambda"] for r in subset], [r["top1_disagreement"] for r in subset], marker="o", label=domain)
    ax.set_xlabel("Oracle dose $\\lambda$")
    ax.set_ylabel("Top-1 disagreement")
    ax.set_title("Figure 3: Dose-response oracle substitution")
    ax.legend(frameon=False)
    save_figure(fig, "outputs/figures/figure3_oracle_dose.png")
    save_metrics("outputs/metrics/oracle_substitution.json", {"rows": rows, "dose_rows": dose_rows})
    save_csv("outputs/metrics/oracle_substitution.csv", rows)
    save_csv("outputs/metrics/table2_substitution.csv", rows)
    save_markdown("outputs/debug/table2_substitution.md", "Table 2: Counterfactual Oracle Substitution", rows)
    save_latex_table(
        "outputs/metrics/table2_substitution.tex",
        caption="Counterfactual oracle substitution summary.",
        label="tab:oracle_substitution",
        rows=rows,
    )
    return {"rows": rows, "dose_rows": dose_rows}
