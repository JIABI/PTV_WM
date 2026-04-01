"""Matched-fidelity analysis for Figure 2 and Table 1."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .evaluator import build_audit_from_cfg, build_model_from_cfg, collect_analysis_batch, compute_future_fidelity_proxy, compute_geometry_summary, env_cfg_from_domain, resolve_domain_roster
from .reports import save_csv, save_figure, save_latex_table, save_markdown, save_metrics
from .stats import safe_rank_correlation


def _infer_baseline_name(path: str | None) -> str | None:
    if not path:
        return None
    name = Path(path).stem.lower()
    for key in ["recon_wm", "value_wm", "policy_wm", "rank_wm"]:
        if key in name:
            return key
    return None


def _partial_r2(x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 3:
        return 0.0
    X = np.stack([np.ones_like(x1), x1, x2], axis=1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-8)


def _select_domains(cfg: Any) -> list[dict[str, Any]]:
    roster = resolve_domain_roster(cfg)
    keep = []
    wanted_groups = {"visual_discrete", "visual_continuous", "nonimage_continuous"}
    seen = set()
    for item in roster:
        group = item.get("group", "")
        if group in wanted_groups and group not in seen:
            keep.append(item)
            seen.add(group)
    return keep if keep else [{"name": "dummy", "label": "Dummy", "group": "dummy"}]


def _bucketize(rows: list[dict]) -> None:
    vals = np.asarray([r["future_fidelity_proxy"] for r in rows], dtype=np.float32)
    if len(vals) == 0:
        return
    q1, q2 = np.quantile(vals, [1 / 3, 2 / 3]) if len(vals) >= 3 else (vals.min(initial=0.0), vals.max(initial=0.0))
    for r in rows:
        v = r["future_fidelity_proxy"]
        r["bucket"] = "low" if v <= q1 else ("mid" if v <= q2 else "high")


def run_matched_fidelity(cfg: Any, checkpoint_paths: list[str] | None = None) -> dict:
    checkpoint_paths = checkpoint_paths or ["outputs/checkpoints/ralag_wm.pt"]
    device = str(getattr(cfg, "device", "cpu"))
    domains = _select_domains(cfg)
    rows: list[dict] = []
    for ckpt in checkpoint_paths:
        baseline_name = _infer_baseline_name(ckpt)
        for domain in domains:
            local_cfg = env_cfg_from_domain(cfg, domain)
            model, _ = build_model_from_cfg(local_cfg, ckpt_path=ckpt, device=device)
            audit = build_audit_from_cfg(local_cfg, device=device, single=False)
            batch = collect_analysis_batch(local_cfg, model, audit, device=device, batch_size=int(getattr(cfg.trainer, "batch_size", 8)))
            geom = compute_geometry_summary(model, batch)
            future_fidelity = compute_future_fidelity_proxy(model, batch, baseline_name=baseline_name)
            rows.append({
                "checkpoint": ckpt,
                "method": baseline_name or "ralag_wm",
                "domain": domain["label"],
                "group": domain.get("group", domain["label"]),
                "future_fidelity_proxy": future_fidelity,
                **geom,
            })
    _bucketize(rows)

    predictiveness = []
    for domain in sorted(set(r["domain"] for r in rows)):
        subset = [r for r in rows if r["domain"] == domain]
        xg = np.asarray([r["refined_geom_error"] for r in subset], dtype=np.float32)
        xf = np.asarray([r["future_fidelity_proxy"] for r in subset], dtype=np.float32)
        y_flip = np.asarray([r["top1_disagreement"] for r in subset], dtype=np.float32)
        y_risk = np.asarray([r["boundary_risk_brier"] for r in subset], dtype=np.float32)
        y_margin = np.asarray([r["margin_error"] for r in subset], dtype=np.float32)
        predictiveness.append({
            "domain": domain,
            "partial_r2_flip": _partial_r2(xg, xf, y_flip),
            "partial_r2_risk": _partial_r2(xg, xf, y_risk),
            "partial_r2_margin": _partial_r2(xg, xf, y_margin),
            "rankcorr_geom_vs_flip": safe_rank_correlation(xg, y_flip),
            "rankcorr_fidelity_vs_flip": safe_rank_correlation(xf, y_flip),
            "geom_error_mean": float(xg.mean()) if len(xg) else 0.0,
            "future_fidelity_mean": float(xf.mean()) if len(xf) else 0.0,
        })

    fig, axes = plt.subplots(1, len(domains), figsize=(4.4 * len(domains), 3.6))
    if len(domains) == 1:
        axes = [axes]
    for ax, domain in zip(axes, domains):
        subset = [r for r in rows if r["domain"] == domain["label"]]
        x = np.asarray([r["future_fidelity_proxy"] for r in subset], dtype=np.float32)
        y = np.asarray([r["refined_geom_error"] for r in subset], dtype=np.float32)
        colors = {"low": "tab:blue", "mid": "tab:orange", "high": "tab:green"}
        for bucket in ["low", "mid", "high"]:
            sb = [r for r in subset if r["bucket"] == bucket]
            if not sb:
                continue
            ax.scatter([r["future_fidelity_proxy"] for r in sb], [r["refined_geom_error"] for r in sb], label=bucket, alpha=0.8, s=28, color=colors[bucket])
        if len(x) >= 2:
            coeff = np.polyfit(x, y, deg=1)
            xs = np.linspace(float(x.min()), float(x.max()), 50)
            ax.plot(xs, coeff[0] * xs + coeff[1], color="black", linewidth=1.0)
        ax.set_title(domain["label"])
        ax.set_xlabel("Future fidelity proxy")
        ax.set_ylabel("RALAG distortion")
    axes[0].legend(frameon=False, title="bucket")
    fig.suptitle("Figure 2: Matched-fidelity analysis")
    save_figure(fig, "outputs/figures/figure2_matched_fidelity.png")
    save_metrics("outputs/metrics/matched_fidelity.json", {"rows": rows, "predictiveness": predictiveness})
    save_csv("outputs/metrics/matched_fidelity.csv", rows)
    save_csv("outputs/metrics/table1_predictiveness.csv", predictiveness)
    save_markdown("outputs/debug/table1_predictiveness.md", "Table 1: Failure diagnostics predictiveness", predictiveness)
    save_latex_table(
        "outputs/metrics/table1_predictiveness.tex",
        caption="Matched-fidelity predictiveness summary.",
        label="tab:predictiveness",
        rows=predictiveness,
    )
    return {"rows": rows, "predictiveness": predictiveness}
