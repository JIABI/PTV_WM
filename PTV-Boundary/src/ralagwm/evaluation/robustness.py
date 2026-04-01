"""Robustness evaluation for Table 4 and SI audit/protocol/deploy diagnostics."""
from __future__ import annotations

from typing import Any

import numpy as np

from ralagwm.chart.manual import ManualChartGenerator
from ralagwm.training.loops import build_ralag_batch_from_replay, collect_rollout
from .evaluator import build_audit_from_cfg, build_model_from_cfg, env_cfg_from_domain, make_policy, resolve_domain_roster
from .reports import save_csv, save_latex_table, save_markdown, save_metrics
from .stats import safe_rank_correlation
from ralagwm.envs import make_env
from ralagwm.data.replay import ReplayBuffer


class _RandomChartGenerator:
    def __init__(self, chart_budget: int = 8):
        self.chart_budget = chart_budget

    def generate(self, chart_state, consensus, disagreement):
        import torch
        n = consensus.shape[0]
        idx = torch.randperm(n, device=consensus.device)[: min(self.chart_budget, n)]
        coords = chart_state.action_coords[idx]
        edges = torch.zeros((0, 2), dtype=torch.long, device=coords.device)
        weights = torch.ones(idx.shape[0], device=coords.device)
        info = torch.eye(coords.shape[-1], device=coords.device)
        from ralagwm.typing import BICChart
        return BICChart(actions=idx, coords=coords, edges=edges, weights=weights, info_matrix=info, selected_indices=idx)


def _pick_domains(cfg: Any) -> list[dict[str, Any]]:
    roster = resolve_domain_roster(cfg)
    keep = []
    seen = set()
    for item in roster:
        group = item.get("group", "")
        if group not in seen:
            keep.append(item)
            seen.add(group)
    return keep[:4] if keep else [{"name": "dummy", "label": "Dummy", "group": "dummy"}]


def _eval_variant(local_cfg, checkpoint_path: str | None, deploy_kind: str | None = None, protocol: str = "bic") -> dict:
    device = str(getattr(local_cfg, "device", "cpu"))
    if deploy_kind is not None:
        local_cfg.deploy.kind = deploy_kind
    model, _ = build_model_from_cfg(local_cfg, ckpt_path=checkpoint_path, device=device)
    if protocol == "manual":
        model.chart_generator = ManualChartGenerator(chart_budget=int(getattr(local_cfg.chart, "chart_budget", 8)))
    elif protocol == "random":
        model.chart_generator = _RandomChartGenerator(chart_budget=int(getattr(local_cfg.chart, "chart_budget", 8)))
    env = make_env(local_cfg.env)
    policy = make_policy(model, is_ralag=True, env=env, device=device)
    from .evaluator import rollout_policy
    metrics = rollout_policy(env, policy, episodes=1, max_steps=min(128, env.spec.max_episode_steps), seed=int(getattr(local_cfg, "seed", 0)))
    env.close()
    return metrics


def _audit_cross_matrix(local_cfg, checkpoint_path: str | None) -> list[dict[str, float | str]]:
    device = str(getattr(local_cfg, "device", "cpu"))
    model, _ = build_model_from_cfg(local_cfg, ckpt_path=checkpoint_path, device=device)
    replay = ReplayBuffer(capacity=128)
    env = make_env(local_cfg.env)
    collect_rollout(env, replay, policy_fn=lambda obs: env.sample_random_action(), episodes=1, max_steps=min(64, env.spec.max_episode_steps), seed=int(getattr(local_cfg, "seed", 0)))
    multi = build_audit_from_cfg(local_cfg, device=device, single=False)
    single = build_audit_from_cfg(local_cfg, device=device, single=True)
    batch_multi = build_ralag_batch_from_replay(replay, model, multi, batch_size=min(8, len(replay)), device=device)
    batch_single = build_ralag_batch_from_replay(replay, model, single, batch_size=min(8, len(replay)), device=device)
    env.close()
    m = batch_multi.geometry_target
    s = batch_single.geometry_target
    rows = [
        {
            "train_audit": "single",
            "test_audit": "multi",
            "top1_agreement": float((m.top_action_index == s.top_action_index).float().mean().item()),
            "margin_alignment": float(1.0 / (1.0 + (m.margin - s.margin).abs().mean().item())),
            "risk_alignment": float(1.0 / (1.0 + (m.boundary_risk - s.boundary_risk).abs().mean().item())),
        },
        {
            "train_audit": "multi",
            "test_audit": "single",
            "top1_agreement": float((m.top_action_index == s.top_action_index).float().mean().item()),
            "margin_alignment": float(1.0 / (1.0 + (m.margin - s.margin).abs().mean().item())),
            "risk_alignment": float(1.0 / (1.0 + (m.boundary_risk - s.boundary_risk).abs().mean().item())),
        },
    ]
    return rows


def run_robustness(cfg: Any, checkpoint_path: str | None = None) -> dict:
    domains = _pick_domains(cfg)
    rows: list[dict] = []
    audit_rows: list[dict] = []
    for domain in domains:
        local_cfg = env_cfg_from_domain(cfg, domain)
        base = _eval_variant(local_cfg, checkpoint_path, deploy_kind=str(cfg.deploy.kind), protocol="bic")
        rows.append({"domain": domain.get("label", domain["name"]), "axis": "protocol", "setting": "bic", **base})
        for protocol in ["manual", "random"]:
            met = _eval_variant(local_cfg, checkpoint_path, deploy_kind=str(cfg.deploy.kind), protocol=protocol)
            rows.append({"domain": domain.get("label", domain["name"]), "axis": "protocol", "setting": protocol, **met})
        for deploy_kind in ["linear", "mlp", "planner"]:
            met = _eval_variant(local_cfg, checkpoint_path, deploy_kind=deploy_kind, protocol="bic")
            rows.append({"domain": domain.get("label", domain["name"]), "axis": "deploy", "setting": deploy_kind, **met})
        audit_rows.extend([{"domain": domain.get("label", domain["name"]), **r} for r in _audit_cross_matrix(local_cfg, checkpoint_path)])

    summary_rows = []
    for axis in sorted(set(r["axis"] for r in rows)):
        subset = [r for r in rows if r["axis"] == axis]
        returns = np.asarray([float(r["mean_return"]) for r in subset], dtype=np.float32)
        flips = np.asarray([float(r["flip_proxy"]) for r in subset], dtype=np.float32)
        summary_rows.append({
            "axis": axis,
            "relative_gain": float(returns.mean()) if len(returns) else 0.0,
            "ranking_stability": float(np.std(returns)) if len(returns) else 0.0,
            "kendall_tau_proxy": safe_rank_correlation(returns, -flips),
            "num_rows": len(subset),
        })

    save_metrics("outputs/metrics/robustness.json", {"rows": rows, "summary": summary_rows, "audit_matrix": audit_rows})
    save_csv("outputs/metrics/robustness.csv", rows)
    save_csv("outputs/metrics/table4_robustness.csv", summary_rows)
    save_csv("outputs/metrics/si_cross_audit_matrix.csv", audit_rows)
    save_markdown("outputs/debug/table4_robustness.md", "Table 4: Robustness", summary_rows)
    save_markdown("outputs/debug/si_cross_audit_matrix.md", "SI: Cross-audit matrix", audit_rows)
    save_latex_table(
        "outputs/metrics/table4_robustness.tex",
        caption="Robustness summary across protocol, deploy, and audit axes.",
        label="tab:robustness",
        rows=summary_rows,
    )
    return {"rows": rows, "summary": summary_rows, "audit_matrix": audit_rows}
