"""Unified benchmark runner for main-paper Table 3 results."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ralagwm.envs import make_env
from .evaluator import build_model_from_cfg, env_cfg_from_domain, make_policy, resolve_domain_roster, rollout_policy
from .reports import save_csv, save_latex_table, save_markdown, save_metrics
from .stats import human_normalized, summarize_metric, with_prefix


def _summarize_group(rows: list[dict], group: str, seed: int = 0) -> dict:
    subset = [r for r in rows if r.get("group") == group]
    out = {"group": group, "num_tasks": len(subset)}
    if not subset:
        return out
    metrics = {
        "return": [float(r["mean_return"]) for r in subset],
        "flip": [float(r["flip_proxy"]) for r in subset],
        "chat": [float(r["chattering_proxy"]) for r in subset],
        "cvar10": [float(r["cvar10"]) for r in subset],
        "length": [float(r["mean_length"]) for r in subset],
        "decision_flops": [float(r.get("decision_flops", 0.0)) for r in subset],
    }
    if any(r.get("human_normalized_return") is not None for r in subset):
        metrics["hn_return"] = [float(r["human_normalized_return"]) for r in subset if r.get("human_normalized_return") is not None]
    if any("success_rate" in r for r in subset):
        metrics["success"] = [float(r.get("success_rate", 0.0)) for r in subset]
    if any("achievement_score" in r for r in subset):
        metrics["achievement"] = [float(r.get("achievement_score", 0.0)) for r in subset]
    for name, vals in metrics.items():
        out.update(with_prefix(summarize_metric(vals, seed=seed), name))
    return out


def _latex_rows_from_group_summary(group_rows: list[dict]) -> list[dict]:
    latex_rows: list[dict] = []
    for row in group_rows:
        latex_rows.append({
            "group": row["group"],
            "iqm_return": row.get("return_iqm", 0.0),
            "median_return": row.get("return_median", 0.0),
            "flip": row.get("flip_mean", 0.0),
            "chat": row.get("chat_mean", 0.0),
            "cvar10": row.get("cvar10_mean", 0.0),
            "flops": row.get("decision_flops_mean", 0.0),
        })
    return latex_rows


def run_benchmark(cfg: Any, checkpoint_path: str | None = None, episodes: int | None = None, max_steps: int | None = None) -> dict:
    device = str(getattr(cfg, "device", "cpu"))
    roster = resolve_domain_roster(cfg)
    all_rows: list[dict] = []
    for domain in roster:
        local_cfg = env_cfg_from_domain(cfg, domain)
        env = make_env(local_cfg.env)
        model, is_ralag = build_model_from_cfg(local_cfg, ckpt_path=checkpoint_path, device=device)
        policy = make_policy(model, is_ralag=is_ralag, env=env, device=device)
        metrics = rollout_policy(
            env,
            policy,
            episodes=int(episodes or getattr(local_cfg.experiment, "episodes", 2) or 2),
            max_steps=int(max_steps or getattr(local_cfg.env, "max_episode_steps", env.spec.max_episode_steps)),
            seed=int(getattr(local_cfg, "seed", 0)),
        )
        row = {
            "domain": domain.get("label", domain.get("name", env.spec.name)),
            "group": domain.get("group", domain.get("name", env.spec.name)),
            "env_name": env.spec.name,
            "obs_type": env.spec.obs_type,
            "action_type": env.spec.action_type,
            "action_dim": env.spec.action_dim,
            "decision_flops": float(getattr(local_cfg.chart, "chart_budget", 8) * max(env.spec.action_dim, 1)),
            **metrics,
        }
        row["human_normalized_return"] = human_normalized(
            float(row["mean_return"]),
            domain.get("random_score", None),
            domain.get("human_score", None),
        )
        all_rows.append(row)
        env.close()

    group_rows = [_summarize_group(all_rows, group, seed=int(getattr(cfg, "seed", 0))) for group in sorted(set(r["group"] for r in all_rows))]

    overall = _summarize_group([{**r, "group": "overall"} for r in all_rows], "overall", seed=int(getattr(cfg, "seed", 0)))
    summary = {
        "checkpoint": checkpoint_path or "none",
        "num_domains": len(all_rows),
        **overall,
    }
    save_metrics("outputs/metrics/main_benchmark.json", {"summary": summary, "per_domain": all_rows, "group_summary": group_rows})
    save_csv("outputs/metrics/main_benchmark.csv", all_rows)
    save_csv("outputs/metrics/main_benchmark_groups.csv", group_rows)
    save_markdown("outputs/debug/main_benchmark.md", "Main Benchmark (Table 3)", all_rows)
    save_latex_table(
        "outputs/metrics/table3_main_benchmark.tex",
        caption="Main benchmark summary across paper domains.",
        label="tab:main_benchmark",
        rows=_latex_rows_from_group_summary(group_rows),
    )
    return {"summary": summary, "per_domain": all_rows, "group_summary": group_rows}
