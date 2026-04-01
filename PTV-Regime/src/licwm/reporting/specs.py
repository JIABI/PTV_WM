from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class TableSpec:
    artifact_id: str
    title: str
    source_csv: str
    output_stem: str
    preferred_columns: Sequence[str]
    description: str
    aliases: Mapping[str, str] = field(default_factory=dict)
    required_columns: Sequence[str] = field(default_factory=tuple)
    sort_by: Sequence[str] = field(default_factory=tuple)
    group_by: Sequence[str] = field(default_factory=tuple)
    caption_stub: str = ""
    value_aliases: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    categorical_orders: Mapping[str, Sequence[str]] = field(default_factory=dict)
    row_filter: Mapping[str, Sequence[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class FigureSpec:
    artifact_id: str
    title: str
    source_csv: str
    output_stem: str
    preferred_columns: Sequence[str]
    description: str
    plot_kind: str = "auto"
    generated: bool = True
    aliases: Mapping[str, str] = field(default_factory=dict)
    required_columns: Sequence[str] = field(default_factory=tuple)
    sort_by: Sequence[str] = field(default_factory=tuple)
    group_by: Sequence[str] = field(default_factory=tuple)
    caption_stub: str = ""
    value_aliases: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    categorical_orders: Mapping[str, Sequence[str]] = field(default_factory=dict)
    row_filter: Mapping[str, Sequence[str]] = field(default_factory=dict)


COMMON_ALIASES = {
    "trainer.epochs": "epochs",
    "trainer.batch_size": "batch_size",
    "trainer.optimizer": "optimizer",
    "trainer.lr": "lr",
    "full.tv_law": "tv_law",
    "leakage.tv_law": "leakage_tv_law",
    "full.hf_ratio": "hf_ratio",
    "leakage.hf_ratio": "leakage_hf_ratio",
    "small.rollout_ade": "small_rollout_ade",
    "medium.rollout_ade": "medium_rollout_ade",
    "large.rollout_ade": "large_rollout_ade",
    "small.rollout_fde": "small_rollout_fde",
    "medium.rollout_fde": "medium_rollout_fde",
    "large.rollout_fde": "large_rollout_fde",
    "gt_climate_oracle.rollout_ade": "gt_climate_oracle_rollout_ade",
    "law_param_oracle.rollout_ade": "law_param_oracle_rollout_ade",
    "bounded_residual.rollout_ade": "bounded_residual_rollout_ade",
    "full.rollout_ade": "full_rollout_ade",
    "with_event.rollout_ade": "with_event_rollout_ade",
    "no_event.rollout_ade": "no_event_rollout_ade",
    "iid.rollout_ade": "iid_rollout_ade",
    "scaled_agents.rollout_ade": "scaled_agents_rollout_ade",
    "iid.rollout_horizon": "iid_rollout_horizon",
    "scaled_agents.rollout_horizon": "scaled_agents_rollout_horizon",
    "model": "method",
    "ablation": "method",
}

METHOD_ALIASES = {
    "lic_wm": "PTV-Regime",
    "PTV-Regime": "PTV-Regime",
    "auto_physick_wm": "AutoPhysiCK-WM",
    "AutoPhysiCK-WM": "AutoPhysiCK-WM",
    "context_wm": "Context-WM",
    "gru_wm": "GRU-WM",
    "transformer_wm": "Transformer-WM",
    "cfc_wm": "CfC-WM",
    "moe_wm": "MoE-WM",
    "full": "PTV-Regime",
    "no_climate": "No Climate",
    "no_jump": "No Jump",
    "no_slow": "No Slow Drift",
    "leakage": "Leakage",
    "bounded_residual": "Bounded Residual",
    "no_event_token": "No Event Token",
    "small_bank": "Small Bank",
    "medium_bank": "Medium Bank",
    "large_bank": "Large Bank",
    "gt_climate_oracle": "GT Climate Oracle",
    "law_param_oracle": "Law-Parameter Oracle",
}

METHOD_ORDER = (
    "PTV-Regime",
    "AutoPhysiCK-WM",
    "Context-WM",
    "GRU-WM",
    "Transformer-WM",
    "CfC-WM",
    "MoE-WM",
    "No Climate",
    "No Jump",
    "No Slow Drift",
    "Leakage",
    "Bounded Residual",
    "No Event Token",
    "Small Bank",
    "Medium Bank",
    "Large Bank",
    "GT Climate Oracle",
    "Law-Parameter Oracle",
)

TASK_ORDER = (
    "existence",
    "identifiability",
    "cross_domain",
    "control_value",
    "capacity_audit",
    "residual_audit",
    "antisteg_audit",
    "no_event",
    "ood_stress",
    "formation",
    "corridor",
    "escort",
    "pursuit",
)

DOMAIN_ORDER = (
    "lic_boids",
    "eth_ucy",
    "sdd",
    "interaction",
    "lic_uav",
)

SCENE_ORDER = (
    "eth",
    "hotel",
    "univ",
    "zara1",
    "zara2",
    "students",
    "bookstore",
    "coupa",
    "deathCircle",
    "hyang",
    "interaction",
)

METRIC_ALIASES = {
    "step_rmse": "Step RMSE",
    "rollout_ade": "Rollout ADE",
    "rollout_fde": "Rollout FDE",
    "rollout_horizon": "Rollout Horizon",
    "geometry_match_quality": "Geometry Match Quality",
    "law_separation": "Law-State Separation",
    "response_separation": "Response Separation",
    "monotonicity": "Counterfactual Monotonicity",
    "success_rate": "Success Rate",
    "safety_violation": "Safety Violation",
    "formation_error": "Formation Error",
    "connectivity_retention": "Connectivity Retention",
    "energy": "Control Energy",
    "recovery_latency": "Recovery Latency",
    "cvar": "Tail Risk (CVaR)",
    "tv_law": "Law-State TV",
    "hf_ratio": "Law-State HF Ratio",
    "scale_transfer_gap": "Scale Transfer Gap",
}

COMMON_VALUE_ALIASES = {
    "method": METHOD_ALIASES,
    "task": {
        "formation": "Formation Keeping",
        "corridor": "Corridor Crossing",
        "escort": "Escort / Protection",
        "pursuit": "Pursuit-Evasion",
        "existence": "Existence",
        "identifiability": "Identifiability",
        "cross_domain": "Cross-Domain",
        "control_value": "Control Value",
        "capacity_audit": "Capacity Audit",
        "residual_audit": "Residual Audit",
        "antisteg_audit": "Anti-Steganography Audit",
        "no_event": "No Explicit Event",
        "ood_stress": "OOD Stress",
    },
    "domain": {
        "lic_boids": "LIC-Boids",
        "eth_ucy": "ETH/UCY",
        "sdd": "SDD",
        "interaction": "INTERACTION",
        "lic_uav": "LIC-UAV",
    },
    "metric_name": METRIC_ALIASES,
}

COMMON_CATEGORICAL_ORDERS = {
    "method": METHOD_ORDER,
    "task": TASK_ORDER,
    "domain": DOMAIN_ORDER,
    "scene": SCENE_ORDER,
}

MAIN_TABLE_SPECS = [
    TableSpec(
        artifact_id="Table1",
        title="Benchmark, protocol, and metrics summary",
        source_csv="main_section5.csv",
        output_stem="Table1_benchmark_protocol_metrics",
        preferred_columns=("domain", "task", "method", "evaluator", "metric", "epochs", "batch_size", "run", "_source"),
        required_columns=("domain", "task"),
        aliases=COMMON_ALIASES,
        value_aliases=COMMON_VALUE_ALIASES,
        categorical_orders=COMMON_CATEGORICAL_ORDERS,
        sort_by=("domain", "task", "method"),
        description="Main Sec.5 setup, benchmark family, protocol, and metric overview.",
        caption_stub="Summary of benchmark families, task grouping, evaluator paths, and run-level protocol used throughout Main Secs. 6--9.",
    ),
    TableSpec(
        artifact_id="Table2",
        title="LIC-Boids quantitative summary",
        source_csv="main_section6.csv",
        output_stem="Table2_lic_boids_quantitative_summary",
        preferred_columns=("task", "method", "step_rmse", "rollout_ade", "rollout_fde", "rollout_horizon", "geometry_match_quality", "law_separation", "response_separation", "monotonicity", "metric_name", "num_pairs", "_source"),
        required_columns=("geometry_match_quality", "law_separation", "response_separation"),
        aliases=COMMON_ALIASES,
        value_aliases=COMMON_VALUE_ALIASES,
        categorical_orders=COMMON_CATEGORICAL_ORDERS,
        sort_by=("task", "method"),
        row_filter={"domain": ("lic_boids", "LIC-Boids")},
        description="Main Sec.6 existence and identifiability results on LIC-Boids.",
        caption_stub="Quantitative mechanism evidence on LIC-Boids, reporting predictive stability, matched-geometry separation, and counterfactual monotonicity.",
    ),
    TableSpec(
        artifact_id="Table3",
        title="Cross-domain predictive stability",
        source_csv="main_section7.csv",
        output_stem="Table3_cross_domain_predictive_stability",
        preferred_columns=("domain", "scene", "method", "step_rmse", "rollout_ade", "rollout_fde", "rollout_horizon", "geometry_match_quality", "law_separation", "response_separation", "_source"),
        required_columns=("domain",),
        aliases=COMMON_ALIASES,
        value_aliases=COMMON_VALUE_ALIASES,
        categorical_orders=COMMON_CATEGORICAL_ORDERS,
        sort_by=("domain", "scene", "method"),
        row_filter={"domain": ("eth_ucy", "sdd", "interaction", "ETH/UCY", "SDD", "INTERACTION")},
        description="Main Sec.7 crowd and traffic cross-domain evidence.",
        caption_stub="Cross-domain predictive stability and matched-geometry separation across crowd and road-interaction domains.",
    ),
    TableSpec(
        artifact_id="Table4",
        title="LIC-UAV control summary",
        source_csv="main_section8.csv",
        output_stem="Table4_lic_uav_control_summary",
        preferred_columns=("task", "method", "success_rate", "safety_violation", "formation_error", "connectivity_retention", "energy", "recovery_latency", "cvar", "_source"),
        required_columns=("success_rate", "safety_violation"),
        aliases=COMMON_ALIASES,
        value_aliases=COMMON_VALUE_ALIASES,
        categorical_orders=COMMON_CATEGORICAL_ORDERS,
        sort_by=("task", "method"),
        row_filter={"domain": ("lic_uav", "LIC-UAV")},
        description="Main Sec.8 closed-loop control value on LIC-UAV tasks.",
        caption_stub="Closed-loop control value on LIC-UAV, including task success, safety, recovery, and tail risk.",
    ),
    TableSpec(
        artifact_id="Table5",
        title="Ablation and stress-test summary",
        source_csv="main_section9.csv",
        output_stem="Table5_ablation_stress_summary",
        preferred_columns=("method", "tv_law", "hf_ratio", "scale_transfer_gap", "rollout_ade", "rollout_fde", "rollout_horizon", "with_event_rollout_ade", "no_event_rollout_ade", "_source"),
        required_columns=("tv_law", "hf_ratio"),
        aliases=COMMON_ALIASES,
        value_aliases=COMMON_VALUE_ALIASES,
        categorical_orders=COMMON_CATEGORICAL_ORDERS,
        sort_by=("method",),
        description="Main Sec.9 capacity, leakage, no-event, and OOD stress tests.",
        caption_stub="Robustness audits covering law-state bandwidth, no-event activation, capacity, and OOD stress behavior.",
    ),
]

SI_TABLE_SPECS = [
    TableSpec("TableS1", "Dataset splits and sequence statistics", "si_a.csv", "TableS1_dataset_splits_sequence_stats", ("domain", "split", "n_samples", "history_len", "pred_len", "num_agents", "_source"), "SI-A datasets, preprocessing, and event construction.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("domain",), sort_by=("domain", "split"), caption_stub="Dataset splits and sequence statistics for all benchmark families."),
    TableSpec("TableS2", "Event definitions and extraction rules", "si_a.csv", "TableS2_event_definitions_rules", ("event_type", "domain", "threshold", "notes", "_source"), "SI-A observable event construction details.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("event_type", "domain"), sort_by=("domain", "event_type"), caption_stub="Observable event tokens and proxy extraction rules used across domains."),
    TableSpec("TableS3", "Baseline architecture and budget matching", "si_b.csv", "TableS3_baseline_architecture_budget", ("method", "params", "epochs", "batch_size", "planner", "_source"), "SI-B baseline and compute matching.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("method",), sort_by=("method",), caption_stub="Baseline architecture capacity and training-budget matching."),
    TableSpec("TableS4", "Training hyperparameters and reproducibility", "si_b.csv", "TableS4_training_hparams_reproducibility", ("seed", "optimizer", "lr", "history_len", "pred_len", "_source"), "SI-B optimization and reproducibility settings.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("seed",), caption_stub="Optimization and reproducibility settings across runs."),
    TableSpec("TableS5", "Per-event LIC-Boids metrics", "si_c.csv", "TableS5_lic_boids_per_event_metrics", ("event_type", "method", "step_rmse", "rollout_ade", "geometry_match_quality", "law_separation", "response_separation", "_source"), "SI-C extended LIC-Boids analyses.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("event_type",), sort_by=("event_type", "method"), caption_stub="LIC-Boids metrics broken down by event type and method."),
    TableSpec("TableS6", "Counterfactual monotonicity scores", "si_d.csv", "TableS6_counterfactual_monotonicity_scores", ("metric_name", "monotonicity", "delta_sweep", "response_curve", "_source"), "SI-D quantitative counterfactual protocol.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("monotonicity",), sort_by=("metric_name",), caption_stub="Quantitative counterfactual monotonicity and response-curve summaries."),
    TableSpec("TableS7", "Per-dataset crowd and traffic results", "si_e.csv", "TableS7_crowd_traffic_per_dataset_results", ("domain", "scene", "method", "step_rmse", "rollout_ade", "rollout_fde", "_source"), "SI-E crowd and traffic extended analyses.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("domain",), sort_by=("domain", "scene", "method"), caption_stub="Per-dataset and per-scene crowd/traffic predictive results."),
    TableSpec("TableS8", "Per-task LIC-UAV control metrics", "si_f.csv", "TableS8_lic_uav_per_task_metrics", ("task", "method", "success_rate", "safety_violation", "formation_error", "connectivity_retention", "recovery_latency", "cvar", "_source"), "SI-F LIC-UAV per-task results.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("task",), sort_by=("task", "method"), caption_stub="Per-task control performance on the LIC-UAV suite."),
    TableSpec("TableS9", "Planner budget and runtime summary", "si_f.csv", "TableS9_planner_budget_runtime", ("planner", "horizon", "iterations", "candidates", "runtime", "_source"), "SI-F planner and runtime details.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("planner",), caption_stub="Planner budgets and runtime settings used for LIC-UAV control evaluation."),
    TableSpec("TableS10", "Capacity, oracle, and bounded residual results", "si_g.csv", "TableS10_capacity_oracle_residual_results", ("method", "small_rollout_ade", "medium_rollout_ade", "large_rollout_ade", "gt_climate_oracle_rollout_ade", "law_param_oracle_rollout_ade", "bounded_residual_rollout_ade", "_source"), "SI-G fast-law capacity and bounded residual audits.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("small_rollout_ade", "medium_rollout_ade", "large_rollout_ade"), caption_stub="Fast-law capacity sweep, oracle upper bounds, and bounded residual ablations."),
    TableSpec("TableS11", "Anti-steganography audit summary", "si_h.csv", "TableS11_antisteg_summary", ("method", "tv_law", "hf_ratio", "leakage_tv_law", "leakage_hf_ratio", "_source"), "SI-H anti-steganography and leakage audits.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("tv_law", "hf_ratio"), caption_stub="Law-state bandwidth audits comparing full, leakage, and no-slow variants."),
    TableSpec("TableS12", "No-event and OOD stress results", "si_i.csv", "TableS12_no_event_ood_results", ("method", "with_event_rollout_ade", "no_event_rollout_ade", "iid_rollout_ade", "scaled_agents_rollout_ade", "iid_rollout_horizon", "scaled_agents_rollout_horizon", "_source"), "SI-I no-event and OOD stress tests.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("with_event_rollout_ade", "no_event_rollout_ade"), caption_stub="No-event activation and out-of-distribution stress-test summaries."),
]

MAIN_FIGURE_SPECS = [
    FigureSpec("Figure1", "Method overview schematic", "", "Figure1_method_overview", tuple(), "Manual schematic aligned to the method section; not auto-generated from CSV.", generated=False, caption_stub="Overview of PTV-Regime. The fast layer models reusable local response prototypes, while the slow climate generates a bounded law state that deforms local interaction laws through a narrow interface."),
    FigureSpec("Figure2", "LIC-Boids mechanism evidence", "main_section6.csv", "Figure2_lic_boids_mechanism", ("geometry_match_quality", "law_separation", "response_separation", "monotonicity"), "Main Sec.6 existence, identifiability, and counterfactual evidence.", plot_kind="line", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("geometry_match_quality", "law_separation", "response_separation"), sort_by=("task", "method"), row_filter={"domain": ("lic_boids", "LIC-Boids")}, caption_stub="Mechanism evidence on LIC-Boids showing matched-geometry separation, law-state separation, response separation, and counterfactual monotonicity."),
    FigureSpec("Figure3", "Cross-domain crowd and road evidence", "main_section7.csv", "Figure3_cross_domain_evidence", ("step_rmse", "rollout_ade", "rollout_fde", "rollout_horizon"), "Main Sec.7 cross-domain predictive stability and event-aligned drift.", plot_kind="line", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("domain",), sort_by=("domain", "scene", "method"), row_filter={"domain": ("eth_ucy", "sdd", "interaction", "ETH/UCY", "SDD", "INTERACTION")}, caption_stub="Cross-domain predictive stability on crowd and road-interaction benchmarks."),
    FigureSpec("Figure4", "LIC-UAV control value", "main_section8.csv", "Figure4_lic_uav_control_value", ("success_rate", "safety_violation", "recovery_latency", "cvar"), "Main Sec.8 control success, safety, and recovery.", plot_kind="bar", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("task",), sort_by=("task", "method"), row_filter={"domain": ("lic_uav", "LIC-UAV")}, caption_stub="Closed-loop control value on LIC-UAV, including task success, safety, recovery, and tail risk."),
    FigureSpec("Figure5", "Defense and stress tests", "main_section9.csv", "Figure5_defense_stress_tests", ("tv_law", "hf_ratio", "scale_transfer_gap", "rollout_horizon"), "Main Sec.9 capacity, leakage, no-event, and OOD stress tests.", plot_kind="bar", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("tv_law", "hf_ratio"), sort_by=("method",), caption_stub="Defense experiments covering law-state bandwidth, no-event activation, and OOD stress behavior."),
]

SI_FIGURE_SPECS = [
    FigureSpec("FigureS1", "LIC-Boids qualitative event panels", "si_c.csv", "FigureS1_lic_boids_event_panels", ("step_rmse", "rollout_ade", "rollout_fde"), "SI-C event-wise LIC-Boids qualitative/summary panels.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("event_type",), sort_by=("event_type", "method"), caption_stub="Event-wise LIC-Boids summary metrics."),
    FigureSpec("FigureS2", "Climate and law-state recovery traces", "si_c.csv", "FigureS2_climate_lawstate_recovery", ("law_separation", "response_separation", "geometry_match_quality"), "SI-C climate/law-state recovery traces.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("law_separation",), sort_by=("method",), caption_stub="Recovery traces relating geometry, law-state, and response separation."),
    FigureSpec("FigureS3", "Jump timing histograms", "si_c.csv", "FigureS3_jump_timing_histograms", ("num_pairs", "law_separation", "response_separation"), "SI-C jump timing and matched-geometry summaries.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, caption_stub="Jump timing and matched-geometry summary statistics."),
    FigureSpec("FigureS4", "Counterfactual intervention curves", "si_d.csv", "FigureS4_counterfactual_curves", ("monotonicity",), "SI-D counterfactual curves and monotonicity.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("monotonicity",), caption_stub="Counterfactual intervention curves across climate-coordinate sweeps."),
    FigureSpec("FigureS5", "Counterfactual monotonicity distributions", "si_d.csv", "FigureS5_counterfactual_monotonicity_distributions", ("monotonicity",), "SI-D monotonicity distribution summaries.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("monotonicity",), caption_stub="Distribution of monotonicity scores across intervention families."),
    FigureSpec("FigureS6", "Crowd event-aligned rollout curves", "si_e.csv", "FigureS6_crowd_event_aligned_rollout_curves", ("step_rmse", "rollout_ade", "rollout_fde"), "SI-E crowd extended results.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("domain",), caption_stub="Event-aligned rollout summaries for crowd scenes."),
    FigureSpec("FigureS7", "Traffic event-aligned conflict recovery", "si_e.csv", "FigureS7_traffic_conflict_recovery", ("step_rmse", "rollout_ade", "rollout_fde"), "SI-E traffic extended results.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("domain",), caption_stub="Conflict-zone recovery summaries for traffic interactions."),
    FigureSpec("FigureS8", "Representative LIC-UAV rollouts", "si_f.csv", "FigureS8_lic_uav_representative_rollouts", ("success_rate", "safety_violation", "formation_error"), "SI-F representative LIC-UAV rollouts.", plot_kind="bar", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("task",), caption_stub="Representative control summaries for each LIC-UAV task."),
    FigureSpec("FigureS9", "Event-to-recovery by task", "si_f.csv", "FigureS9_event_to_recovery_by_task", ("recovery_latency", "cvar"), "SI-F task-level recovery and tail-risk summaries.", plot_kind="bar", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("task",), caption_stub="Event-to-recovery and tail-risk summaries by LIC-UAV task."),
    FigureSpec("FigureS10", "Fast-law capacity sweep", "si_g.csv", "FigureS10_fast_law_capacity_sweep", ("small_rollout_ade", "medium_rollout_ade", "large_rollout_ade"), "SI-G bank-size sweep on representative domains.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("small_rollout_ade",), caption_stub="Fast-law capacity sweep from small to large prototype banks."),
    FigureSpec("FigureS11", "Cross-domain capacity sweep", "si_g.csv", "FigureS11_cross_domain_capacity_sweep", ("gt_climate_oracle_rollout_ade", "law_param_oracle_rollout_ade"), "SI-G oracle upper bounds across capacity settings.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, caption_stub="Oracle upper bounds comparing GT climate and law-parameter supervision."),
    FigureSpec("FigureS12", "Bounded residual audit", "si_g.csv", "FigureS12_bounded_residual_audit", ("bounded_residual_rollout_ade",), "SI-G bounded residual audit.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, caption_stub="Bounded residual audit showing that residual channels are not the primary driver of OOD behavior."),
    FigureSpec("FigureS13", "Law-state temporal variation", "si_h.csv", "FigureS13_law_state_temporal_variation", ("tv_law", "leakage_tv_law"), "SI-H law-state total variation audits.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("tv_law",), caption_stub="Temporal variation of the law state across full and leakage variants."),
    FigureSpec("FigureS14", "Law-state spectral energy", "si_h.csv", "FigureS14_law_state_spectral_energy", ("hf_ratio", "leakage_hf_ratio"), "SI-H high-frequency energy audits.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("hf_ratio",), caption_stub="High-frequency spectral energy ratio of the law-state trajectory."),
    FigureSpec("FigureS15", "Leakage and time-shuffle sensitivity", "si_h.csv", "FigureS15_leakage_time_shuffle_sensitivity", ("tv_law", "hf_ratio"), "SI-H leakage and time-shuffling sensitivity summaries.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, caption_stub="Bandwidth-restriction audit under leakage and time-shuffling probes."),
    FigureSpec("FigureS16", "With / without / noisy / delayed event tokens", "si_i.csv", "FigureS16_event_token_variants", ("with_event_rollout_ade", "no_event_rollout_ade"), "SI-I event-token ablations.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, required_columns=("with_event_rollout_ade", "no_event_rollout_ade"), caption_stub="Event-token ablations comparing explicit, absent, noisy, and delayed event inputs."),
    FigureSpec("FigureS17", "Scale and density transfer", "si_i.csv", "FigureS17_scale_density_transfer", ("iid_rollout_ade", "scaled_agents_rollout_ade"), "SI-I scale and density transfer.", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, caption_stub="OOD transfer under larger agent populations and higher density."),
    FigureSpec("FigureS18", "Unseen event combinations", "si_i.csv", "FigureS18_unseen_event_combinations", ("iid_rollout_horizon", "scaled_agents_rollout_horizon"), "SI-I unseen event combinations and longer-horizon stress.", plot_kind="bar", aliases=COMMON_ALIASES, value_aliases=COMMON_VALUE_ALIASES, categorical_orders=COMMON_CATEGORICAL_ORDERS, caption_stub="OOD stress under unseen event combinations and larger coordination scopes."),
]

ALL_TABLE_SPECS = MAIN_TABLE_SPECS + SI_TABLE_SPECS
ALL_FIGURE_SPECS = MAIN_FIGURE_SPECS + SI_FIGURE_SPECS
