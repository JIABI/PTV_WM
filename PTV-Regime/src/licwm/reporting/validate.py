from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from licwm.reporting.specs import ALL_FIGURE_SPECS, ALL_TABLE_SPECS


def _load(csv_path: str):
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def _apply_aliases(cols: set[str], aliases: dict[str, str]) -> set[str]:
    return cols | set(aliases.values())


def validate_artifacts(agg_dir: str = 'outputs/aggregates', out_dir: str = 'outputs/tables') -> dict[str, Any]:
    report: dict[str, Any] = {"tables": [], "figures": []}
    for spec in ALL_TABLE_SPECS:
        df = _load(os.path.join(agg_dir, spec.source_csv)) if spec.source_csv else None
        entry = {
            "artifact_id": spec.artifact_id,
            "source_csv": spec.source_csv,
            "exists": df is not None,
            "missing_required": [],
            "missing_preferred": [],
        }
        if df is not None:
            cols = _apply_aliases(set(df.columns), dict(spec.aliases))
            entry["missing_required"] = [c for c in spec.required_columns if c not in cols]
            entry["missing_preferred"] = [c for c in spec.preferred_columns if c not in cols]
            entry["row_count"] = int(len(df))
        report["tables"].append(entry)
    for spec in ALL_FIGURE_SPECS:
        df = _load(os.path.join(agg_dir, spec.source_csv)) if spec.source_csv else None
        entry = {
            "artifact_id": spec.artifact_id,
            "source_csv": spec.source_csv,
            "generated": spec.generated,
            "exists": True if not spec.generated else df is not None,
            "missing_required": [],
            "missing_preferred": [],
        }
        if spec.generated and df is not None:
            cols = _apply_aliases(set(df.columns), dict(spec.aliases))
            entry["missing_required"] = [c for c in spec.required_columns if c not in cols]
            entry["missing_preferred"] = [c for c in spec.preferred_columns if c not in cols]
            entry["row_count"] = int(len(df))
        report["figures"].append(entry)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'artifact_validation_report.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    return report
