from __future__ import annotations

import json
import os
from typing import Iterable

import pandas as pd

from licwm.reporting.specs import MAIN_TABLE_SPECS, SI_TABLE_SPECS, TableSpec


def _load_table_frame(csv_path: str) -> pd.DataFrame | None:
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def _apply_aliases(df: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    rename = {k: v for k, v in spec.aliases.items() if k in df.columns}
    if rename:
        df = df.rename(columns=rename)
    return df


def _apply_value_aliases(df: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    for col, mp in spec.value_aliases.items():
        if col in df.columns:
            df[col] = df[col].map(lambda x: mp.get(x, x))
    return df


def _apply_row_filter(df: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    out = df
    for col, allowed in spec.row_filter.items():
        if col in out.columns:
            out = out[out[col].isin(list(allowed))]
    return out


def _apply_categorical_order(df: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    out = df
    for col, order in spec.categorical_orders.items():
        if col in out.columns:
            out[col] = pd.Categorical(out[col], categories=list(order), ordered=True)
    return out


def _normalize(df: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    df = _apply_aliases(df, spec)
    for c in spec.required_columns:
        if c not in df.columns:
            df[c] = pd.NA
    df = _apply_value_aliases(df, spec)
    df = _apply_row_filter(df, spec)
    df = _apply_categorical_order(df, spec)
    if spec.sort_by:
        sort_cols = [c for c in spec.sort_by if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, kind="stable")
    return df


def _select_columns(df: pd.DataFrame, preferred_columns: Iterable[str]) -> pd.DataFrame:
    preferred = [c for c in preferred_columns if c in df.columns]
    rest = [c for c in df.columns if c not in preferred]
    return df[preferred + rest]


def _metadata(spec: TableSpec, df: pd.DataFrame) -> dict:
    return {
        "artifact_id": spec.artifact_id,
        "title": spec.title,
        "description": spec.description,
        "caption_stub": spec.caption_stub,
        "source_csv": spec.source_csv,
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "required_columns": list(spec.required_columns),
        "sort_by": list(spec.sort_by),
        "group_by": list(spec.group_by),
        "row_filter": {k: list(v) for k, v in spec.row_filter.items()},
        "categorical_orders": {k: list(v) for k, v in spec.categorical_orders.items()},
    }


def _write_table_artifact(spec: TableSpec, agg_dir: str, out_dir: str) -> None:
    csv_path = os.path.join(agg_dir, spec.source_csv)
    df = _load_table_frame(csv_path)
    if df is None:
        return
    df = _normalize(df, spec)
    df = _select_columns(df, spec.preferred_columns)
    os.makedirs(out_dir, exist_ok=True)
    artifact_csv = os.path.join(out_dir, f"{spec.output_stem}.csv")
    artifact_md = os.path.join(out_dir, f"{spec.output_stem}.md")
    artifact_meta = os.path.join(out_dir, f"{spec.output_stem}.json")
    artifact_caption = os.path.join(out_dir, f"{spec.output_stem}_caption.txt")
    df.to_csv(artifact_csv, index=False)
    with open(artifact_md, "w", encoding="utf-8") as f:
        f.write(f"# {spec.artifact_id}: {spec.title}\n\n")
        f.write(f"{spec.description}\n\n")
        if len(df) > 0:
            f.write(df.to_markdown(index=False))
            f.write("\n")
        else:
            f.write("No rows after filtering.\n")
    with open(artifact_meta, "w", encoding="utf-8") as f:
        json.dump(_metadata(spec, df), f, indent=2)
    with open(artifact_caption, "w", encoding="utf-8") as f:
        f.write(spec.caption_stub.strip() + "\n")


def make_main_tables(agg_dir: str = "outputs/aggregates", out_dir: str = "outputs/tables") -> None:
    for spec in MAIN_TABLE_SPECS:
        _write_table_artifact(spec, agg_dir, out_dir)


def make_si_tables(agg_dir: str = "outputs/aggregates", out_dir: str = "outputs/tables") -> None:
    for spec in SI_TABLE_SPECS:
        _write_table_artifact(spec, agg_dir, out_dir)
