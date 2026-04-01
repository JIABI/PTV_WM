from __future__ import annotations

import json
import os
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from licwm.reporting.specs import MAIN_FIGURE_SPECS, SI_FIGURE_SPECS, FigureSpec

IDENTITY_CANDIDATES = ("method", "domain", "task", "scene", "run", "metric_name", "_source")


def _load_frame(csv_path: str) -> pd.DataFrame | None:
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def _apply_aliases(df: pd.DataFrame, spec: FigureSpec) -> pd.DataFrame:
    rename = {k: v for k, v in spec.aliases.items() if k in df.columns}
    if rename:
        df = df.rename(columns=rename)
    return df


def _apply_value_aliases(df: pd.DataFrame, spec: FigureSpec) -> pd.DataFrame:
    for col, mp in spec.value_aliases.items():
        if col in df.columns:
            df[col] = df[col].map(lambda x: mp.get(x, x))
    return df


def _apply_row_filter(df: pd.DataFrame, spec: FigureSpec) -> pd.DataFrame:
    out = df
    for col, allowed in spec.row_filter.items():
        if col in out.columns:
            out = out[out[col].isin(list(allowed))]
    return out


def _apply_categorical_order(df: pd.DataFrame, spec: FigureSpec) -> pd.DataFrame:
    out = df
    for col, order in spec.categorical_orders.items():
        if col in out.columns:
            out[col] = pd.Categorical(out[col], categories=list(order), ordered=True)
    return out


def _normalize(df: pd.DataFrame, spec: FigureSpec) -> pd.DataFrame:
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


def _choose_plot_columns(df: pd.DataFrame, preferred_columns: Iterable[str]) -> list[str]:
    cols = [c for c in preferred_columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if cols:
        return cols[:4]
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return numeric[:4]


def _choose_index(df: pd.DataFrame) -> pd.Series:
    for c in IDENTITY_CANDIDATES:
        if c in df.columns:
            return df[c].astype(str)
    return pd.Series(range(len(df)), name="index")


def _metadata(spec: FigureSpec, plot_cols: list[str], row_count: int) -> dict:
    return {
        "artifact_id": spec.artifact_id,
        "title": spec.title,
        "description": spec.description,
        "caption_stub": spec.caption_stub,
        "source_csv": spec.source_csv,
        "plot_kind": spec.plot_kind,
        "plotted_columns": plot_cols,
        "required_columns": list(spec.required_columns),
        "sort_by": list(spec.sort_by),
        "group_by": list(spec.group_by),
        "row_filter": {k: list(v) for k, v in spec.row_filter.items()},
        "categorical_orders": {k: list(v) for k, v in spec.categorical_orders.items()},
        "row_count": int(row_count),
    }


def _write_figure_artifact(spec: FigureSpec, agg_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, f"{spec.output_stem}.json")
    caption_path = os.path.join(out_dir, f"{spec.output_stem}_caption.txt")
    note_path = os.path.join(out_dir, f"{spec.output_stem}.txt")
    if not spec.generated:
        with open(note_path, "w", encoding="utf-8") as f:
            f.write(f"{spec.artifact_id}: {spec.title}\n")
            f.write(spec.description + "\n")
            f.write("This artifact is intentionally manual/non-generated.\n")
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(spec.caption_stub.strip() + "\n")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(_metadata(spec, [], 0), f, indent=2)
        return

    csv_path = os.path.join(agg_dir, spec.source_csv)
    df = _load_frame(csv_path)
    if df is None:
        return
    df = _normalize(df, spec)
    plot_cols = _choose_plot_columns(df, spec.preferred_columns)
    if not plot_cols:
        return
    figure_ready_dir = os.path.join(out_dir, "figure_ready")
    os.makedirs(figure_ready_dir, exist_ok=True)
    idx = _choose_index(df)
    fr = pd.DataFrame({"index": idx})
    for c in plot_cols:
        fr[c] = df[c]
    fr.to_csv(os.path.join(figure_ready_dir, f"{spec.output_stem}.csv"), index=False)

    plt.figure(figsize=(7.8, 4.6))
    if spec.plot_kind == "bar":
        width = 0.8 / max(1, len(plot_cols))
        xs = list(range(len(fr)))
        for i, c in enumerate(plot_cols):
            offs = [x + (i - (len(plot_cols) - 1) / 2) * width for x in xs]
            plt.bar(offs, fr[c], width=width, alpha=0.85, label=c)
        plt.xticks(xs, fr["index"].astype(str), rotation=20, ha="right")
    else:
        for c in plot_cols:
            plt.plot(fr["index"].astype(str), fr[c], marker="o", label=c)
        plt.xticks(rotation=20, ha="right")
    plt.title(f"{spec.artifact_id}: {spec.title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{spec.output_stem}.png"), dpi=180)
    plt.close()

    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(spec.caption_stub.strip() + "\n")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_metadata(spec, plot_cols, len(df)), f, indent=2)


def make_main_figures(agg_dir: str = "outputs/aggregates", out_dir: str = "outputs/figures") -> None:
    for spec in MAIN_FIGURE_SPECS:
        _write_figure_artifact(spec, agg_dir, out_dir)


def make_si_figures(agg_dir: str = "outputs/aggregates", out_dir: str = "outputs/figures") -> None:
    for spec in SI_FIGURE_SPECS:
        _write_figure_artifact(spec, agg_dir, out_dir)
