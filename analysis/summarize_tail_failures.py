import argparse

import json

from pathlib import Path

from typing import List, Optional

import matplotlib.pyplot as plt

import pandas as pd


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".pkl":
        return pd.read_pickle(path)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type: {path}")


def find_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:

        if c in df.columns:
            return c

    return None


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def build_method_column(df: pd.DataFrame) -> pd.Series:
    for c in ["mode", "method", "runner", "train.mode", "exp_id"]:

        if c in df.columns:
            return df[c].astype(str)

    return pd.Series(["unknown"] * len(df), index=df.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize tail failures across runs.")

    parser.add_argument("--input", required=True, help="Input result table (csv/pkl/parquet).")

    parser.add_argument("--outdir", required=True, help="Output directory.")

    args = parser.parse_args()

    outdir = Path(args.outdir)

    ensure_outdir(outdir)

    df = load_table(Path(args.input)).copy()

    if "method_name" not in df.columns:
        df["method_name"] = build_method_column(df)

    tail_candidates = {

        "worst_k_score": ["tail.worst_k_score", "worst_k_score"],

        "percentile_95": ["tail.percentile_95", "percentile_95"],

        "percentile_99": ["tail.percentile_99", "percentile_99"],

        "rare_failure_rate": ["tail.rare_failure_rate", "rare_failure_rate"],

        "collapse_rate": ["quality.collapse_rate", "collapse_rate"],

    }

    resolved = {}

    for name, candidates in tail_candidates.items():

        col = find_first_existing(df, candidates)

        if col is not None:
            df[col] = pd.to_numeric(df[col], errors="coerce")

            resolved[name] = col

    if not resolved:
        raise RuntimeError(

            "No tail-failure columns found. Expected columns like "

            "`tail.worst_k_score`, `tail.percentile_95`, `tail.percentile_99`, "

            "`tail.rare_failure_rate`, or `quality.collapse_rate`."

        )

    grouped = df.groupby("method_name").agg(

        {col: ["mean", "std", "count"] for col in resolved.values()}

    )

    grouped.columns = [

        f"{orig_col}__{stat}" for orig_col, stat in grouped.columns.to_flat_index()

    ]

    grouped = grouped.reset_index()

    grouped.to_csv(outdir / "tail_failure_summary.csv", index=False)

    with open(outdir / "resolved_tail_columns.json", "w") as f:

        json.dump(resolved, f, indent=2)

    # Ranking by worst tail risk: lower is better

    ranking_metric_name = None

    for candidate in ["percentile_99", "rare_failure_rate", "worst_k_score", "collapse_rate"]:

        if candidate in resolved:
            ranking_metric_name = candidate

            break

    if ranking_metric_name is not None:
        ranking_col = f"{resolved[ranking_metric_name]}__mean"

        ranked = grouped.sort_values(ranking_col, ascending=True)

        ranked.to_csv(outdir / "tail_failure_ranking.csv", index=False)

    for metric_name, col in resolved.items():
        mean_col = f"{col}__mean"

        std_col = f"{col}__std"

        plot_df = grouped[["method_name", mean_col, std_col]].copy()

        plot_df = plot_df.sort_values(mean_col, ascending=True)

        plt.figure(figsize=(8, 4.5))

        plt.bar(plot_df["method_name"], plot_df[mean_col], yerr=plot_df[std_col], capsize=4)

        plt.xticks(rotation=25, ha="right")

        plt.ylabel(metric_name)

        plt.title(f"Tail failure summary: {metric_name}")

        plt.tight_layout()

        plt.savefig(outdir / f"tail_{metric_name}.png", dpi=200)

        plt.close()

    # Optional pair plot against pathology score if available

    pathology_col = find_first_existing(df, ["pathology.pathology_score", "pathology_score"])

    if pathology_col is not None:

        df[pathology_col] = pd.to_numeric(df[pathology_col], errors="coerce")

        for metric_name, col in resolved.items():
            plt.figure(figsize=(5, 4))

            plt.scatter(df[pathology_col], df[col], alpha=0.8)

            plt.xlabel("pathology_score")

            plt.ylabel(metric_name)

            plt.title(f"pathology vs {metric_name}")

            plt.tight_layout()

            plt.savefig(outdir / f"pathology_vs_{metric_name}.png", dpi=200)

            plt.close()

    print(f"[OK] Saved tail summaries to: {outdir}")


if __name__ == "__main__":
    main()
