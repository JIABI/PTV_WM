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
    candidates = [

        "mode",

        "method",

        "runner",

        "train.mode",

        "exp_id",

    ]

    col = find_first_existing(df, candidates)

    if col is not None:
        return df[col].astype(str)

    # fallback: derive from filename-ish columns if present

    return pd.Series(["unknown"] * len(df), index=df.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple run result tables.")

    parser.add_argument(

        "--inputs",

        nargs="+",

        required=True,

        help="Paths to result tables (csv/pkl/parquet).",

    )

    parser.add_argument(

        "--outdir",

        required=True,

        help="Directory to save summary tables and plots.",

    )

    args = parser.parse_args()

    outdir = Path(args.outdir)

    ensure_outdir(outdir)

    dfs = []

    for inp in args.inputs:

        path = Path(inp)

        df = load_table(path)

        df = df.copy()

        if "source_file" not in df.columns:
            df["source_file"] = str(path)

        if "method_name" not in df.columns:
            df["method_name"] = build_method_column(df)

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    metric_candidates = {

        "fid": ["quality.fid", "fid"],

        "sfid": ["quality.sfid", "sfid"],

        "lpips": ["quality.lpips", "lpips"],

        "psnr": ["quality.psnr", "psnr"],

        "feature_fd": ["quality.feature_fd", "feature_fd"],

        "collapse_rate": ["quality.collapse_rate", "collapse_rate"],

        "rare_failure_rate": ["tail.rare_failure_rate", "rare_failure_rate"],

        "worst_k_score": ["tail.worst_k_score", "worst_k_score"],

        "percentile_95": ["tail.percentile_95", "percentile_95"],

        "percentile_99": ["tail.percentile_99", "percentile_99"],

        "pathology_score": ["pathology.pathology_score", "pathology_score"],

        "converged": ["trainability.converged", "converged"],

    }

    resolved = {}

    for name, candidates in metric_candidates.items():

        col = find_first_existing(df, candidates)

        if col is not None:
            resolved[name] = col

    summary_frames = []

    group_col = "method_name"

    agg_dict = {}

    for metric_name, col in resolved.items():

        if metric_name == "converged":

            df[col] = df[col].astype(float)

        else:

            df[col] = pd.to_numeric(df[col], errors="coerce")

        agg_dict[col] = ["mean", "std", "count"]

    grouped = df.groupby(group_col).agg(agg_dict)

    grouped.columns = [

        f"{orig_col}__{stat}" for orig_col, stat in grouped.columns.to_flat_index()

    ]

    grouped = grouped.reset_index()

    grouped.to_csv(outdir / "run_comparison_summary.csv", index=False)

    with open(outdir / "resolved_columns.json", "w") as f:

        json.dump(resolved, f, indent=2)

    # Build a readable ranking table

    rank_rows = []

    for method in grouped[group_col].tolist():
        row = grouped[grouped[group_col] == method].iloc[0].to_dict()

        rank_rows.append(row)

    ranking_df = pd.DataFrame(rank_rows)

    # Sort by best available metric

    if "fid" in resolved:

        fid_col = f"{resolved['fid']}__mean"

        ranking_df = ranking_df.sort_values(fid_col, ascending=True)

    elif "psnr" in resolved:

        psnr_col = f"{resolved['psnr']}__mean"

        ranking_df = ranking_df.sort_values(psnr_col, ascending=False)

    elif "pathology_score" in resolved:

        p_col = f"{resolved['pathology_score']}__mean"

        ranking_df = ranking_df.sort_values(p_col, ascending=True)

    ranking_df.to_csv(outdir / "run_ranking.csv", index=False)

    # Plot a few key metrics if present

    plot_metrics = ["fid", "psnr", "collapse_rate", "rare_failure_rate", "pathology_score"]

    for metric_name in plot_metrics:

        if metric_name not in resolved:
            continue

        col = resolved[metric_name]

        mean_col = f"{col}__mean"

        std_col = f"{col}__std"

        plot_df = grouped[[group_col, mean_col, std_col]].copy()

        plot_df = plot_df.sort_values(mean_col, ascending=(metric_name not in ["psnr", "converged"]))

        plt.figure(figsize=(8, 4.5))

        plt.bar(plot_df[group_col], plot_df[mean_col], yerr=plot_df[std_col], capsize=4)

        plt.xticks(rotation=25, ha="right")

        plt.ylabel(metric_name)

        plt.title(f"Run comparison: {metric_name}")

        plt.tight_layout()

        plt.savefig(outdir / f"compare_{metric_name}.png", dpi=200)

        plt.close()

    print(f"[OK] Saved comparison summary to: {outdir}")


if __name__ == "__main__":
    main()
