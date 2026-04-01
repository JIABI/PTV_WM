import argparse

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


def plot_1d(df: pd.DataFrame, xcol: str, ycol: str, outpath: Path, title: str) -> None:
    plot_df = df[[xcol, ycol]].copy()

    plot_df[xcol] = pd.to_numeric(plot_df[xcol], errors="coerce")

    plot_df[ycol] = pd.to_numeric(plot_df[ycol], errors="coerce")

    plot_df = plot_df.dropna().sort_values(xcol)

    if len(plot_df) == 0:
        return

    plt.figure(figsize=(6, 4))

    plt.plot(plot_df[xcol], plot_df[ycol], marker="o")

    plt.xlabel(xcol)

    plt.ylabel(ycol)

    plt.title(title)

    plt.tight_layout()

    plt.savefig(outpath, dpi=200)

    plt.close()


def plot_phase_regions(df: pd.DataFrame, xcol: str, phase_col: str, outpath: Path, title: str) -> None:
    plot_df = df[[xcol, phase_col]].copy()

    plot_df[xcol] = pd.to_numeric(plot_df[xcol], errors="coerce")

    plot_df = plot_df.dropna().sort_values(xcol)

    if len(plot_df) == 0:
        return

    phase_to_num = {

        "trainable": 0,

        "rescue": 1,

        "failure": 2,

    }

    plot_df["phase_num"] = plot_df[phase_col].astype(str).map(phase_to_num)

    plt.figure(figsize=(6, 4))

    plt.scatter(plot_df[xcol], plot_df["phase_num"], c=plot_df["phase_num"], s=80)

    plt.yticks([0, 1, 2], ["trainable", "rescue", "failure"])

    plt.xlabel(xcol)

    plt.ylabel("phase_region")

    plt.title(title)

    plt.tight_layout()

    plt.savefig(outpath, dpi=200)

    plt.close()


def plot_simplex_scatter(

        df: pd.DataFrame,

        alpha_col: str,

        beta_col: str,

        color_col: str,

        outpath: Path,

        title: str,

) -> None:
    plot_df = df[[alpha_col, beta_col, color_col]].copy()

    plot_df[alpha_col] = pd.to_numeric(plot_df[alpha_col], errors="coerce")

    plot_df[beta_col] = pd.to_numeric(plot_df[beta_col], errors="coerce")

    plot_df[color_col] = pd.to_numeric(plot_df[color_col], errors="coerce")

    plot_df = plot_df.dropna()

    if len(plot_df) == 0:
        return

    plt.figure(figsize=(6, 5))

    sc = plt.scatter(plot_df[alpha_col], plot_df[beta_col], c=plot_df[color_col], s=80)

    plt.xlabel(alpha_col)

    plt.ylabel(beta_col)

    plt.title(title)

    plt.colorbar(sc, label=color_col)

    plt.tight_layout()

    plt.savefig(outpath, dpi=200)

    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate phase-diagram style plots from atlas outputs.")

    parser.add_argument("--input", required=True, help="Input atlas table (csv/pkl/parquet).")

    parser.add_argument("--outdir", required=True, help="Output directory.")

    args = parser.parse_args()

    outdir = Path(args.outdir)

    ensure_outdir(outdir)

    df = load_table(Path(args.input))

    alpha_col = find_first_existing(

        df,

        [

            "prediction_spec.params.alpha",

            "target_lambda.alpha",

            "lambda.alpha",

            "alpha",

        ],

    )

    beta_col = find_first_existing(

        df,

        [

            "prediction_spec.params.beta",

            "target_lambda.beta",

            "lambda.beta",

            "beta",

        ],

    )

    gamma_col = find_first_existing(

        df,

        [

            "prediction_spec.params.gamma",

            "target_lambda.gamma",

            "lambda.gamma",

            "gamma",

        ],

    )

    pathology_col = find_first_existing(df, ["pathology.pathology_score", "pathology_score"])

    psnr_col = find_first_existing(df, ["quality.psnr", "psnr"])

    collapse_col = find_first_existing(df, ["quality.collapse_rate", "collapse_rate"])

    feature_fd_col = find_first_existing(df, ["quality.feature_fd", "feature_fd"])

    phase_col = find_first_existing(df, ["phase_region", "phase.region", "atlas.phase_region"])

    resolution_col = find_first_existing(df, ["resolution", "dataset.resolution", "exp.resolution"])

    # 1D line-style plots if alpha exists

    if alpha_col is not None:

        if pathology_col is not None:
            plot_1d(

                df, alpha_col, pathology_col,

                outdir / "phase_alpha_vs_pathology.png",

                "alpha vs pathology_score"

            )

        if psnr_col is not None:
            plot_1d(

                df, alpha_col, psnr_col,

                outdir / "phase_alpha_vs_psnr.png",

                "alpha vs PSNR"

            )

        if collapse_col is not None:
            plot_1d(

                df, alpha_col, collapse_col,

                outdir / "phase_alpha_vs_collapse.png",

                "alpha vs collapse_rate"

            )

        if feature_fd_col is not None:
            plot_1d(

                df, alpha_col, feature_fd_col,

                outdir / "phase_alpha_vs_feature_fd.png",

                "alpha vs feature_fd"

            )

        if phase_col is not None:
            plot_phase_regions(

                df, alpha_col, phase_col,

                outdir / "phase_alpha_vs_region.png",

                "alpha vs phase_region"

            )

    # Simplex scatter if alpha and beta exist

    if alpha_col is not None and beta_col is not None:

        if pathology_col is not None:
            plot_simplex_scatter(

                df, alpha_col, beta_col, pathology_col,

                outdir / "simplex_pathology_scatter.png",

                "Simplex scatter colored by pathology"

            )

        if psnr_col is not None:
            plot_simplex_scatter(

                df, alpha_col, beta_col, psnr_col,

                outdir / "simplex_psnr_scatter.png",

                "Simplex scatter colored by PSNR"

            )

        if collapse_col is not None:
            plot_simplex_scatter(

                df, alpha_col, beta_col, collapse_col,

                outdir / "simplex_collapse_scatter.png",

                "Simplex scatter colored by collapse"

            )

    # Resolution faceting if both alpha and resolution exist

    if alpha_col is not None and resolution_col is not None and pathology_col is not None:

        unique_res = sorted(df[resolution_col].dropna().unique())

        if len(unique_res) > 1:

            n = len(unique_res)

            fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

            axes = axes[0]

            for ax, r in zip(axes, unique_res):

                sub = df[df[resolution_col] == r].copy()

                sub[alpha_col] = pd.to_numeric(sub[alpha_col], errors="coerce")

                sub[pathology_col] = pd.to_numeric(sub[pathology_col], errors="coerce")

                sub = sub.dropna().sort_values(alpha_col)

                if len(sub) > 0:
                    ax.plot(sub[alpha_col], sub[pathology_col], marker="o")

                ax.set_title(f"resolution={r}")

                ax.set_xlabel(alpha_col)

                ax.set_ylabel(pathology_col)

            plt.tight_layout()

            plt.savefig(outdir / "phase_resolution_facets.png", dpi=200)

            plt.close()

    print(f"[OK] Saved phase diagrams to: {outdir}")


if __name__ == "__main__":
    main()
