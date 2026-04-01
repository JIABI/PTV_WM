from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == '.csv':
        return pd.read_csv(path)
    if path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.suffix == '.pkl':
        return pd.read_pickle(path)
    raise ValueError(f'Unsupported file: {path}')


def collect_summaries(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(root.rglob('summary.json')):
        with p.open('r', encoding='utf-8') as f:
            row = json.load(f)
        row['run_dir'] = str(p.parent)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f'No summary.json files found under {root}')
    return pd.json_normalize(rows)


def _find(df: pd.DataFrame, *cands: str) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def _numeric(df: pd.DataFrame, col: str | None) -> pd.Series | None:
    if col is None:
        return None
    s = pd.to_numeric(df[col], errors='coerce')
    return s if s.notna().any() else None


def export_line_family(df: pd.DataFrame, outdir: Path) -> None:
    alpha_col = _find(df, 'prediction_spec.params.alpha', 'loss_spec.params.alpha')
    psnr_col = _find(df, 'quality.psnr')
    fd_col = _find(df, 'quality.feature_fd')
    conv_col = _find(df, 'trainability.converged')
    if alpha_col is None:
        return
    use = pd.DataFrame({
        'alpha': _numeric(df, alpha_col),
        'psnr': _numeric(df, psnr_col),
        'feature_fd': _numeric(df, fd_col),
        'converged': _numeric(df, conv_col),
    }).dropna(subset=['alpha'])
    agg = use.groupby('alpha', as_index=False).agg({'psnr': ['mean', 'std'], 'feature_fd': ['mean', 'std'], 'converged': ['mean', 'std', 'count']})
    agg.columns = ['alpha', 'psnr_mean', 'psnr_std', 'feature_fd_mean', 'feature_fd_std', 'conv_mean', 'conv_std', 'n']
    agg.to_csv(outdir / 'line_family_summary.csv', index=False)


def export_phase_table(atlas_path: Path, outdir: Path) -> None:
    df = read_table(atlas_path)
    cols = [c for c in ['phase_region', 'quality.mse', 'tail.rare_failure_rate', 'pathology.pathology_score', 'prediction_spec.family'] if c in df.columns]
    df[cols].to_csv(outdir / 'atlas_phase_table.csv', index=False)


def export_method_table(df: pd.DataFrame, outdir: Path) -> None:
    mode_col = _find(df, 'mode')
    if mode_col is None:
        return
    cols = {
        'overall_conv': _find(df, 'trainability.converged'),
        'collapse_rate': _find(df, 'trainability.collapse_rate'),
        'feature_fd': _find(df, 'quality.feature_fd'),
        'psnr': _find(df, 'quality.psnr'),
        'worst_k': _find(df, 'tail.worst_k_score'),
        'pathology': _find(df, 'pathology.pathology_score'),
    }
    use = pd.DataFrame({'mode': df[mode_col].astype(str)})
    for k, v in cols.items():
        use[k] = _numeric(df, v)
    agg = use.groupby('mode', as_index=False).agg(['mean', 'std'])
    agg.columns = ['_'.join([c for c in tup if c]).strip('_') for tup in agg.columns.to_flat_index()]
    agg.to_csv(outdir / 'method_summary.csv', index=False)


def export_tail_table(df: pd.DataFrame, outdir: Path) -> None:
    mode_col = _find(df, 'mode')
    if mode_col is None:
        return
    use = pd.DataFrame({'mode': df[mode_col].astype(str)})
    for p in [90, 95, 99]:
        col = _find(df, f'tail.percentile_{p}')
        if col:
            use[f'p{p}'] = _numeric(df, col)
    wk = _find(df, 'tail.worst_k_score')
    if wk:
        use['worst_k'] = _numeric(df, wk)
    use.to_csv(outdir / 'tail_records.csv', index=False)


def plot_if_possible(df: pd.DataFrame, x: str, y: str, out: Path, title: str) -> None:
    if x not in df.columns or y not in df.columns:
        return
    plt.figure(figsize=(5.2, 4.0))
    plt.plot(df[x], df[y], marker='o')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description='Export plot-ready tables for the paper.')
    ap.add_argument('--summary-root', required=True, help='Root directory containing run summaries.')
    ap.add_argument('--atlas', default=None, help='Optional atlas table path.')
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summaries = collect_summaries(Path(args.summary_root))
    summaries.to_csv(outdir / 'all_summaries_flat.csv', index=False)
    export_line_family(summaries, outdir)
    export_method_table(summaries, outdir)
    export_tail_table(summaries, outdir)

    line_csv = outdir / 'line_family_summary.csv'
    if line_csv.exists():
        line_df = pd.read_csv(line_csv)
        plot_if_possible(line_df, 'alpha', 'conv_mean', outdir / 'line_conv.png', 'Line-family convergence')
        plot_if_possible(line_df, 'alpha', 'psnr_mean', outdir / 'line_psnr.png', 'Line-family PSNR')
        plot_if_possible(line_df, 'alpha', 'feature_fd_mean', outdir / 'line_feature_fd.png', 'Line-family feature FD')

    if args.atlas:
        export_phase_table(Path(args.atlas), outdir)

    with (outdir / 'paper_results_manifest.json').open('w', encoding='utf-8') as f:
        json.dump({
            'summary_root': args.summary_root,
            'atlas': args.atlas,
            'exports': sorted([p.name for p in outdir.iterdir() if p.is_file()]),
        }, f, indent=2)


if __name__ == '__main__':
    main()
