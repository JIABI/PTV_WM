from __future__ import annotations
import argparse
from pathlib import Path
import json
import math
import pandas as pd
import matplotlib.pyplot as plt


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # suffix match fallback
    for cand in candidates:
        for c in cols:
            if c.endswith(cand):
                return c
    return None


def coerce_numeric(df: pd.DataFrame, col: str | None) -> pd.Series | None:
    if col is None:
        return None
    s = pd.to_numeric(df[col], errors='coerce')
    if s.notna().sum() == 0:
        return None
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, required=True)
    ap.add_argument('--outdir', type=str, required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    alpha_col = find_col(df, ['params.alpha', 'alpha'])
    family_col = find_col(df, ['pathology.family', 'quality.family', 'trainability.family', 'family'])
    phase_col = find_col(df, ['phase_region'])

    pathology_col = find_col(df, [
        'pathology.pathology_score', 'loss_spec.pathology_score', 'trainability.pathology_score', 'quality.pathology_score', 'tail.pathology_score',
        'pathology_score'
    ])
    collapse_col = find_col(df, [
        'quality.collapse_rate', 'trainability.collapse_rate', 'loss_spec.collapse_rate', 'tail.collapse_rate', 'collapse_rate'
    ])
    psnr_col = find_col(df, [
        'quality.psnr', 'trainability.psnr', 'loss_spec.psnr', 'tail.psnr', 'psnr'
    ])
    featfd_col = find_col(df, [
        'quality.feature_fd', 'trainability.feature_fd', 'loss_spec.feature_fd', 'tail.feature_fd', 'feature_fd'
    ])
    rho_nor_col = find_col(df, [
        'pathology.rho_nor', 'loss_spec.rho_nor', 'trainability.rho_nor', 'quality.rho_nor', 'tail.rho_nor', 'rho_nor'
    ])
    cond_col = find_col(df, [
        'pathology.conditioning', 'loss_spec.conditioning', 'trainability.conditioning', 'quality.conditioning', 'tail.conditioning', 'conditioning'
    ])

    alpha = coerce_numeric(df, alpha_col)
    pathology = coerce_numeric(df, pathology_col)
    collapse = coerce_numeric(df, collapse_col)
    psnr = coerce_numeric(df, psnr_col)
    featfd = coerce_numeric(df, featfd_col)
    rho_nor = coerce_numeric(df, rho_nor_col)
    cond = coerce_numeric(df, cond_col)

    summary = {
        'rows': int(len(df)),
        'detected_columns': {
            'alpha': alpha_col,
            'family': family_col,
            'phase_region': phase_col,
            'pathology_score': pathology_col,
            'collapse_rate': collapse_col,
            'psnr': psnr_col,
            'feature_fd': featfd_col,
            'rho_nor': rho_nor_col,
            'conditioning': cond_col,
        }
    }

    # Plot 1: pathology vs PSNR
    if pathology is not None and psnr is not None:
        tmp = pd.DataFrame({'pathology': pathology, 'psnr': psnr}).dropna()
        if len(tmp) > 0:
            plt.figure(figsize=(5.2, 4.0))
            plt.scatter(tmp['pathology'], tmp['psnr'], alpha=0.8)
            plt.xlabel('pathology score')
            plt.ylabel('PSNR')
            plt.title('Pathology vs PSNR')
            plt.tight_layout()
            plt.savefig(outdir / 'pathology_vs_psnr.png', dpi=200)
            plt.close()
            summary['pathology_vs_psnr_corr'] = float(tmp.corr(numeric_only=True).loc['pathology', 'psnr'])

    # Plot 2: pathology vs collapse
    if pathology is not None and collapse is not None:
        tmp = pd.DataFrame({'pathology': pathology, 'collapse': collapse}).dropna()
        if len(tmp) > 0:
            plt.figure(figsize=(5.2, 4.0))
            plt.scatter(tmp['pathology'], tmp['collapse'], alpha=0.8)
            plt.xlabel('pathology score')
            plt.ylabel('collapse rate')
            plt.title('Pathology vs Collapse')
            plt.tight_layout()
            plt.savefig(outdir / 'pathology_vs_collapse.png', dpi=200)
            plt.close()
            summary['pathology_vs_collapse_corr'] = float(tmp.corr(numeric_only=True).loc['pathology', 'collapse'])

    # Plot 3: alpha trend
    if alpha is not None:
        for y_name, y in [('psnr', psnr), ('feature_fd', featfd), ('collapse', collapse), ('pathology', pathology)]:
            if y is None:
                continue
            tmp = pd.DataFrame({'alpha': alpha, y_name: y}).dropna()
            if len(tmp) == 0:
                continue
            grp = tmp.groupby('alpha', as_index=False).mean(numeric_only=True).sort_values('alpha')
            plt.figure(figsize=(5.4, 4.0))
            plt.plot(grp['alpha'], grp[y_name], marker='o')
            plt.xlabel('alpha')
            plt.ylabel(y_name)
            plt.title(f'alpha vs {y_name}')
            plt.tight_layout()
            plt.savefig(outdir / f'alpha_vs_{y_name}.png', dpi=200)
            plt.close()

    # Plot 4: phase region counts
    if phase_col is not None:
        counts = df[phase_col].astype(str).value_counts(dropna=False)
        plt.figure(figsize=(5.0, 4.0))
        plt.bar(counts.index.astype(str), counts.values)
        plt.xlabel('phase region')
        plt.ylabel('count')
        plt.title('Phase region counts')
        plt.tight_layout()
        plt.savefig(outdir / 'phase_region_counts.png', dpi=200)
        plt.close()
        summary['phase_region_counts'] = {str(k): int(v) for k, v in counts.items()}

    # Plot 5: rho_nor vs pathology
    if rho_nor is not None and pathology is not None:
        tmp = pd.DataFrame({'rho_nor': rho_nor, 'pathology': pathology}).dropna()
        if len(tmp) > 0:
            plt.figure(figsize=(5.2, 4.0))
            plt.scatter(tmp['rho_nor'], tmp['pathology'], alpha=0.8)
            plt.xlabel('rho_nor')
            plt.ylabel('pathology score')
            plt.title('Normal burden vs Pathology')
            plt.tight_layout()
            plt.savefig(outdir / 'rho_nor_vs_pathology.png', dpi=200)
            plt.close()
            summary['rho_nor_vs_pathology_corr'] = float(tmp.corr(numeric_only=True).loc['rho_nor', 'pathology'])

    # Simple textual ranking table
    keep = {}
    for key, series in [('alpha', alpha), ('pathology_score', pathology), ('psnr', psnr), ('feature_fd', featfd), ('collapse_rate', collapse)]:
        if series is not None:
            keep[key] = series
    if keep:
        tdf = pd.DataFrame(keep)
        tdf = tdf.dropna(subset=[c for c in ['pathology_score'] if c in tdf.columns], how='any')
        if len(tdf) > 0:
            rank_cols = [c for c in ['alpha', 'pathology_score', 'psnr', 'feature_fd', 'collapse_rate'] if c in tdf.columns]
            tdf[rank_cols].sort_values('pathology_score').to_csv(outdir / 'ranked_by_pathology.csv', index=False)

    with open(outdir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved plots to: {outdir}")


if __name__ == '__main__':
    main()
