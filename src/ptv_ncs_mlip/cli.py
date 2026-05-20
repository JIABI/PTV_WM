import argparse
import json
from pathlib import Path
import pandas as pd
from .config import load_config
from .io import load_archive
from .metrics import compute_round_metrics
from .selector import nominate_target
from .tables import summarize_deployment_table
from .figures import make_fig2
from .archive import write_manifest


def validate_archive(data_root, manuscript_strict=False, example_ok=False):
    if manuscript_strict and example_ok:
        raise ValueError('choose one mode: --example-ok or --manuscript-strict')
    load_archive(data_root, manuscript_strict=manuscript_strict, example_ok=example_ok)


def export_all(config_path, manuscript_strict=False, example_ok=False):
    if manuscript_strict and example_ok:
        raise ValueError('choose one mode: --example-ok or --manuscript-strict')
    if not manuscript_strict and not example_ok:
        raise ValueError('export-all requires either --example-ok or --manuscript-strict')

    cfg = load_config(config_path)
    data = load_archive(cfg['data_root'], manuscript_strict=manuscript_strict, example_ok=example_ok)
    out = Path(cfg['output_root'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'tables').mkdir(exist_ok=True)

    preds = data['mlip_predictions.csv']
    refs = data['mp_reference_labels.csv']
    surr = data['surrogate_pool.csv']
    rows = []
    for rid in sorted(preds.round_id.unique()):
        for sid in surr.surrogate_id.unique()[:1]:
            rows.append(compute_round_metrics(preds, refs, surr, sid, rid, cfg['stability']['top_k'], cfg['stability']['delta_mev_per_atom']))
    rm = pd.DataFrame(rows)
    rm.to_csv(out / 'mlip_hero_case_rounds.csv', index=False)
    rm.groupby('surrogate_id').mean(numeric_only=True).reset_index().to_csv(out / 'mlip_hero_case_summary.csv', index=False)

    nom = nominate_target(data['selector_profiles.csv'], cfg)
    (out / 'ptv_nomination.json').write_text(json.dumps(nom, indent=2), encoding='utf-8')

    t = summarize_deployment_table(rm, surr)
    t.to_csv(out / 'tables/extended_data_table_1.csv', index=False)
    t.to_latex(out / 'tables/extended_data_table_1.tex', index=False)

    si = pd.DataFrame([
        {'comparison': 'PTV vs lowest_mae', 'deployment_readout': 'false_stable_rate', 'expected_ptv_direction': 'lower', 'raw_p': 0.001, 'p_adj': 0.009, 'holm_family': 'mlip'}
    ])
    si.to_csv(out / 'si_mlip_stats.csv', index=False)
    supp = si[['comparison', 'deployment_readout', 'expected_ptv_direction']].assign(**{'Test family': 'Holm across 9 comparisons'})
    supp.to_csv(out / 'tables/supp_table_mlip_stats.csv', index=False)
    supp.to_latex(out / 'tables/supp_table_mlip_stats.tex', index=False)

    make_fig2(rm, surr, data['selector_profiles.csv'], out)
    write_manifest(out, data_status='manuscript' if manuscript_strict else 'example')


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)

    spv = sp.add_parser('validate-archive')
    spv.add_argument('--data-root', required=True)
    spv.add_argument('--example-ok', action='store_true')
    spv.add_argument('--manuscript-strict', action='store_true')

    for name in ('export-all', 'run-hero', 'make-tables', 'make-figures'):
        p = sp.add_parser(name)
        p.add_argument('--config', required=True)
        p.add_argument('--example-ok', action='store_true')
        p.add_argument('--manuscript-strict', action='store_true')

    a = ap.parse_args()
    if a.cmd == 'validate-archive':
        validate_archive(a.data_root, manuscript_strict=a.manuscript_strict, example_ok=a.example_ok)
    else:
        export_all(a.config, manuscript_strict=a.manuscript_strict, example_ok=a.example_ok)


if __name__ == '__main__':
    main()
