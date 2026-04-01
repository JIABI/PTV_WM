import glob
import json
import os
from pathlib import Path
import pandas as pd

SECTION_MAP = {
    'main_section5.csv': ['all_runs.csv'],
    'main_section6.csv': ['eval_predictive.json', 'eval_matched_geometry.json', 'eval_counterfactual.json'],
    'main_section7.csv': ['eval_predictive.json'],
    'main_section8.csv': ['eval_control.json'],
    'main_section9.csv': ['eval_antisteg.json', 'capacity_sweep.json', 'no_event.json', 'ood_stress.json', 'leakage.json'],
    'si_a.csv': ['all_runs.csv'],
    'si_b.csv': ['all_runs.csv'],
    'si_c.csv': ['eval_predictive.json', 'eval_matched_geometry.json'],
    'si_d.csv': ['eval_counterfactual.json'],
    'si_e.csv': ['eval_predictive.json'],
    'si_f.csv': ['eval_control.json'],
    'si_g.csv': ['capacity_sweep.json', 'bounded_residual.json', 'oracle_runs.json'],
    'si_h.csv': ['eval_antisteg.json', 'leakage.json'],
    'si_i.csv': ['no_event.json', 'ood_stress.json'],
    'si_capacity.csv': ['capacity_sweep.json'],
    'si_counterfactual.csv': ['eval_counterfactual.json'],
    'si_ood.csv': ['ood_stress.json'],
}


def _safe_load_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            raw = f.read().strip()
        if not raw:
            return None
        obj = json.loads(raw)
    except Exception:
        return None
    if isinstance(obj, dict):
        obj['_source'] = os.path.basename(path)
    return obj


def _collect_run_metrics(run_dir: str):
    rows = []
    for p in glob.glob(f"{run_dir}/*/metrics.json"):
        obj = _safe_load_json(p)
        if not isinstance(obj, dict):
            continue
        obj['run'] = Path(p).parent.name
        rows.append(obj)
    return rows


def _collect_raw_for_section(section: str, out_dir: str):
    raw_dir = Path(out_dir) / 'raw'
    if not raw_dir.exists():
        return []
    rows = []
    for p in sorted(raw_dir.glob(f'{section}__*.json')):
        obj = _safe_load_json(str(p))
        if obj is None:
            continue
        if isinstance(obj, list):
            rows.extend(obj)
        else:
            rows.append(obj)
    return rows


def aggregate_runs(run_dir: str = 'outputs/runs', out_dir: str = 'outputs/aggregates', sections: list[str] | None = None, verbose: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    run_rows = _collect_run_metrics(run_dir)
    all_runs_path = os.path.join(out_dir, 'all_runs.csv')
    if run_rows:
        pd.DataFrame(run_rows).to_csv(all_runs_path, index=False)

    requested = sections or list(SECTION_MAP.keys())
    aggregated = []
    for out_csv in requested:
        section = out_csv.replace('.csv', '')
        rows = []
        if out_csv in {'main_section5.csv', 'si_a.csv', 'si_b.csv'} and os.path.exists(all_runs_path):
            df = pd.read_csv(all_runs_path)
            if 'section' in df.columns:
                sec_df = df[df['section'] == section]
                if not sec_df.empty:
                    rows.extend(sec_df.to_dict(orient='records'))
        rows.extend(_collect_raw_for_section(section, out_dir))
        out_path = os.path.join(out_dir, out_csv)
        if rows:
            pd.json_normalize(rows).to_csv(out_path, index=False)
            aggregated.append(out_csv)
        else:
            if verbose:
                print(f"[aggregate] no valid inputs for {out_csv}; skipping")
    if aggregated:
        print('[aggregate] wrote:', ', '.join(aggregated))
    else:
        print('[aggregate] no section outputs were written')
