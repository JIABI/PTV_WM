from pathlib import Path
import json
import pandas as pd
from .schema import REQUIRED, SELECTION_RULES, TARGETS, FAMILIES

BANNED_TOKENS = ("placeholder", "todo", "tbd", "example", "dummy", "mp-xxxx", "real mp-id")


def load_archive(data_root, manuscript_strict=False, example_ok=False):
    root = Path(data_root)
    out = {}
    for fn, cols in REQUIRED.items():
        p = root / fn
        if not p.exists():
            raise FileNotFoundError(fn)
        df = pd.read_csv(p)
        miss = [c for c in cols if c not in df.columns]
        if miss:
            raise ValueError(f"{fn} missing {miss}")
        out[fn] = df

    q = root / 'mp_query.json'
    if not q.exists():
        raise FileNotFoundError('mp_query.json')
    out['mp_query.json'] = json.loads(q.read_text(encoding='utf-8'))

    if manuscript_strict:
        _validate_manuscript_strict(root, out)

    _validate_schema(out)
    return out


def _validate_schema(a):
    if set(a['surrogate_pool.csv']['selection_rule']) - SELECTION_RULES:
        raise ValueError('bad selection_rule')
    if set(a['selector_profiles.csv']['target_name']) - TARGETS:
        raise ValueError('bad target_name')
    if set(a['selector_profiles.csv']['family']) - FAMILIES:
        raise ValueError('bad family')


def _validate_manuscript_strict(root: Path, archive):
    required_files = [
        'mp_splits.csv',
        'mlip_hero_case_rounds.csv',
        'mlip_hero_case_summary.csv',
        'thresholds_lock.json',
        'selector_config.yaml',
        'code_commit.txt',
        'heldout_systems.csv',
        'external_systems.csv',
    ]
    missing = [name for name in required_files if not (root / name).exists()]

    q = archive['mp_query.json']
    missing_meta = [k for k in ('query_date', 'mp_api_version', 'code_commit') if k not in q or q[k] in (None, '')]

    if missing or missing_meta:
        parts = []
        if missing:
            parts.append('missing required files: ' + ', '.join(missing))
        if missing_meta:
            parts.append('missing metadata: ' + ', '.join(missing_meta))
        raise ValueError('; '.join(parts))

    for file_name, df in archive.items():
        if isinstance(df, pd.DataFrame):
            _check_banned_tokens(df, file_name)
            _check_fake_mpid(df, file_name)


def _check_banned_tokens(df: pd.DataFrame, file_name: str):
    for col in df.columns:
        s = df[col].astype(str).str.lower()
        for tok in BANNED_TOKENS:
            if s.str.contains(tok, regex=False).any():
                raise ValueError(f'{file_name}:{col} contains banned token {tok}')


def _check_fake_mpid(df: pd.DataFrame, file_name: str):
    mp_cols = [c for c in df.columns if c.lower() in ('mp-id', 'mp_id', 'material_id')]
    for col in mp_cols:
        s = df[col].astype(str)
        if s.str.contains(r'^(C1|C2|C3)$', regex=True).any():
            raise ValueError(f'{file_name}:{col} contains fake C1/C2/C3 ids')
        if s.str.contains('mp-XXXX', case=False, regex=False).any():
            raise ValueError(f'{file_name}:{col} contains mp-XXXX')
        if s.str.contains('real MP-ID', case=False, regex=False).any():
            raise ValueError(f'{file_name}:{col} contains real MP-ID placeholder')
