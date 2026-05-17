from pathlib import Path
import json, pandas as pd
from .schema import REQUIRED, SELECTION_RULES, TARGETS, FAMILIES

def load_archive(data_root):
    root=Path(data_root)
    out={}
    for fn,cols in REQUIRED.items():
        p=root/fn
        if not p.exists(): raise FileNotFoundError(fn)
        df=pd.read_csv(p)
        miss=[c for c in cols if c not in df.columns]
        if miss: raise ValueError(f"{fn} missing {miss}")
        out[fn]=df
    q=root/'mp_query.json'
    out['mp_query.json']=json.loads(q.read_text()) if q.exists() else {}
    _validate(out)
    return out

def _validate(a):
    if set(a['surrogate_pool.csv']['selection_rule'])-SELECTION_RULES: raise ValueError('bad selection_rule')
    if set(a['selector_profiles.csv']['target_name'])-TARGETS: raise ValueError('bad target_name')
    if set(a['selector_profiles.csv']['family'])-FAMILIES: raise ValueError('bad family')
