import json
from pathlib import Path

from ptv_ncs_mlip.io import load_archive


def _write_min_archive(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'mp_query.json').write_text(json.dumps({'query_date': None, 'mp_api_version': None}), encoding='utf-8')
    (root / 'mp_screening_pool.csv').write_text('material_id,formula_pretty,structure_class,split,round_id\nmp-1,H2O,molecule,train,1\n', encoding='utf-8')
    (root / 'mp_reference_labels.csv').write_text('material_id,E_hull_ref_mev_atom,d_ref_mev_atom,reference_stable\nmp-1,10,-15,True\n', encoding='utf-8')
    (root / 'surrogate_pool.csv').write_text('surrogate_id,selection_rule,target_preserved,backbone,objective,seed,force_mae_ev_a,energy_mae,in_nominated_target,deployable,hindsight_only\ns1,lowest_mae,False,mace,mae,0,0.028,0.01,False,True,False\n', encoding='utf-8')
    (root / 'mlip_predictions.csv').write_text('round_id,material_id,surrogate_id,d_hat_mev_atom,acquisition_score\n1,mp-1,s1,-5,0.9\n', encoding='utf-8')
    (root / 'selector_profiles.csv').write_text('target_name,family,relevance,auditability,sufficiency,pre_deployment_value,composite_score,admissible,nominated,oracle_family_match\nstability_threshold,regime,0.87,0.87,0.87,0.87,0.87,True,True,True\n', encoding='utf-8')


def test_archive_loads(tmp_path):
    data_root = tmp_path / 'ncs_mlip'
    _write_min_archive(data_root)
    data = load_archive(data_root)
    assert 'mp_reference_labels.csv' in data
