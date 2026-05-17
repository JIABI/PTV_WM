import json
from pathlib import Path

import pytest

from ptv_ncs_mlip.archive import write_manifest
from ptv_ncs_mlip.io import load_archive


def _write_min_archive(root: Path, with_placeholders=False, fake_mp=False, strict_meta=False):
    root.mkdir(parents=True, exist_ok=True)
    q = {'query_date': None, 'mp_api_version': None}
    if strict_meta:
        q = {'query_date': '2026-05-01', 'mp_api_version': 'v1', 'code_commit': 'abc123'}
    (root / 'mp_query.json').write_text(json.dumps(q), encoding='utf-8')

    mid = 'mp-1'
    if fake_mp:
        mid = 'mp-XXXX'
    formula = 'H2O'
    if with_placeholders:
        formula = 'placeholder'

    (root / 'mp_screening_pool.csv').write_text(f'material_id,formula_pretty,structure_class,split,round_id\n{mid},{formula},molecule,train,1\n', encoding='utf-8')
    (root / 'mp_reference_labels.csv').write_text(f'material_id,E_hull_ref_mev_atom,d_ref_mev_atom,reference_stable\n{mid},10,-15,True\n', encoding='utf-8')
    (root / 'surrogate_pool.csv').write_text('surrogate_id,selection_rule,target_preserved,backbone,objective,seed,force_mae_ev_a,energy_mae,in_nominated_target,deployable,hindsight_only\ns1,lowest_mae,False,mace,mae,0,0.028,0.01,False,True,False\n', encoding='utf-8')
    (root / 'mlip_predictions.csv').write_text(f'round_id,material_id,surrogate_id,d_hat_mev_atom,acquisition_score\n1,{mid},s1,-5,0.9\n', encoding='utf-8')
    (root / 'selector_profiles.csv').write_text('target_name,family,relevance,auditability,sufficiency,pre_deployment_value,composite_score,admissible,nominated,oracle_family_match\nstability_threshold,regime,0.87,0.87,0.87,0.87,0.87,True,True,True\n', encoding='utf-8')


def _write_strict_required(root: Path):
    for name in [
        'mp_splits.csv',
        'mlip_hero_case_rounds.csv',
        'mlip_hero_case_summary.csv',
        'representative_false_stable_cases.csv',
        'heldout_systems.csv',
        'external_systems.csv',
    ]:
        (root / name).write_text('ok\n', encoding='utf-8')
    (root / 'thresholds_lock.json').write_text('{}', encoding='utf-8')
    (root / 'selector_config.yaml').write_text('selector: {}\n', encoding='utf-8')


def test_example_mode_allows_smoke_data(tmp_path):
    data_root = tmp_path / 'ncs_mlip'
    _write_min_archive(data_root)
    data = load_archive(data_root, example_ok=True)
    assert 'mp_reference_labels.csv' in data


def test_manuscript_strict_rejects_placeholder_data(tmp_path):
    data_root = tmp_path / 'ncs_mlip'
    _write_min_archive(data_root, with_placeholders=True, strict_meta=True)
    _write_strict_required(data_root)
    with pytest.raises(ValueError, match='banned token'):
        load_archive(data_root, manuscript_strict=True)


def test_fake_mp_ids_rejected_in_manuscript_strict(tmp_path):
    data_root = tmp_path / 'ncs_mlip'
    _write_min_archive(data_root, fake_mp=True, strict_meta=True)
    _write_strict_required(data_root)
    with pytest.raises(ValueError, match='mp-xxxx'):
        load_archive(data_root, manuscript_strict=True)


def test_manifest_marks_example_not_manuscript(tmp_path):
    out = tmp_path / 'out'
    out.mkdir(parents=True)
    (out / 'a.csv').write_text('x\n', encoding='utf-8')
    write_manifest(out, data_status='example')
    m = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    assert m['data_status'] == 'example'
    assert m['manuscript_reproducible'] is False
