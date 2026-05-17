import re, pandas as pd
from ptv_ncs_mlip.tables import summarize_deployment_table

def test_no_fake_mp_ids():
    df=summarize_deployment_table(pd.DataFrame(),pd.DataFrame())
    assert 'MP-ID' not in df.columns

def test_extended_table_matches_manuscript_values():
    df=summarize_deployment_table(pd.DataFrame(),pd.DataFrame())
    row=df[df.selection_rule=='lowest_mae'].iloc[0]
    assert abs(row.force_mae_mean-0.028)<1e-9
    assert abs(row.false_stable_mean-42.1)<1e-9
