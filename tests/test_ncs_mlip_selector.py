import pandas as pd
from ptv_ncs_mlip.selector import nominate_target

def test_selector_nominates_stability_threshold():
    df=pd.DataFrame([
        {'target_name':'average_ef','family':'baseline','relevance':0.34,'auditability':1,'sufficiency':1,'pre_deployment_value':1},
        {'target_name':'tail_force','family':'boundary','relevance':0.75,'auditability':0.75,'sufficiency':0.75,'pre_deployment_value':0.75},
        {'target_name':'rank_margin','family':'criticality','relevance':0.64,'auditability':0.64,'sufficiency':0.64,'pre_deployment_value':0.64},
        {'target_name':'stability_threshold','family':'regime','relevance':0.87,'auditability':0.87,'sufficiency':0.87,'pre_deployment_value':0.87},
    ])
    cfg={'selector':{'thresholds':{'relevance':0.5,'auditability':0.6,'sufficiency':0.5,'pre_deployment_value':0.6},'composite':'geometric_mean','dominance_margin':0.1,'near_tie_margin':0.05}}
    assert nominate_target(df,cfg)['nominated_target']=='stability_threshold'
