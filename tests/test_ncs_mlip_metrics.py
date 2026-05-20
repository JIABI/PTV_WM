import pandas as pd
from ptv_ncs_mlip.metrics import compute_round_metrics

def test_false_stable_metric():
    preds=pd.DataFrame({'round_id':[1,1,1],'material_id':['1','2','3'],'surrogate_id':['s']*3,'d_hat_mev_atom':[-3,-2,-1],'acquisition_score':[1,1,1]})
    ref=pd.DataFrame({'material_id':['1','2','3'],'d_ref_mev_atom':[-10,5,20]})
    surr=pd.DataFrame({'surrogate_id':['s'],'selection_rule':['lowest_mae'],'target_preserved':[False],'force_mae_ev_a':[0.1]})
    m=compute_round_metrics(preds,ref,surr,'s',1,3,25)
    assert abs(m.false_stable_rate-2/3)<1e-9

def test_topk_overlap_not_jaccard():
    preds=pd.DataFrame({'round_id':[1,1,1,1],'material_id':['1','2','3','4'],'surrogate_id':['s']*4,'d_hat_mev_atom':[-3,-2,-1,1],'acquisition_score':[1,1,1,1]})
    ref=pd.DataFrame({'material_id':['1','2','3','4'],'d_ref_mev_atom':[-10,20,-5,30]})
    surr=pd.DataFrame({'surrogate_id':['s'],'selection_rule':['lowest_mae'],'target_preserved':[False],'force_mae_ev_a':[0.1]})
    m=compute_round_metrics(preds,ref,surr,'s',1,2,25)
    assert abs(m.topk_oracle_overlap-0.5)<1e-9

def test_wasted_budget_not_complement():
    preds=pd.DataFrame({'round_id':[1,1,1,1],'material_id':['1','2','3','4'],'surrogate_id':['s']*4,'d_hat_mev_atom':[-3,-2,-1,1],'acquisition_score':[1,1,1,1]})
    ref=pd.DataFrame({'material_id':['1','2','3','4'],'d_ref_mev_atom':[-10,5,-5,20]})
    surr=pd.DataFrame({'surrogate_id':['s'],'selection_rule':['lowest_mae'],'target_preserved':[False],'force_mae_ev_a':[0.1]})
    m=compute_round_metrics(preds,ref,surr,'s',1,2,25)
    assert abs(m.wasted_validation_budget-0.5)<1e-9
