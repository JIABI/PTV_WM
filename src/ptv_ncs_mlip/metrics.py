import pandas as pd
from .acquisition import topk_for_surrogate

def compute_round_metrics(predictions, reference_labels, surrogate_pool, surrogate_id, round_id, k, delta, deployment_values=None):
    topk = topk_for_surrogate(predictions, surrogate_id, round_id, k)
    ref = reference_labels[['material_id','d_ref_mev_atom']].copy()
    merged = topk.merge(ref,on='material_id',how='left')
    fsr=(merged.d_ref_mev_atom>0).mean()
    oracle=reference_labels.sort_values('d_ref_mev_atom').head(k)
    overlap=len(set(topk.material_id)&set(oracle.material_id))/k
    reference_rank={m:i+1 for i,m in enumerate(reference_labels.sort_values('d_ref_mev_atom').material_id.tolist())}
    wasted=((merged.d_ref_mev_atom>0)&(merged.material_id.map(reference_rank)>k)).mean()
    dep=1-fsr
    oracle_dep=1-(oracle.d_ref_mev_atom>0).mean()
    ndv=(dep/oracle_dep) if oracle_dep else 0.0
    if deployment_values is not None and surrogate_id in deployment_values:
        ndv=deployment_values[surrogate_id]
    srow=surrogate_pool[surrogate_pool.surrogate_id==surrogate_id].iloc[0]
    return pd.Series({'round_id':round_id,'surrogate_id':surrogate_id,'selection_rule':srow.selection_rule,'target_preserved':srow.target_preserved,'force_mae_ev_a':srow.force_mae_ev_a,'false_stable_rate':fsr,'wasted_validation_budget':wasted,'topk_oracle_overlap':overlap,'normalized_deployment_value':ndv})
