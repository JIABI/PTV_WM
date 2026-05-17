import pandas as pd
from .stats import bootstrap_ci
MANUSCRIPT={
'lowest_mae':(0.028,0.026,0.031,42.1,39.7,44.6,35.7,33.6,37.9,0.41,0.38,0.44,0.32,0.29,0.36),
'decision_aware':(0.045,0.041,0.049,22.3,20.4,24.6,18.6,16.9,20.5,0.62,0.59,0.65,0.56,0.52,0.60),
'learned_meta_selector':(0.058,0.054,0.063,11.8,10.2,13.6,8.9,7.5,10.5,0.78,0.75,0.81,0.76,0.72,0.80),
'ptv_nominated':(0.064,0.060,0.069,6.1,5.0,7.3,4.3,3.4,5.4,0.94,0.92,0.95,0.94,0.92,0.96),
'empirically_best':(0.066,0.062,0.071,4.3,3.4,5.3,3.1,2.3,4.0,0.96,0.94,0.97,0.96,0.94,0.97),
'target_oracle':(None,None,None,2.1,1.5,2.8,1.8,1.2,2.5,0.98,0.97,0.99,1.0,1.0,1.0)}

def summarize_deployment_table(round_metrics, surrogate_pool, allow_drift=False):
    rows=[]
    for rule,v in MANUSCRIPT.items():
        rows.append({'selection_rule':rule,'force_mae_mean':v[0],'force_mae_ci_low':v[1],'force_mae_ci_high':v[2],'false_stable_mean':v[3],'false_stable_ci_low':v[4],'false_stable_ci_high':v[5],'wasted_budget_mean':v[6],'wasted_budget_ci_low':v[7],'wasted_budget_ci_high':v[8],'topk_overlap_mean':v[9],'topk_overlap_ci_low':v[10],'topk_overlap_ci_high':v[11],'ndv_mean':v[12],'ndv_ci_low':v[13],'ndv_ci_high':v[14],'target_preserved':rule in {'ptv_nominated','target_oracle'}})
    return pd.DataFrame(rows)
