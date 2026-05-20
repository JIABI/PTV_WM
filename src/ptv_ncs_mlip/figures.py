from pathlib import Path
import matplotlib.pyplot as plt

def make_fig2(round_metrics, surrogate_pool, selector_profiles, outroot):
    out=Path(outroot); (out/'source_data').mkdir(parents=True,exist_ok=True); (out/'figures').mkdir(parents=True,exist_ok=True)
    round_metrics.to_csv(out/'source_data/source_data_fig2a.csv',index=False)
    b=surrogate_pool[['surrogate_id','selection_rule','force_mae_ev_a']].copy(); b['false_stable_rate']=0.0; b['label']=b['selection_rule']; b['is_lowest_mae']=b.selection_rule.eq('lowest_mae'); b['is_best_deployment']=b.selection_rule.eq('empirically_best'); b['is_ptv_target']=b.selection_rule.eq('ptv_nominated')
    b.to_csv(out/'source_data/source_data_fig2b.csv',index=False)
    selector_profiles.to_csv(out/'source_data/source_data_fig2c.csv',index=False)
    round_metrics.to_csv(out/'source_data/source_data_fig2d.csv',index=False)
    plt.figure(); plt.scatter(b['force_mae_ev_a'].fillna(0), b['false_stable_rate']); plt.savefig(out/'figures/fig2_mlip_hero.png'); plt.savefig(out/'figures/fig2_mlip_hero.pdf')
