REQUIRED = {
 'mp_screening_pool.csv':['material_id','formula_pretty','structure_class','split','round_id'],
 'mp_reference_labels.csv':['material_id','E_hull_ref_mev_atom','d_ref_mev_atom','reference_stable'],
 'surrogate_pool.csv':['surrogate_id','selection_rule','target_preserved','backbone','objective','seed','force_mae_ev_a','energy_mae','in_nominated_target','deployable','hindsight_only'],
 'mlip_predictions.csv':['round_id','material_id','surrogate_id','d_hat_mev_atom','acquisition_score'],
 'selector_profiles.csv':['target_name','family','relevance','auditability','sufficiency','pre_deployment_value','composite_score','admissible','nominated','oracle_family_match'],
}
SELECTION_RULES={'lowest_mae','decision_aware','learned_meta_selector','ptv_nominated','empirically_best','target_oracle'}
TARGETS={'average_ef','tail_force','rank_margin','stability_threshold'}
FAMILIES={'baseline','criticality','boundary','regime'}
