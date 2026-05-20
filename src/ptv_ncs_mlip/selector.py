def admissible(row, thresholds):
    return row.relevance>=thresholds['relevance'] and row.auditability>=thresholds['auditability'] and row.sufficiency>=thresholds['sufficiency'] and row.pre_deployment_value>=thresholds['pre_deployment_value']

def composite_score(row, mode='geometric_mean'):
    if mode!='geometric_mean': raise ValueError('unsupported')
    return (row.relevance*row.auditability*row.sufficiency*row.pre_deployment_value)**0.25

def nominate_target(selector_profiles, config):
    t=config['selector']['thresholds']
    df=selector_profiles.copy()
    df['admissible']=df.apply(lambda r: admissible(r,t),axis=1)
    df['composite_score']=df.apply(lambda r: composite_score(r,config['selector']['composite']),axis=1)
    elig=df[df.admissible].sort_values('composite_score',ascending=False)
    if elig.empty: raise ValueError('no admissible targets')
    top=elig.iloc[0]
    return {'nominated_target':top.target_name,'family':top.family,'composite_score':float(top.composite_score),'thresholds':t,'dominance_margin':config['selector']['dominance_margin'],'near_tie_margin':config['selector']['near_tie_margin'],'zero_leakage':True,'used_deployment_outcomes':False}
