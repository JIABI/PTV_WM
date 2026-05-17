def topk_for_surrogate(predictions, surrogate_id, round_id, k):
    df=predictions[(predictions.surrogate_id==surrogate_id)&(predictions.round_id==round_id)].sort_values('d_hat_mev_atom',ascending=True)
    return df.head(k)
