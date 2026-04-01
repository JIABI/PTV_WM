def evaluate_ood(metrics_in: dict, metrics_out: dict):
    return {"scale_transfer_gap": metrics_out["rollout_ade"] - metrics_in["rollout_ade"]}
