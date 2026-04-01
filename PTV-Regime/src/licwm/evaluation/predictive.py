import torch
from licwm.metrics.predictive import step_rmse, ade_fde, rollout_horizon_at_threshold

def evaluate_predictive(model, loader, cfg):
    model.eval()
    with torch.no_grad():
        all_pred, all_tgt = [], []
        for b in loader:
            out = model(b.obs_hist, b.action_hist, b.event_hist, horizon=cfg.trainer.pred_len, teacher_forcing=False)
            all_pred.append(out.pred_obs)
            all_tgt.append(b.fut_state)
        pred = torch.cat(all_pred)
        tgt = torch.cat(all_tgt)
    ade, fde = ade_fde(pred, tgt)
    return {"step_rmse": step_rmse(pred, tgt), "rollout_ade": ade, "rollout_fde": fde, "rollout_horizon": rollout_horizon_at_threshold(pred, tgt, cfg.evaluator.threshold)}
