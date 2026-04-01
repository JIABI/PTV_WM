import torch


def rollout_loss(pred: torch.Tensor, target: torch.Tensor, horizon_weights: torch.Tensor | None = None,
                 event_hist: torch.Tensor | None = None, event_weight: float = 0.0):
    h = min(pred.shape[1], target.shape[1])
    # per-sample, per-horizon error [B,H]
    err = (pred[:, :h] - target[:, :h]).pow(2).mean(dim=(-1, -2))
    w = torch.ones(h, device=pred.device) if horizon_weights is None else horizon_weights[:h]
    weighted = err * w.unsqueeze(0)
    if event_hist is not None and event_weight > 0:
        event_boost = 1 + event_weight * event_hist[:, :h].abs().sum(dim=-1)
        weighted = weighted * event_boost
    return weighted.mean()
