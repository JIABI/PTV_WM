import torch

def recovery_latency(metric_t: torch.Tensor, event_t: torch.Tensor, threshold: float):
    idx = torch.where(event_t > 0)[0]
    if len(idx) == 0: return 0.0
    start = idx[0].item()
    post = metric_t[start:]
    hit = torch.where(post < threshold)[0]
    return float(hit[0].item()) if len(hit) else float(len(post))
