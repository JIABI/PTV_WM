import torch

def cvar_tail(risk: torch.Tensor, alpha: float = 0.9):
    flat = risk.flatten()
    k = max(1, int((1-alpha) * flat.numel()))
    topk = torch.topk(flat, k).values
    return topk.mean().item()
