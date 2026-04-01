import torch

def step_rmse(pred, target):
    return torch.sqrt(((pred[:, :1] - target[:, :1]) ** 2).mean()).item()

def ade_fde(pred, target):
    err = torch.norm(pred[..., :2] - target[..., :2], dim=-1)
    ade = err.mean().item()
    fde = err[:, -1].mean().item()
    return ade, fde

def rollout_horizon_at_threshold(pred, target, threshold=0.5):
    err = torch.norm(pred[..., :2] - target[..., :2], dim=-1).mean(dim=(-1, -2))
    idx = (err < threshold).float().sum(dim=-1).mean().item()
    return idx
