import torch

def build_interaction_events(obs_hist: torch.Tensor):
    speed = torch.norm(torch.diff(obs_hist[..., :2], dim=0, prepend=obs_hist[:1, ..., :2]), dim=-1).mean(dim=-1)
    braking_wave = (speed < speed.median()).float().unsqueeze(-1)
    return braking_wave
