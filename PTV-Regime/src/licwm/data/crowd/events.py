import torch
from ..event_builders import density_burst_events, near_collision_events

def build_crowd_events(obs_hist: torch.Tensor):
    return torch.cat([density_burst_events(obs_hist), near_collision_events(obs_hist)], dim=-1)
