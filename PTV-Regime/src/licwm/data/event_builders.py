"""Generic event-proxy builders."""
import torch

def density_burst_events(obs_hist: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    # obs_hist: [T,N,D]
    pos = obs_hist[..., :2]
    d = torch.cdist(pos, pos)
    density = (d < threshold).float().mean(dim=(-1, -2))
    return (density > density.mean()).float().unsqueeze(-1)

def near_collision_events(obs_hist: torch.Tensor, threshold: float = 0.2) -> torch.Tensor:
    pos = obs_hist[..., :2]
    d = torch.cdist(pos, pos)
    min_d = d.masked_fill(torch.eye(d.size(-1), device=d.device).bool(), 999).amin(dim=(-1, -2))
    return (min_d < threshold).float().unsqueeze(-1)
