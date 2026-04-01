import torch

def geometry_distance(obs_a, obs_b):
    return torch.norm(obs_a[..., :2] - obs_b[..., :2], dim=-1).mean().item()

def law_separation(law_a, law_b):
    return torch.norm(law_a - law_b, dim=-1).mean().item()

def response_separation(resp_a, resp_b):
    return torch.norm(resp_a - resp_b, dim=-1).mean().item()
