import torch

def control_metrics(traj):
    pos = traj[..., :2]
    inter = torch.cdist(pos[:, -1], pos[:, -1])
    violation = (inter < 0.1).float().mean().item()
    formation = inter.std().item()
    energy = traj[..., 2:4].pow(2).mean().item() if traj.shape[-1] >= 4 else 0.0
    return {"success_rate": float(violation < 0.2), "safety_violation": violation, "formation_error": formation, "connectivity_retention": float((inter < 1.0).float().mean().item()), "energy": energy}
