import torch

def emb_constraint(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)

def score_consistency(scores_pred: torch.Tensor, scores_ref: torch.Tensor) -> torch.Tensor:
    return torch.mean((scores_pred - scores_ref) ** 2)

def weak_reg(z: torch.Tensor) -> torch.Tensor:
    return torch.mean(z ** 2)

def soft_jaccard(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    inter = (p * q).sum(dim=-1)
    union = p.sum(dim=-1) + q.sum(dim=-1) - inter
    return (inter / (union + eps)).mean()
