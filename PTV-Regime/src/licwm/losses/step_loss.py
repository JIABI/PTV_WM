import torch

def step_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    err = (pred[:, :1] - target[:, :1]).pow(2).mean(dim=-1)
    if mask is not None:
        err = err * mask[:, :1]
    return err.mean()
