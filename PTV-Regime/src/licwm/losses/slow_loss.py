import torch

def slow_loss(c_next: torch.Tensor, c_prev: torch.Tensor) -> torch.Tensor:
    return (c_next[:, 1:] - c_prev[:, :-1]).pow(2).mean()
