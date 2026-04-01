from __future__ import annotations

import torch

from .utils import stopgrad


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax projection onto the probability simplex."""
    z = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    z_cumsum = z_sorted.cumsum(dim) - 1.0
    ks = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    view = [1] * z.ndim
    view[dim] = -1
    ks = ks.view(*view)
    support = z_sorted > (z_cumsum / ks)
    k = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = z_cumsum.gather(dim, k - 1) / k.to(z.dtype)
    return torch.clamp(z - tau, min=0.0)


def topk_binary_mask(scores: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    if k <= 0:
        return torch.zeros_like(scores)
    k = min(int(k), scores.shape[dim])
    topi = torch.topk(scores, k=k, dim=dim).indices
    mask = torch.zeros_like(scores)
    mask.scatter_(dim, topi, 1.0)
    return mask


def budgeted_allocation(
    logits: torch.Tensor,
    budget: float,
    *,
    temperature: float = 1.0,
    kind: str = "sparsemax",
    lambda_dual: float | torch.Tensor | None = None,
    stop_grad: bool = False,
) -> torch.Tensor:
    x = logits if not stop_grad else stopgrad(logits)
    x = x / max(float(temperature), 1e-6)
    if lambda_dual is not None:
        x = x - float(lambda_dual)

    if kind == "sigmoid":
        mass = torch.sigmoid(x)
        w = mass / (mass.sum(dim=-1, keepdim=True) + 1e-8)
    else:
        w = sparsemax(x, dim=-1)
        denom = w.sum(dim=-1, keepdim=True)
        w = torch.where(denom > 0, w / (denom + 1e-8), torch.full_like(w, 1.0 / w.shape[-1]))
    return float(budget) * w


def softmax_truncated(scores: torch.Tensor, B: int, temperature: float = 1.0, stop_grad: bool = True):
    scores = stopgrad(scores) if stop_grad else scores
    scores = scores / max(float(temperature), 1e-6)
    w = torch.softmax(scores, dim=-1)
    if B is None or B <= 0 or B >= w.shape[-1]:
        return w
    topv, topi = torch.topk(w, k=B, dim=-1)
    mask = torch.zeros_like(w)
    mask.scatter_(-1, topi, 1.0)
    w = w * mask
    return w / (w.sum(dim=-1, keepdim=True) + 1e-8)
