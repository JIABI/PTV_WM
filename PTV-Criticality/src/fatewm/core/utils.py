import copy
import random
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stopgrad(x: torch.Tensor) -> torch.Tensor:
    return x.detach()


def clone_model(module: torch.nn.Module) -> torch.nn.Module:
    cloned = copy.deepcopy(module)
    for p in cloned.parameters():
        p.requires_grad_(False)
    cloned.eval()
    return cloned


def ema_update(target: torch.nn.Module, source: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.mul_(float(decay)).add_(s.data, alpha=1.0 - float(decay))
        for t_buf, s_buf in zip(target.buffers(), source.buffers()):
            t_buf.copy_(s_buf)


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    norms = []
    for p in parameters:
        if p.grad is None:
            continue
        norms.append(torch.norm(p.grad.detach(), p=2))
    if not norms:
        return torch.tensor(0.0)
    return torch.norm(torch.stack(norms), p=2)
