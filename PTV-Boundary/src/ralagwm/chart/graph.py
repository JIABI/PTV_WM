"""Graph construction utilities for local action subgraphs."""
from __future__ import annotations

import torch


def knn_edges(coords: torch.Tensor, k: int = 3) -> torch.Tensor:
    if coords.shape[0] <= 1:
        return torch.zeros((0, 2), dtype=torch.long, device=coords.device)
    dist = torch.cdist(coords, coords)
    idx = dist.topk(k=min(k + 1, coords.shape[0]), largest=False).indices[:, 1:]
    src = torch.arange(coords.shape[0], device=coords.device).unsqueeze(1).expand_as(idx)
    return torch.stack([src.reshape(-1), idx.reshape(-1)], dim=1)


def anchor_edges(num_nodes: int, anchor_index: int = 0, device: torch.device | None = None) -> torch.Tensor:
    if num_nodes <= 1:
        return torch.zeros((0, 2), dtype=torch.long, device=device)
    others = [i for i in range(num_nodes) if i != anchor_index]
    src = torch.full((len(others),), anchor_index, device=device, dtype=torch.long)
    dst = torch.tensor(others, device=device, dtype=torch.long)
    return torch.stack([src, dst], dim=1)


def merge_edges(*edge_lists: torch.Tensor) -> torch.Tensor:
    valid = [e for e in edge_lists if e.numel() > 0]
    if not valid:
        return torch.zeros((0, 2), dtype=torch.long)
    edges = torch.cat(valid, dim=0)
    return torch.unique(edges, dim=0)
