"""Constrained basis families for local response prototypes.

All functions return tensors with explicit semantic-channel and prototype axes:
    [B, N, N, C, P]
where C is the number of semantic channels and P is the number of prototypes
per channel.
"""
from __future__ import annotations
import torch


def radial_basis(dist: torch.Tensor, centers: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
    """Compact radial responses.

    Args:
        dist: [B,N,N]
        centers: [C,P]
        widths: [C,P]
    Returns:
        [B,N,N,C,P]
    """
    dist_e = dist[..., None, None]
    centers_e = centers[None, None, None, :, :]
    widths_e = widths.abs()[None, None, None, :, :] + 1e-4
    return torch.exp(-((dist_e - centers_e) ** 2) / widths_e)



def threshold_basis(dist: torch.Tensor, thresholds: torch.Tensor, sharpness: float = 10.0) -> torch.Tensor:
    """Smooth threshold responses with one gate per prototype."""
    dist_e = dist[..., None, None]
    thr_e = thresholds[None, None, None, :, :]
    return torch.sigmoid(sharpness * (thr_e - dist_e))



def directional_basis(phi_ij: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Directional projection responses.

    Args:
        phi_ij: [B,N,N,2]
        directions: [C,P,2]
    Returns:
        [B,N,N,C,P]
    """
    unit = phi_ij / (phi_ij.norm(dim=-1, keepdim=True) + 1e-6)
    return torch.einsum("bijd,cpd->bijcp", unit, directions)
