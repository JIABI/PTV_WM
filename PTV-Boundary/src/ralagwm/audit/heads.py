from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .base import BaseAuditHead


def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() in (1, 3):
        x = x.unsqueeze(0)
    return x


def _is_channel_axis_small(v: int) -> bool:
    return int(v) <= 16


def _is_spatial_axis(v: int) -> bool:
    return int(v) > 16


class _ImageLayout:
    BCHW = "bchw"
    BHWC = "bhwc"
    THWC = "thwc"
    TCHW = "tchw"


def _infer_4d_layout(x: torch.Tensor, expected_channels: int | None = None) -> str:
    shape = tuple(int(v) for v in x.shape)
    if expected_channels is not None:
        if shape[1] == expected_channels and _is_spatial_axis(shape[2]) and _is_spatial_axis(shape[3]):
            return _ImageLayout.BCHW
        if shape[3] == expected_channels and _is_spatial_axis(shape[1]) and _is_spatial_axis(shape[2]):
            return _ImageLayout.BHWC
        if shape[0] * shape[1] == expected_channels and _is_channel_axis_small(shape[1]) and _is_spatial_axis(shape[2]) and _is_spatial_axis(shape[3]):
            return _ImageLayout.TCHW
        if shape[0] * shape[3] == expected_channels and _is_channel_axis_small(shape[3]) and _is_spatial_axis(shape[1]) and _is_spatial_axis(shape[2]):
            return _ImageLayout.THWC
    if _is_channel_axis_small(shape[1]) and _is_spatial_axis(shape[2]) and _is_spatial_axis(shape[3]):
        return _ImageLayout.BCHW
    if _is_channel_axis_small(shape[3]) and _is_spatial_axis(shape[1]) and _is_spatial_axis(shape[2]) and not _is_channel_axis_small(shape[0]):
        return _ImageLayout.BHWC
    if _is_channel_axis_small(shape[0]) and _is_channel_axis_small(shape[1]) and _is_spatial_axis(shape[2]) and _is_spatial_axis(shape[3]):
        return _ImageLayout.TCHW
    if _is_channel_axis_small(shape[0]) and _is_channel_axis_small(shape[3]) and _is_spatial_axis(shape[1]) and _is_spatial_axis(shape[2]):
        return _ImageLayout.THWC
    if _is_channel_axis_small(shape[3]) and _is_spatial_axis(shape[1]) and _is_spatial_axis(shape[2]):
        return _ImageLayout.BHWC
    raise ValueError(f"Could not infer 4D image layout from shape {shape}")


def _ensure_bchw(x: torch.Tensor, expected_channels: int | None = None) -> torch.Tensor:
    x = _ensure_batch(x)
    if x.dim() == 2:
        raise ValueError(f"Expected image-like tensor, got vector tensor {tuple(x.shape)}")
    if x.dim() == 4:
        layout = _infer_4d_layout(x, expected_channels=expected_channels)
        if layout == _ImageLayout.BHWC:
            return x.permute(0, 3, 1, 2).contiguous().float()
        if layout == _ImageLayout.BCHW:
            return x.contiguous().float()
        if layout == _ImageLayout.THWC:
            t, h, w, c = x.shape
            return x.permute(0, 3, 1, 2).contiguous().view(1, t * c, h, w).float()
        if layout == _ImageLayout.TCHW:
            t, c, h, w = x.shape
            return x.contiguous().view(1, t * c, h, w).float()
        raise ValueError(f"Unsupported inferred 4D layout for shape {tuple(x.shape)}")
    if x.dim() == 5:
        if _is_channel_axis_small(x.shape[-1]):
            b, t, h, w, c = x.shape
            x = x.permute(0, 1, 4, 2, 3).contiguous()
            return x.view(b, t * c, h, w).float()
        if _is_channel_axis_small(x.shape[2]):
            b, t, c, h, w = x.shape
            return x.contiguous().view(b, t * c, h, w).float()
        raise ValueError(f"Could not infer 5D image layout from shape {tuple(x.shape)}")
    raise ValueError(f"Expected image tensor with 4 or 5 dims after batching, got {tuple(x.shape)}")


class MLPAuditHead(BaseAuditHead):
    def __init__(self, input_dim: int, num_actions: int, hidden_dims: Sequence[int] = (128, 128)) -> None:
        super().__init__(num_actions=num_actions)
        dims = [int(input_dim), *map(int, hidden_dims), int(num_actions)]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_batch(x)
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)
        return self.net(x.float())


class CNNAuditHead(BaseAuditHead):
    def __init__(self, obs_shape: Sequence[int], num_actions: int, conv_channels: Sequence[int] = (32, 64, 128), hidden_dim: int = 256) -> None:
        super().__init__(num_actions=num_actions)
        c1, c2, c3 = map(int, conv_channels)
        obs_shape = tuple(int(v) for v in obs_shape)
        self.expected_channels = _infer_expected_channels(obs_shape)
        self.conv = nn.Sequential(
            nn.LazyConv2d(c1, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.LazyLinear(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_actions))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(x, expected_channels=self.expected_channels)
        z = self.conv(x)
        return self.head(z)


def _infer_expected_channels(obs_shape: Sequence[int]) -> int | None:
    obs_shape = tuple(int(v) for v in obs_shape)
    if len(obs_shape) == 3:
        if obs_shape[-1] <= 16:
            return int(obs_shape[-1])
        if obs_shape[0] <= 16:
            return int(obs_shape[0])
    if len(obs_shape) == 4:
        if obs_shape[-1] <= 16:
            return int(obs_shape[0] * obs_shape[-1])
        if obs_shape[1] <= 16:
            return int(obs_shape[0] * obs_shape[1])
    return None


def build_audit_head_for_obs(obs_shape: Sequence[int], num_actions: int) -> BaseAuditHead:
    obs_shape = tuple(int(v) for v in obs_shape)
    if len(obs_shape) in (3, 4):
        return CNNAuditHead(obs_shape=obs_shape, num_actions=num_actions)
    if len(obs_shape) == 1:
        input_dim = int(obs_shape[0])
    else:
        input_dim = int(torch.tensor(obs_shape).prod().item())
    return MLPAuditHead(input_dim=input_dim, num_actions=num_actions)
