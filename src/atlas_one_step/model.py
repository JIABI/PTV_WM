from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / max(half - 1, 1))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return self.proj(emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0, groups: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, base_channels: int = 64, time_dim: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.time = TimeEmbedding(time_dim)
        self.in_conv = nn.Conv2d(in_ch, base_channels, 3, padding=1)
        self.block1 = ResBlock(base_channels, base_channels, time_dim, dropout=dropout)
        self.block2 = ResBlock(base_channels, base_channels, time_dim, dropout=dropout)
        self.block3 = ResBlock(base_channels, base_channels, time_dim, dropout=dropout)
        self.out_norm = nn.GroupNorm(min(8, base_channels), base_channels)
        self.out = nn.Conv2d(base_channels, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time(t)
        h = self.in_conv(x)
        h = self.block1(h, t_emb)
        h = self.block2(h, t_emb)
        h = self.block3(h, t_emb)
        return self.out(F.silu(self.out_norm(h)))


class GNResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0, groups: int = 8) -> None:
        super().__init__()
        g1 = min(groups, in_ch)
        g2 = min(groups, out_ch)
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, heads: int = 4, groups: int = 8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(groups, channels), channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.norm(x).reshape(b, c, h * w).permute(0, 2, 1)
        y, _ = self.attn(y, y, y, need_weights=False)
        y = y.permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(y)


class PaperUNet(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        base_channels: int = 96,
        time_dim: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 2),
        num_res_blocks: int = 2,
        use_attention: bool = True,
        attention_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.time = TimeEmbedding(time_dim)
        self.in_conv = nn.Conv2d(in_ch, base_channels, 3, padding=1)

        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels
        self.skip_channels: list[int] = []
        for i, mult in enumerate(channel_mults):
            outc = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(GNResBlock(ch, outc, time_dim, dropout=dropout))
                ch = outc
            self.downs.append(blocks)
            self.skip_channels.append(ch)
            self.downsamples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1) if i < len(channel_mults) - 1 else nn.Identity())

        self.mid1 = GNResBlock(ch, ch, time_dim, dropout=dropout)
        self.mid_attn = SelfAttention2d(ch, heads=attention_heads) if use_attention else nn.Identity()
        self.mid2 = GNResBlock(ch, ch, time_dim, dropout=dropout)

        self.ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            outc = base_channels * mult
            blocks = nn.ModuleList()
            blocks.append(GNResBlock(ch + self.skip_channels[i], outc, time_dim, dropout=dropout))
            ch = outc
            for _ in range(num_res_blocks - 1):
                blocks.append(GNResBlock(ch, outc, time_dim, dropout=dropout))
                ch = outc
            self.ups.append(blocks)
            self.upsamples.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1) if i > 0 else nn.Identity())

        self.out_norm = nn.GroupNorm(min(8, ch), ch)
        self.out = nn.Conv2d(ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time(t)
        h = self.in_conv(x)
        skips: list[torch.Tensor] = []

        for blocks, down in zip(self.downs, self.downsamples):
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)
            h = down(h)

        h = self.mid2(self.mid_attn(self.mid1(h, t_emb)), t_emb)

        for blocks, up in zip(self.ups, self.upsamples):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode='nearest')
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, t_emb)
            h = up(h)

        return self.out(F.silu(self.out_norm(h)))


class IdentityPhi(nn.Module):
    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return z

    def regularization_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=next(iter(self.parameters()), torch.tensor(0.0)).device)


class AffinePhi(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.scale * z + self.bias

    def regularization_loss(self) -> torch.Tensor:
        return 1e-4 * (self.scale.pow(2).mean() + self.bias.pow(2).mean())


class ShallowPhi(nn.Module):
    def __init__(self, channels: int, hidden_channels: int = 16, spectral_bound: float = 1.0) -> None:
        super().__init__()
        self.spectral_bound = float(spectral_bound)
        self.conv1 = nn.Conv2d(channels, hidden_channels, 1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(hidden_channels, channels, 1)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return z + 0.1 * self.conv2(self.act(self.conv1(z)))

    def regularization_loss(self) -> torch.Tensor:
        l2 = torch.tensor(0.0, device=next(self.parameters()).device)
        spectral_pen = torch.tensor(0.0, device=l2.device)
        for p in self.parameters():
            l2 = l2 + p.pow(2).mean()
        for layer in (self.conv1, self.conv2):
            w = layer.weight.reshape(layer.weight.shape[0], -1)
            sigma_max = torch.linalg.matrix_norm(w, ord=2)
            spectral_pen = spectral_pen + torch.relu(sigma_max - self.spectral_bound).pow(2)
        return 1e-4 * l2 + 1e-3 * spectral_pen


def build_model(cfg: dict, channels: int) -> nn.Module:
    name = cfg.get('name', 'tiny_unet')
    if name == 'tiny_unet':
        return TinyUNet(channels, channels, base_channels=int(cfg.get('base_channels', 64)), time_dim=int(cfg.get('time_dim', 64)), dropout=float(cfg.get('dropout', 0.0)))
    if name == 'paper_unet':
        return PaperUNet(
            channels,
            channels,
            base_channels=int(cfg.get('base_channels', 96)),
            time_dim=int(cfg.get('time_dim', 128)),
            channel_mults=tuple(int(v) for v in cfg.get('channel_mults', [1, 2, 2])),
            num_res_blocks=int(cfg.get('num_res_blocks', 2)),
            use_attention=bool(cfg.get('use_attention', True)),
            attention_heads=int(cfg.get('attention_heads', 4)),
            dropout=float(cfg.get('dropout', 0.0)),
        )
    raise ValueError(f'Unsupported model: {name}')


def build_phi_map(cfg: dict, channels: int) -> nn.Module:
    typ = cfg.get('type', 'identity')
    if typ == 'identity':
        return IdentityPhi()
    if typ == 'affine':
        return AffinePhi(channels)
    if typ == 'shallow':
        return ShallowPhi(channels, hidden_channels=int(cfg.get('hidden_channels', 16)), spectral_bound=float(cfg.get('spectral_bound', 1.0)))
    raise ValueError(f'Unsupported phi_map type: {typ}')
