from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


@dataclass
class DatasetBundle:
    dataset: Dataset
    loader: DataLoader
    channels: int
    image_size: int


class SyntheticPatternDataset(Dataset):
    def __init__(self, num_samples: int = 256, image_size: int = 32, channels: int = 3) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.channels = channels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        rng = np.random.default_rng(idx)
        h = w = self.image_size
        yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
        freq = 1 + (idx % 5)
        phase = idx * 0.1
        img = np.stack([
            np.sin(freq * np.pi * xx + phase),
            np.cos(freq * np.pi * yy - phase),
            np.sin(freq * np.pi * (xx + yy) + phase),
        ], axis=0)[: self.channels]
        blob = np.exp(-((xx - 0.3 * np.sin(phase)) ** 2 + (yy - 0.3 * np.cos(phase)) ** 2) / 0.1)
        img[0] += blob
        img += 0.03 * rng.standard_normal(img.shape)
        img = img / max(np.max(np.abs(img)), 1e-6)
        img = np.clip(img, -1.0, 1.0)
        return torch.tensor(img, dtype=torch.float32)


class ImageFolderDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int) -> None:
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"ImageFolder root not found: {root}")
        self.paths = sorted([p for p in root.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}])
        if not self.paths:
            raise RuntimeError(f"No images found under {root}")
        from torchvision import transforms
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.tf(img)


class H5FieldDataset(Dataset):
    def __init__(self, path: str | Path, key: str = 'fields') -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"HDF5 dataset not found: {path}")
        self.path = path
        self.key = key
        with h5py.File(path, 'r') as f:
            if key not in f:
                raise KeyError(f"Key '{key}' not found in {path}")
            self.length = len(f[key])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        with h5py.File(self.path, 'r') as f:
            arr = f[self.key][idx]
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        return torch.tensor(arr)


def build_dataset_bundle(cfg: dict[str, Any]) -> DatasetBundle:
    name = cfg['name']
    batch_size = int(cfg.get('batch_size', 32))
    image_size = int(cfg.get('image_size', 32))

    if name == 'synthetic':
        ds = SyntheticPatternDataset(
            num_samples=int(cfg.get('num_samples', 256)),
            image_size=image_size,
            channels=int(cfg.get('channels', 3)),
        )
        channels = int(cfg.get('channels', 3))
    elif name == 'cifar10':
        from torchvision import datasets, transforms
        tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        ds = datasets.CIFAR10(
            root=str(cfg.get('root', 'data/cifar10')),
            train=bool(cfg.get('train', True)),
            transform=tf,
            download=bool(cfg.get('download', True)),
        )
        channels = 3
    elif name == 'imagefolder':
        ds = ImageFolderDataset(cfg['root'], image_size=image_size)
        channels = 3
    elif name == 'pde_h5':
        ds = H5FieldDataset(cfg['root'], key=str(cfg.get('key', 'fields')))
        sample = ds[0]
        channels = int(sample.shape[0])
        image_size = int(sample.shape[-1])
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    num_workers = int(cfg.get('num_workers', 0))
    pin_memory = bool(cfg.get('pin_memory', torch.cuda.is_available()))
    persistent_workers = bool(cfg.get('persistent_workers', False)) and num_workers > 0
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )
    return DatasetBundle(dataset=ds, loader=loader, channels=channels, image_size=image_size)
