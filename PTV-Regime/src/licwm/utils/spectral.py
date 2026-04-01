import torch

def hf_ratio(x: torch.Tensor) -> torch.Tensor:
    spec = torch.fft.rfft(x, dim=1)
    power = spec.real.square() + spec.imag.square()
    split = power.shape[1] // 2
    return (power[:, split:].sum(dim=1) / (power.sum(dim=1) + 1e-8)).mean()
