import torch

def tv_law(law: torch.Tensor):
    return (law[:, 1:] - law[:, :-1]).abs().mean().item()

def hf_law(law: torch.Tensor):
    spec = torch.fft.rfft(law, dim=1)
    power = spec.real.square() + spec.imag.square()
    mid = power.shape[1] // 2
    return (power[:, mid:].sum() / (power.sum() + 1e-8)).item()
