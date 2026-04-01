import torch

def safe_logdet(x: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(x.shape[-1], device=x.device)
    return torch.logdet(x + 1e-6 * eye)
