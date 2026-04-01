import torch


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def get_device(name: str) -> torch.device:
    return resolve_device(name)
