import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Set Python/NumPy/PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
