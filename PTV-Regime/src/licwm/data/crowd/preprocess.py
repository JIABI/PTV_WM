"""Fallback preprocessing for crowd trajectories."""
import os
import numpy as np

def load_or_mock(data_root: str, split: str, n_agents: int = 10, steps: int = 64):
    path = os.path.join(data_root, f"{split}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["traj"]
    t = np.linspace(0, 1, steps)
    traj = np.stack([np.sin(t), np.cos(t)], axis=-1)
    return np.tile(traj[:, None, :], (1, n_agents, 1)).astype(np.float32)
