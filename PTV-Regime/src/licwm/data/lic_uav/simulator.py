import numpy as np
from .scenarios import SCENARIOS

class UAVSimulator:
    def __init__(self, n_agents: int, scenario: str, seed: int = 0):
        assert scenario in SCENARIOS
        self.rng = np.random.default_rng(seed)
        self.n_agents = n_agents
        self.scenario = scenario

    def rollout(self, horizon: int):
        pos = self.rng.normal(size=(self.n_agents, 2)).astype(np.float32)
        vel = np.zeros_like(pos)
        events = np.zeros((horizon, 4), dtype=np.float32)
        out = []
        for t in range(horizon):
            target = np.array([np.sin(t / 8), np.cos(t / 8)], dtype=np.float32)
            if self.scenario == "formation":
                ctrl = (pos.mean(0, keepdims=True) - pos)
            elif self.scenario == "corridor":
                ctrl = np.stack([1.0 - pos[:, 0], -0.2 * pos[:, 1]], axis=-1)
            elif self.scenario == "escort":
                ctrl = target - pos
            else:
                ctrl = (target - pos) * (1 if t < horizon // 2 else -1)
                if t == horizon // 2:
                    events[t, 0] = 1
            vel = 0.85 * vel + 0.15 * ctrl
            pos = pos + 0.1 * vel
            out.append(np.concatenate([pos, vel], axis=-1))
        return np.stack(out), events
