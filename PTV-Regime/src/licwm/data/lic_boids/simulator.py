"""LIC-Boids simulator with event hooks."""
from __future__ import annotations
import numpy as np

EVENTS = ["threat_onset", "obstacle_burst", "leader_removal", "comm_outage", "goal_switch"]

class LICBoidsSimulator:
    def __init__(self, n_agents: int, mode: str = "homogeneous", seed: int = 0):
        self.n_agents = n_agents
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    def rollout(self, horizon: int):
        pos = self.rng.normal(size=(self.n_agents, 2)).astype(np.float32)
        vel = self.rng.normal(scale=0.1, size=(self.n_agents, 2)).astype(np.float32)
        obs = []
        law = []
        events = np.zeros((horizon, len(EVENTS)), dtype=np.float32)
        for t in range(horizon):
            if t in [horizon//4, horizon//2, 3*horizon//4]:
                e = self.rng.integers(0, len(EVENTS))
                events[t, e] = 1.0
            center = pos.mean(axis=0, keepdims=True)
            align = vel.mean(axis=0, keepdims=True)
            rep = (pos[:, None] - pos[None, :])
            rep = (rep / (np.linalg.norm(rep, axis=-1, keepdims=True) + 1e-3)).sum(axis=1)
            goal = np.array([1.0, 0.0], dtype=np.float32)
            risk = events[t, 0] - events[t, 3]
            if self.mode == "hetero":
                w = self.rng.uniform(0.8, 1.2, size=(self.n_agents, 1)).astype(np.float32)
            else:
                w = 1.0
            acc = 0.3 * (center - pos) + 0.2 * align + 0.1 * rep + 0.1 * goal
            acc = acc * w
            vel = 0.9 * vel + 0.1 * acc
            pos = pos + vel
            obs.append(np.concatenate([pos, vel], axis=-1))
            law.append(np.array([0.5, 0.2, 0.2, 0.1, risk], dtype=np.float32))
        return np.stack(obs), events, np.stack(law)
