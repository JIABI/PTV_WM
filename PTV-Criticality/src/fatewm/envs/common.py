from dataclasses import dataclass
import numpy as np

class EnvWrapper:
    @property
    def obs_shape(self):
        raise NotImplementedError
    @property
    def action_dim(self):
        raise NotImplementedError
    @property
    def is_discrete(self):
        raise NotImplementedError
    @property
    def action_bounds(self):
        return None
    def reset(self, seed=None):
        raise NotImplementedError
    def step(self, action):
        raise NotImplementedError
