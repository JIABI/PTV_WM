import numpy as np
from .common import EnvWrapper

class ToyEnv(EnvWrapper):
    def __init__(self, obs_dim=8, action_dim=3, episode_len=200,
                 noise_std=0.05, slow_decay=0.995, fast_decay=0.6, event_threshold=1.5, name="toy"):
        assert obs_dim % 2 == 0
        self.obs_dim = obs_dim
        self._action_dim = action_dim
        self.episode_len = episode_len
        self.noise_std = noise_std
        self.slow_decay = slow_decay
        self.fast_decay = fast_decay
        self.event_threshold = event_threshold
        self.t = 0
        self.state = None

    @property
    def obs_shape(self):
        return (self.obs_dim,)
        
    @property
    def action_dim(self):
        return self._action_dim
        
    @property
    def is_discrete(self):
        return True

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.t = 0
        self.state = np.random.randn(self.obs_dim).astype(np.float32) * 0.1
        return self.state.copy()

    def step(self, action):
        self.t += 1
        a = int(action)
        x_fast = self.state[: self.obs_dim//2]
        x_slow = self.state[self.obs_dim//2 :]

        a_eff_fast = (a - 1) * 0.15
        a_eff_slow = (1 - abs(a - 1)) * 0.05

        noise = np.random.randn(self.obs_dim).astype(np.float32) * self.noise_std
        x_fast = self.fast_decay * x_fast + a_eff_fast + noise[: self.obs_dim//2]
        coupling = np.mean(np.abs(x_fast))
        x_slow = self.slow_decay * x_slow + 0.02 * coupling + a_eff_slow + noise[self.obs_dim//2 :]
        self.state = np.concatenate([x_fast, x_slow]).astype(np.float32)

        slow_norm = float(np.linalg.norm(x_slow))
        reward = 1.0 - 0.3 * slow_norm
        done = self.t >= self.episode_len
        info = {"failure": slow_norm > self.event_threshold}
        if info["failure"]:
            reward -= 5.0
        return self.state.copy(), float(reward), bool(done), info
