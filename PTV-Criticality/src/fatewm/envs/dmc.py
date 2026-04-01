from .common import EnvWrapper

class DMCEnv(EnvWrapper):
    def __init__(self, task: str, from_pixels: bool, frame_skip: int, episode_len: int):
        try:
            from dm_control import suite  # type: ignore
        except Exception as e:
            raise ImportError("Install dm_control to use DMCEnv: pip install dm_control") from e
        self.env = suite.load(domain_name=task.split('_',1)[0], task_name=task.split('_',1)[1])
        self.frame_skip = frame_skip
        self.episode_len = episode_len
        self.t = 0
        self.from_pixels = from_pixels
        self._action_dim = self.env.action_spec().shape[0]
        self._action_bounds = (self.env.action_spec().minimum, self.env.action_spec().maximum)
        ts = self.env.reset()
        self._obs_shape = self._obs(ts).shape

    @property
    def obs_shape(self):
        return self._obs_shape
        
    @property
    def action_dim(self):
        return self._action_dim
        
    @property
    def is_discrete(self):
        return False

    @property
    def action_bounds(self):
        return self._action_bounds

    def reset(self, seed=None):
        self.t = 0
        ts = self.env.reset()
        return self._obs(ts)

    # ---- state snapshot utilities (for auditable oracle rollouts) ----
    def get_state(self):
        """Return a copy of the underlying physics state."""
        return self.env.physics.get_state().copy()

    def set_state(self, state):
        """Set physics state and forward the simulator."""
        self.env.physics.set_state(state)
        self.env.physics.forward()

    def step(self, action):
        self.t += 1
        r = 0.0
        done = False
        info = {}
        for _ in range(self.frame_skip):
            ts = self.env.step(action)
            r += float(ts.reward or 0.0)
            done = bool(ts.last()) or (self.t >= self.episode_len)
            if done:
                break
        return self._obs(ts), r, done, info

    def _obs(self, ts):
        import numpy as np
        obs = ts.observation
        x = []
        for k in sorted(obs.keys()):
            x.append(obs[k].reshape(-1))
        return np.concatenate(x, axis=0).astype('float32')
