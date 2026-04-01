import numpy as np
from .common import EnvWrapper

class AtariEnv(EnvWrapper):
    def __init__(self, game: str, frame_skip: int, sticky_actions: bool, episode_len: int):
        try:
            import gymnasium as gym
            import ale_py
            from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
        except Exception as e:
            raise ImportError("Install gymnasium[atari] and ale-py: pip install gymnasium[atari] ale-py") from e
            
        game_name = "".join([w.capitalize() for w in game.split("_")])
        env_id = f"ALE/{game_name}-v5"
        env = gym.make(env_id, frameskip=1, repeat_action_probability=0.25 if sticky_actions else 0.0)
        env = AtariPreprocessing(env, noop_max=30, frame_skip=frame_skip, screen_size=84, terminal_on_life_loss=False, grayscale_obs=True, grayscale_newaxis=False)
        env = FrameStackObservation(env, stack_size=4)
        self.env = env
        self.episode_len = episode_len
        self.t = 0
        self._action_dim = self.env.action_space.n
        self._obs_shape = self.env.observation_space.shape # (4, 84, 84)

    @property
    def obs_shape(self):
        return self._obs_shape
        
    @property
    def action_dim(self):
        return self._action_dim
        
    @property
    def is_discrete(self):
        return True

    def reset(self, seed=None):
        self.t = 0
        obs, _ = self.env.reset(seed=seed)
        return np.array(obs, dtype=np.float32) / 255.0

    def step(self, action):
        self.t += 1
        obs, r, terminated, truncated, info = self.env.step(int(action))
        done = bool(terminated or truncated or (self.t >= self.episode_len))
        return np.array(obs, dtype=np.float32) / 255.0, float(r), done, info
