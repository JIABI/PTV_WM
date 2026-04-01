import numpy as np
from collections import deque
from .common import EnvWrapper

class ProcgenEnv(EnvWrapper):
    def __init__(self, env_name: str, num_levels: int, start_level: int, episode_len: int, frame_stack: int = 1):
        try:
            from procgen import ProcgenEnv as _Procgen  # type: ignore
        except Exception as e:
            raise ImportError("Install procgen: pip install procgen") from e
        self.env = _Procgen(num_envs=1, env_name=env_name, num_levels=num_levels, start_level=start_level)
        self.episode_len = episode_len
        self.t = 0
        self._action_dim = self.env.action_space.n
        self.frame_stack = frame_stack
        self._obs_shape = (3 * frame_stack, 64, 64)
        self.frames = deque(maxlen=frame_stack)
        # If the underlying procgen env auto-resets (gym3 "first" semantics), the
        # observation returned by step() may already be the first frame of the next
        # episode. We stash it so that a subsequent reset() can return it without
        # calling env.reset() again (avoids double-reset shortening episodes).
        self._pending_obs = None

    @property
    def obs_shape(self):
        return self._obs_shape
        
    @property
    def action_dim(self):
        return self._action_dim
        
    @property
    def is_discrete(self):
        return True

    def _process_obs(self, obs):
        obs = obs.transpose(2, 0, 1)
        return obs.astype(np.float32) / 255.0

    def reset(self, seed=None):
        self.t = 0
        if self._pending_obs is not None:
            obs = self._pending_obs
            self._pending_obs = None
        else:
            obs = self.env.reset()["rgb"][0]
        obs = self._process_obs(obs)
        for _ in range(self.frame_stack):
            self.frames.append(obs)
        return np.concatenate(list(self.frames), axis=0)

    def step(self, action):
        self.t += 1
        # Procgen (gym3) expects a numpy array of shape (num_envs,), not a Python list.
        # Passing a list triggers downstream failures (list has no .astype).
        ac = np.asarray([int(action)], dtype=np.int32)
        out = self.env.step(ac)
        info = {}
        if isinstance(out, tuple) and len(out) == 4:
            obs, r, done_vec, info = out
        elif isinstance(out, tuple) and len(out) == 3:
            # gym3 semantics: (obs, reward, first)
            obs, r, first = out
            done_vec = first
            info = {"first": first}
        else:
            raise RuntimeError(f"Unexpected procgen step() return: type={type(out)}")

        obs_raw = obs["rgb"][0]
        obs_proc = self._process_obs(obs_raw)
        self.frames.append(obs_proc)
        done = bool(done_vec[0] or (self.t >= self.episode_len))
        # If procgen auto-reset occurred, the current obs is already the first frame
        # of the next episode. Stash it so reset() can return it without calling env.reset().
        if bool(done_vec[0]):
            # store raw rgb (uint8) so reset() can process consistently
            self._pending_obs = obs_raw.copy()
        return np.concatenate(list(self.frames), axis=0), float(r[0]), done, info

    # ------------------------
    # State snapshot utilities
    # ------------------------
    def get_state(self):
        """Snapshot for auditable short-horizon oracle rollouts.

        Procgen's C++ env supports get_state()/set_state() (bytes per env).
        We also snapshot the Python-side frame stack and time index.
        """
        if not hasattr(self.env, "get_state"):
            raise RuntimeError("ProcgenEnv.get_state unavailable: procgen version missing get_state().")
        st = self.env.get_state()
        frames = list(self.frames)
        return {"procgen": st, "frames": frames, "t": int(self.t)}

    def set_state(self, state):
        if not hasattr(self.env, "set_state"):
            raise RuntimeError("ProcgenEnv.set_state unavailable: procgen version missing set_state().")
        self.env.set_state(state["procgen"])
        self.frames.clear()
        for f in state.get("frames", []):
            self.frames.append(f)
        self.t = int(state.get("t", 0))
