from __future__ import annotations

import numpy as np


def test_crafter_adapter_handles_legacy_four_tuple(monkeypatch):
    from ralagwm.envs import crafter_adapter as mod

    class FakeActionSpace:
        n = 7
        def sample(self):
            return 3

    class FakeLegacyCrafterEnv:
        def __init__(self):
            self.action_space = FakeActionSpace()
            self.reset_calls = 0
            self.steps = 0
            self.unwrapped = self

        def reset(self, seed=None):
            self.reset_calls += 1
            self.steps = 0
            obs = np.zeros((64, 64, 3), dtype=np.uint8)
            return obs

        def step(self, action):
            self.steps += 1
            obs = np.full((64, 64, 3), fill_value=action, dtype=np.uint8)
            reward = 1.0
            done = False
            info = {"steps": self.steps}
            return obs, reward, done, info

        def close(self):
            pass

    class FakeGymModule:
        def __init__(self, env):
            self._env = env
        def make(self, env_id, **kwargs):
            return self._env

    fake_env = FakeLegacyCrafterEnv()
    fake_gym = FakeGymModule(fake_env)

    def fake_require_dependency(name, msg):
        if name == "crafter":
            return object()
        if name == "gym":
            return fake_gym
        raise AssertionError(name)

    monkeypatch.setattr(mod, "require_dependency", fake_require_dependency)
    adapter = mod.CrafterEnvAdapter(max_episode_steps=2)

    obs, info = adapter.reset(seed=0)
    assert obs.shape == (3, 64, 64)
    assert info == {}

    out1 = adapter.step(1)
    assert out1.terminated is False
    assert out1.truncated is False

    out2 = adapter.step(2)
    assert out2.terminated is False
    assert out2.truncated is True
    assert out2.reward == 1.0
