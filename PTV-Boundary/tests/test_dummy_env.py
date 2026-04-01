from ralagwm.envs.dummy_env import DummyEnv

def test_dummy_env_step():
    env = DummyEnv()
    env.reset()
    obs, reward, done, info = env.step(0)
    assert obs.shape[0] == env.obs_dim
    assert isinstance(done, bool)
