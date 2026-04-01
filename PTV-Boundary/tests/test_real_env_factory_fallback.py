from ralagwm.envs import make_env


def test_make_dummy_env_from_factory():
    env = make_env({"name": "dummy", "obs_dim": 8, "action_dim": 4, "obs_type": "proprio", "action_type": "discrete"})
    obs, info = env.reset(seed=0)
    assert obs.shape == (8,)
    step = env.step(env.sample_random_action())
    assert step.observation.shape == (8,)
    env.close()
