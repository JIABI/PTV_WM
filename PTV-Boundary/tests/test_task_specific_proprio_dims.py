from types import SimpleNamespace

from ralagwm.evaluation.evaluator import infer_obs_dim
from ralagwm.training.paper_pipeline import _infer_flat_obs_dim


class _FakeEnv:
    def __init__(self, obs_shape):
        self.spec = SimpleNamespace(observation_shape=obs_shape)


def test_training_obs_dim_uses_env_shape_for_task_specific_proprio():
    cfg_env = {'name': 'dmc_proprio', 'obs_type': 'proprio', 'obs_dim': 5}
    env = _FakeEnv((17,))
    assert _infer_flat_obs_dim(cfg_env, env) == 17


def test_evaluator_obs_dim_uses_env_shape_for_task_specific_proprio():
    cfg_env = {'name': 'dmc_proprio', 'obs_type': 'proprio', 'obs_dim': 5}
    env = _FakeEnv((17,))
    assert infer_obs_dim(cfg_env, env) == 17
