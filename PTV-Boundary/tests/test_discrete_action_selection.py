import torch

from ralagwm.baselines import ReconWM
from ralagwm.envs.base import BaseEnvAdapter, EnvSpec, StepOutput
from ralagwm.training.loops import collect_rollout, select_action_from_baseline
from ralagwm.data.replay import ReplayBuffer


class StrictDiscreteEnv(BaseEnvAdapter):
    def __init__(self, obs_dim: int = 8, action_dim: int = 4, horizon: int = 5):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self._step_count = 0
        self._spec = EnvSpec(
            name="strict_discrete",
            obs_type="proprio",
            action_type="discrete",
            observation_shape=(self.obs_dim,),
            action_dim=self.action_dim,
            max_episode_steps=self.horizon,
        )

    def reset(self, seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)
        self._step_count = 0
        return torch.zeros(self.obs_dim, dtype=torch.float32).numpy(), {"seed": seed}

    def step(self, action):
        action = int(action)
        if not (0 <= action < self.action_dim):
            raise IndexError(f"invalid discrete action {action} for action_dim={self.action_dim}")
        self._step_count += 1
        obs = torch.full((self.obs_dim,), float(action), dtype=torch.float32).numpy()
        terminated = self._step_count >= self.horizon
        return StepOutput(
            observation=obs,
            reward=1.0,
            terminated=terminated,
            truncated=False,
            info={"action": action},
        )

    def sample_random_action(self):
        return 1

    def close(self):
        return None


def test_recon_baseline_action_falls_back_to_valid_random_action_when_logits_are_not_action_scores():
    env = StrictDiscreteEnv(obs_dim=8, action_dim=4, horizon=3)
    model = ReconWM(obs_dim=8, action_dim=4, hidden_dim=16, obs_type="proprio")
    obs, _ = env.reset(seed=0)

    action = select_action_from_baseline(model, obs, env, device="cpu")

    assert action == 1
    assert 0 <= action < env.spec.action_dim


def test_collect_rollout_with_recon_baseline_on_strict_discrete_env_does_not_emit_invalid_actions():
    env = StrictDiscreteEnv(obs_dim=8, action_dim=4, horizon=3)
    model = ReconWM(obs_dim=8, action_dim=4, hidden_dim=16, obs_type="proprio")
    replay = ReplayBuffer(capacity=32)

    stats = collect_rollout(
        env,
        replay,
        policy_fn=lambda obs: select_action_from_baseline(model, obs, env, device="cpu"),
        episodes=2,
        max_steps=3,
        seed=0,
    )

    assert stats["episodes"] == 2.0
    assert stats["replay_size"] == 6.0
    assert len(replay) == 6
