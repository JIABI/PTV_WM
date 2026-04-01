import numpy as np
import torch
from tqdm import trange
from omegaconf import OmegaConf

from fatewm.core.utils import ema_update, set_seed
from fatewm.envs import make_env
from fatewm.runners.replay_seq import SeqReplayBuffer
from fatewm.runners.eval_loop import evaluate
from fatewm.core.decision_interface import ShieldState, min_slow_with_shield
from fatewm.core.risk_functional import sample_dmc_candidates, dmc_oracle_action_costs, procgen_oracle_action_costs


def _select_action_train(algo, obs, components, cfg):
    device = components["device"]
    is_discrete = bool(components.get("is_discrete", True))
    beh = str(OmegaConf.select(cfg, "behavior.name", default="random"))
    env_name = str(OmegaConf.select(cfg, "env.name", default=""))
    if beh == "algo":
        return algo.act(obs, eval_mode=False) if hasattr(algo, "act") else 0
    if beh == "random":
        if is_discrete:
            return int(np.random.randint(int(components.get("action_dim", 2))))
        return algo.act(obs, eval_mode=False) if hasattr(algo, "act") else 0
    if beh == "fixed":
        return int(OmegaConf.select(cfg, "behavior.fixed_action", default=0))

    if env_name != "toy" or not is_discrete:
        return algo.act(obs, eval_mode=False) if hasattr(algo, "act") else 0

    shield_enabled = bool(OmegaConf.select(cfg, "interface.shield.enabled", default=True))
    alpha = float(OmegaConf.select(cfg, "interface.shield.alpha", default=0.9))
    fallback = int(OmegaConf.select(cfg, "interface.shield.fallback_action", default=1))
    thr = float(OmegaConf.select(cfg, "env.event_threshold", default=1.0))
    hys_margin = float(OmegaConf.select(cfg, "interface.hysteresis.margin", default=0.0))

    shield_state = components.get("shield_state", None)
    if shield_state is None:
        shield_state = ShieldState.new()
        components["shield_state"] = shield_state

    if shield_enabled:
        return min_slow_with_shield(algo, obs, device=device, event_threshold=thr, alpha=alpha,
                                    fallback_action=fallback, shield_state=shield_state,
                                    hysteresis_margin=hys_margin)
    return min_slow_with_shield(algo, obs, device=device, event_threshold=thr, alpha=1e18,
                                fallback_action=fallback, shield_state=shield_state,
                                hysteresis_margin=hys_margin)


def _dmc_extra_reference(env, algo, obs, prev_action, cfg, deltas):
    K = int(OmegaConf.select(cfg, "method.num_candidates", default=32))
    sigma = float(OmegaConf.select(cfg, "method.candidate_sigma", default=0.3))
    sigma_large = OmegaConf.select(cfg, "method.candidate_sigma_large", default=None)
    sigma_large = None if sigma_large is None else float(sigma_large)
    frac_large = float(OmegaConf.select(cfg, "method.candidate_frac_large", default=0.25))
    fail_pen = float(OmegaConf.select(cfg, "method.oracle_failure_penalty", default=0.0))

    low = torch.tensor(env.action_bounds[0], dtype=torch.float32)
    high = torch.tensor(env.action_bounds[1], dtype=torch.float32)
    prev_t = torch.tensor(prev_action, dtype=torch.float32)
    try:
        a_actor = algo.act(obs, eval_mode=False) if hasattr(algo, "act") else None
        a_actor_t = torch.tensor(a_actor, dtype=torch.float32) if a_actor is not None else None
    except Exception:
        a_actor_t = None

    cands = sample_dmc_candidates(
        prev_t,
        num_candidates=K,
        sigma=sigma,
        sigma_large=sigma_large,
        frac_large=frac_large,
        action_low=low,
        action_high=high,
        actor_action=a_actor_t,
        include_zero=True,
        include_uniform=True,
    )
    costs = []
    for h in deltas:
        costs.append(dmc_oracle_action_costs(env, cands, horizon=int(h), failure_penalty=fail_pen))
    return {
        "candidates": cands.cpu().numpy().astype(np.float32),
        "oracle_costs_by_tau": torch.stack(costs, dim=0).cpu().numpy().astype(np.float32),
    }


def _procgen_extra_reference(env, cfg, deltas):
    A = int(getattr(env, "action_dim", 0))
    fail_pen = float(OmegaConf.select(cfg, "method.oracle_failure_penalty", default=0.0))
    env.oracle_particles = int(OmegaConf.select(cfg, "method.oracle_particles", default=1))
    env.oracle_continuation = str(OmegaConf.select(cfg, "method.oracle_continuation", default="constant"))
    costs = []
    for h in deltas:
        costs.append(procgen_oracle_action_costs(env, A, horizon=int(h), failure_penalty=fail_pen))
    return {
        "candidates": np.arange(A, dtype=np.int64),
        "oracle_costs_by_tau": torch.stack(costs, dim=0).cpu().numpy().astype(np.float32),
    }


def train(cfg, build_components, eval_env_cfg=None):
    set_seed(int(OmegaConf.select(cfg, "seed", default=0)))
    device = torch.device(str(OmegaConf.select(cfg, "device", default="cpu")))

    env = make_env(cfg.env)
    eval_env = make_env(eval_env_cfg) if eval_env_cfg is not None else None

    deltas = list(OmegaConf.select(cfg, "timescales.deltas", default=[1, 2, 4, 8]))
    seq_len = max(deltas) + 1
    rb = SeqReplayBuffer(int(OmegaConf.select(cfg, "replay.capacity", default=100000)), seq_len=seq_len)

    components = build_components(cfg, device)
    algo = components["algo"]
    optimizer = components["optimizer"]
    router = components.get("fate_estimator", None)
    teacher_algo = components.get("teacher_algo", None)

    components["device"] = device
    components["shield_state"] = ShieldState.new()

    obs = env.reset(seed=int(OmegaConf.select(cfg, "seed", default=0)))
    total_steps = int(OmegaConf.select(cfg, "train.total_steps", default=10000))
    prev_action = None
    if str(OmegaConf.select(cfg, "env.name", default="")) == "dmc":
        prev_action = np.zeros((int(getattr(env, "action_dim", 1)),), dtype=np.float32)

    for step in trange(total_steps, desc=f"train[{cfg.env.name}|{cfg.method.name}|{cfg.algo.name}]"):
        action = _select_action_train(algo, obs, components, cfg)
        extra = None
        env_name = str(OmegaConf.select(cfg, "env.name", default=""))
        ref_name = str(OmegaConf.select(cfg, "method.reference", default="teacher"))
        compute_every = int(OmegaConf.select(cfg, "method.oracle_compute_every", default=1))

        if env_name == "dmc" and ref_name in ("dmc_oracle", "oracle") and (step % compute_every == 0):
            try:
                if prev_action is None:
                    prev_action = np.zeros((env.action_dim,), dtype=np.float32)
                extra = _dmc_extra_reference(env, algo, obs, prev_action, cfg, deltas)
            except Exception:
                extra = None

        if env_name == "procgen" and ref_name in ("procgen_oracle", "oracle") and (step % compute_every == 0):
            try:
                extra = _procgen_extra_reference(env, cfg, deltas)
            except Exception:
                extra = None

        next_obs, reward, done, info = env.step(action)
        rb.add(obs, action, reward, next_obs, done, extra=extra)
        obs = next_obs if not done else env.reset(seed=int(OmegaConf.select(cfg, "seed", default=0)) + step + 1)

        if prev_action is not None:
            try:
                prev_action = np.array(action, dtype=np.float32)
            except Exception:
                pass

        if len(rb) >= int(OmegaConf.select(cfg, "replay.warmup_steps", default=1000)):
            for _ in range(int(OmegaConf.select(cfg, "train.update_ratio", default=1))):
                batch = rb.sample(int(OmegaConf.select(cfg, "train.batch_size", default=64)))
                att_gate = components.get("att_gate", None)
                out = algo.update(batch, deltas, cfg.method, router, att_gate=att_gate, env_cfg=cfg.env, teacher_algo=teacher_algo)
                optimizer.zero_grad()
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(algo.parameters(), float(OmegaConf.select(cfg, "train.grad_clip", default=10.0)))
                optimizer.step()

                if cfg.method.name in ("ptv_criticality", "rrrm", "fatewm") and hasattr(router, "dual_update"):
                    mean_sum_w = float(out.logs.get("mean_sum_w", 0.0))
                    B_budget = float(out.logs.get("budget_B", OmegaConf.select(cfg, "method.B", default=8.0)))
                    router.dual_update(mean_sum_w - B_budget)
                if teacher_algo is not None:
                    ema_update(teacher_algo, algo, decay=float(OmegaConf.select(cfg, "method.teacher_ema", default=0.995)))

        interval = int(OmegaConf.select(cfg, "eval.interval", default=1000))
        episodes = int(OmegaConf.select(cfg, "eval.episodes", default=10))
        if (step > 0) and (step % interval == 0):
            stats = evaluate(cfg, components, n_episodes=episodes, env=eval_env)
            print(f"\n[eval@{step}] {stats}\n")

    episodes = int(OmegaConf.select(cfg, "eval.episodes", default=10))
    stats = evaluate(cfg, components, n_episodes=episodes, env=eval_env)
    print(f"\n[final] {stats}\n")
    return stats
