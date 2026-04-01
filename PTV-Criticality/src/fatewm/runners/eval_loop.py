import numpy as np
import torch
from omegaconf import OmegaConf

from fatewm.envs import make_env
from fatewm.core.metrics import (
    episode_return,
    drift_metric,
    ping_pong_rate,
    switch_rate,
    tail_mean,
)
from fatewm.core.decision_interface import (
    ShieldState,
    min_slow_decision,
    predict_next_slow_norm,
)
from fatewm.core.risk_functional import (
    sample_dmc_candidates,
    refine_action_by_energy_gd,
    refine_action_by_energy_mirror,
)


def _infer_num_actions(env, is_discrete: bool, default_n: int = 5) -> int:
    """Best-effort infer discrete action count for histograms."""
    if not is_discrete:
        return 0
    # Prefer wrapper-provided action_dim if available (covers Procgen/Atari wrappers).
    n2 = getattr(env, "action_dim", None)
    if isinstance(n2, int) and n2 > 0:
        return int(n2)
    n = getattr(getattr(env, "action_space", None), "n", None)
    if isinstance(n, int) and n > 0:
        return n
    return int(default_n)


def evaluate(cfg, components, n_episodes: int = 10, env=None):
    """Evaluate a policy under a fixed environment.

    Supported eval policies:
      - eval.policy=fixed: always take eval.fixed_action
      - eval.policy=interface: score-based interface on toy env (min predicted slow norm) + optional shield
      - eval.policy=slow_only: score-based interface on toy env (min predicted slow norm), no shield
    """
    if env is None:
        env = make_env(cfg.env)

    algo = components["algo"]
    device = components.get(
        "device", torch.device(str(OmegaConf.select(cfg, "device", default="cpu")))
    )
    is_discrete = bool(components.get("is_discrete", True))
    env_name = str(OmegaConf.select(cfg, "env.name", default=""))

    eval_policy = str(OmegaConf.select(cfg, "eval.policy", default="interface"))
    eval_fixed_action = int(OmegaConf.select(cfg, "eval.fixed_action", default=0))

    shield_enabled = bool(OmegaConf.select(cfg, "interface.shield.enabled", default=True))
    alpha = float(OmegaConf.select(cfg, "interface.shield.alpha", default=0.9))
    fallback = int(OmegaConf.select(cfg, "interface.shield.fallback_action", default=1))
    thr = float(OmegaConf.select(cfg, "env.event_threshold", default=1.0))
    hys_margin = float(OmegaConf.select(cfg, "interface.hysteresis.margin", default=0.0))

    returns, drifts, tails, abas, switches = [], [], [], [], []
    ep_lens = []
    failures_ep = 0

    # diagnostics accumulators (across all episodes)
    n_actions = _infer_num_actions(env, is_discrete=is_discrete, default_n=5)
    action_hist_global = np.zeros((n_actions,), dtype=np.int64) if n_actions > 0 else None
    total_steps = 0
    total_shield_trigger = 0
    total_fallback = 0
    total_hys_decisions = 0
    total_hys_holds = 0
    total_refine_triggers = 0
    total_margin_sum = 0.0
    total_margin_count = 0

    for ep in range(int(n_episodes)):
        obs = env.reset(seed=int(OmegaConf.select(cfg, "seed", default=0)) + 10000 + ep)
        done = False
        ep_rew = []
        ep_failed = False
        ep_states = []
        ep_actions = []
        ep_actions_cont = []
        shield_state = ShieldState.new()

        # DMC continuous-control interface state
        prev_a_cont = None
        prev_e_cont = None

        # Discrete energy-interface state (Procgen/Atari)
        prev_a_disc = None

        while not done:
            # ----------------
            # Action selection
            # ----------------
            if eval_policy == "fixed":
                action = int(eval_fixed_action)

            elif env_name == "toy" and is_discrete:
                # Pure rule: minimise predicted next-step slow norm.
                a_star = int(
                    min_slow_decision(
                        algo,
                        obs,
                        device=device,
                        hysteresis_margin=hys_margin,
                        hysteresis_state=shield_state,
                    )
                )

                if eval_policy == "slow_only":
                    action = a_star
                else:
                    # Interface policy: same rule + optional hard shield.
                    slow_hat = float(predict_next_slow_norm(algo, obs, a_star, device=device))
                    risk = float(slow_hat / (thr + 1e-6))
                    shield_state.update(risk)
                    triggered = bool(shield_enabled and (risk > alpha))
                    total_shield_trigger += int(triggered)
                    if triggered:
                        action = int(fallback)
                        total_fallback += 1
                    else:
                        action = a_star

            elif env_name == "dmc" and (not is_discrete) and eval_policy in ("interface", "slow_only"):
                # Candidate-based energy interface for continuous control.
                # Energy = -predicted reward for next-step latent.
                K = int(OmegaConf.select(cfg, "method.num_candidates", default=32))
                sigma = float(OmegaConf.select(cfg, "method.candidate_sigma", default=0.3))
                sigma_large = OmegaConf.select(cfg, "method.candidate_sigma_large", default=None)
                if sigma_large is not None:
                    sigma_large = float(sigma_large)
                frac_large = float(OmegaConf.select(cfg, "method.candidate_frac_large", default=0.25))

                # Adaptive refinement (Scheme-1): if margin is small, locally optimize
                # top candidate in continuous action space to increase decision margin.
                refine_enabled = bool(OmegaConf.select(cfg, "interface.refine.enabled", default=True))
                refine_margin_trigger = float(OmegaConf.select(cfg, "interface.refine.margin_trigger", default=0.02))
                refine_method = str(OmegaConf.select(cfg, "interface.refine.method", default="mirror"))
                refine_steps = int(OmegaConf.select(cfg, "interface.refine.steps", default=2))
                # GD-specific params
                refine_step_size = float(OmegaConf.select(cfg, "interface.refine.step_size", default=0.1))
                refine_grad_clip = float(OmegaConf.select(cfg, "interface.refine.grad_clip", default=10.0))
                # Mirror-descent params
                refine_eta = float(OmegaConf.select(cfg, "interface.refine.eta", default=10.0))
                refine_temp = float(OmegaConf.select(cfg, "interface.refine.temperature", default=1.0))
                refine_resample = bool(OmegaConf.select(cfg, "interface.refine.resample", default=False))
                refine_resample_num = int(OmegaConf.select(cfg, "interface.refine.resample_num", default=0))
                refine_resample_sigma = float(OmegaConf.select(cfg, "interface.refine.resample_sigma", default=0.1))

                low = torch.tensor(env.action_bounds[0], dtype=torch.float32)
                high = torch.tensor(env.action_bounds[1], dtype=torch.float32)
                if prev_a_cont is None:
                    prev_a_cont = np.zeros((env.action_dim,), dtype=np.float32)
                prev_t = torch.tensor(prev_a_cont, dtype=torch.float32)

                # Proposal anchor from the algo's actor (if available).
                try:
                    a_actor = algo.act(obs, eval_mode=True) if hasattr(algo, "act") else None
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
                # compute energy per candidate
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                z0 = algo.encode(obs_t)  # [1,latent]
                Kc, Ad = int(cands.shape[0]), int(cands.shape[1])
                z0_rep = z0.unsqueeze(1).expand(1, Kc, z0.shape[-1]).reshape(Kc, -1)
                a1 = cands.to(device).reshape(Kc, Ad).unsqueeze(1)
                z1 = algo.predict(z0_rep, a1, delta=1)
                r_hat = algo.reward(z1).squeeze(-1)
                energy_t = (-r_hat).view(Kc)
                energy = energy_t.detach().cpu().numpy().reshape(Kc)
                idx_star = int(np.argmin(energy))
                a_star = cands[idx_star].cpu().numpy().astype(np.float32)
                e_star = float(energy[idx_star])

                # Current-step energy of previous action (candidate 0 is prev_action by construction)
                e_prev = float(energy[0])

                # Adaptive refinement when near-tied (small margin)
                # margin = e2 - e1 (lower energy is better). Small margin => boundary sensitive.
                e_sorted = np.sort(energy)
                margin = float(e_sorted[1] - e_sorted[0]) if e_sorted.shape[0] >= 2 else 0.0

                refined = False
                if refine_enabled and (margin < refine_margin_trigger):
                    try:
                        if refine_method.lower() in ("gd", "grad", "gradient"):
                            a_ref_t = torch.tensor(a_star, dtype=torch.float32)
                            a_refined_t, e_ref = refine_action_by_energy_gd(
                                algo,
                                obs_t.squeeze(0),
                                a_ref_t,
                                device=device,
                                steps=refine_steps,
                                step_size=refine_step_size,
                                action_low=low,
                                action_high=high,
                                grad_clip=refine_grad_clip,
                            )
                            a_star = a_refined_t.detach().cpu().numpy().astype(np.float32)
                            e_star = float(e_ref)
                        else:
                            # Mirror descent / exponentiated reweighting over candidates.
                            a_refined_t, e_ref = refine_action_by_energy_mirror(
                                algo,
                                obs_t.squeeze(0),
                                cands,
                                device=device,
                                energies=energy_t,
                                steps=refine_steps,
                                eta=refine_eta,
                                temperature=refine_temp,
                                action_low=low,
                                action_high=high,
                                resample=refine_resample,
                                resample_num=refine_resample_num,
                                resample_sigma=refine_resample_sigma,
                            )
                            a_star = a_refined_t.detach().cpu().numpy().astype(np.float32)
                            e_star = float(e_ref)
                        refined = True
                    except Exception:
                        refined = False

                total_margin_sum += float(margin)
                total_margin_count += 1
                total_refine_triggers += int(refined)

                # hysteresis: only switch if improvement exceeds margin, measured on current obs.
                # This avoids over-holding due to stale prev_e values.
                improvement = float(e_prev) - float(e_star)
                total_hys_decisions += 1
                if improvement > float(hys_margin):
                    action = a_star
                else:
                    action = prev_a_cont
                    total_hys_holds += 1
                prev_a_cont = action
                prev_e_cont = float(e_star)

                # (metrics are aggregated outside)

            elif env_name in ("procgen", "atari100k") and is_discrete and eval_policy in ("interface", "slow_only"):
                # Discrete energy interface: choose action minimizing energy=-predicted reward.
                A = int(getattr(env, "action_dim", components.get("action_dim", 0)))
                if prev_a_disc is None:
                    prev_a_disc = int(0)
                energy_h = int(OmegaConf.select(cfg, "method.energy_horizon", default=OmegaConf.select(cfg, "method.oracle_horizon", default=1)))
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                z0 = algo.encode(obs_t)
                energies = []
                for a in range(A):
                    a1 = torch.tensor([[int(a)]], dtype=torch.long, device=device)
                    z = z0
                    cum_r = 0.0
                    for _k in range(max(1, int(energy_h))):
                        z = algo.predict(z, a1, delta=1)
                        cum_r = cum_r + algo.reward(z).view(())
                    energies.append(float((-(cum_r)).detach().cpu()))
                energies = np.array(energies, dtype=np.float32)
                a_star = int(np.argmin(energies))
                e_star = float(energies[a_star])
                # margin = e2 - e1 (lower is better). Small margin => boundary sensitive.
                e_sorted = np.sort(energies)
                margin = float(e_sorted[1] - e_sorted[0]) if e_sorted.shape[0] >= 2 else 0.0
                total_margin_sum += float(margin)
                total_margin_count += 1

                e_prev = float(energies[prev_a_disc]) if 0 <= prev_a_disc < A else e_star
                improvement = float(e_prev) - float(e_star)

                # Count a "decision" only when the best action differs from the previous.
                if int(a_star) != int(prev_a_disc):
                    total_hys_decisions += 1
                    # Hysteresis: keep previous action unless improvement exceeds margin.
                    if improvement > float(hys_margin):
                        action = int(a_star)
                    else:
                        action = int(prev_a_disc)
                        total_hys_holds += 1
                else:
                    # No boundary crossing; keeping the same action is not a hysteresis hold.
                    action = int(prev_a_disc)
                prev_a_disc = int(action)

            else:
                action = algo.act(obs, eval_mode=True) if hasattr(algo, "act") else 0
                action = int(action) if np.isscalar(action) else action

            # ----------------
            # Diagnostics
            # ----------------
            total_steps += 1
            if action_hist_global is not None and np.isscalar(action) and 0 <= int(action) < action_hist_global.shape[0]:
                action_hist_global[int(action)] += 1

            ep_states.append(obs.copy() if isinstance(obs, np.ndarray) else np.array(obs))
            if np.isscalar(action):
                ep_actions.append(int(action))
            else:
                ep_actions_cont.append(np.array(action, dtype=np.float32))

            obs, r, done, info = env.step(action)
            ep_rew.append(r)
            if isinstance(info, dict) and info.get("failure", False):
                ep_failed = True

        # episode length diagnostics
        ep_lens.append(int(len(ep_rew)))

        # Aggregate hysteresis stats (per-episode state).
        total_hys_decisions += int(getattr(shield_state, "n_decisions", 0))
        total_hys_holds += int(getattr(shield_state, "n_holds", 0))

        failures_ep += int(ep_failed)
        returns.append(episode_return(ep_rew))
        tails.append(tail_mean(ep_rew, q=0.9))

        states_arr = np.stack(ep_states, axis=0)
        if env_name == "toy" and states_arr.shape[1] % 2 == 0:
            sl = slice(states_arr.shape[1] // 2, states_arr.shape[1])
        else:
            sl = None
        drifts.append(drift_metric(states_arr, idx_slice=sl))

        # ping_pong_rate / switch_rate
        if env_name == "dmc" and (not is_discrete):
            # continuous: define switch if ||a_t-a_{t-1}||>eps
            eps_sw = float(OmegaConf.select(cfg, "eval.switch_eps", default=1e-3))
            a_arr = np.stack(ep_actions_cont, axis=0) if len(ep_actions_cont) else np.zeros((0, env.action_dim), dtype=np.float32)
            if a_arr.shape[0] <= 2:
                switches.append(0.0)
                abas.append(0.0)
            else:
                dif = np.linalg.norm(a_arr[1:] - a_arr[:-1], axis=-1)
                switches.append(float(np.mean(dif > eps_sw)))
                # ABA: a[t] close to a[t-2] but far from a[t-1]
                dif02 = np.linalg.norm(a_arr[2:] - a_arr[:-2], axis=-1)
                dif12 = np.linalg.norm(a_arr[2:] - a_arr[1:-1], axis=-1)
                aba = np.mean((dif02 <= eps_sw) & (dif12 > eps_sw))
                abas.append(float(aba))
        else:
            abas.append(ping_pong_rate(ep_actions))
            switches.append(switch_rate(ep_actions))

    out = {
        "env": str(OmegaConf.select(cfg, "env.name", default="")),
        "task": str(OmegaConf.select(cfg, "env.task", default="")),
        "game": str(OmegaConf.select(cfg, "env.game", default="")),
        "procgen_env": str(OmegaConf.select(cfg, "env.env_name", default="")),
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "return_std": float(np.std(returns)) if returns else 0.0,
        "failures_ep": int(failures_ep),
        "episodes": int(n_episodes),
        "drift_mean": float(np.mean(drifts)) if drifts else 0.0,
        "aba_rate_mean": float(np.mean(abas)) if abas else 0.0,
        "switch_rate_mean": float(np.mean(switches)) if switches else 0.0,
        "tail_return_mean": float(np.mean(tails)) if tails else 0.0,
        "episode_len_mean": float(np.mean(ep_lens)) if ep_lens else 0.0,
        "episode_len_std": float(np.std(ep_lens)) if ep_lens else 0.0,
        "action_hist": action_hist_global.tolist() if action_hist_global is not None else [],
        "fallback_rate": float(total_fallback / max(total_steps, 1)),
        "shield_trigger_rate": float(total_shield_trigger / max(total_steps, 1)),
        "hysteresis_margin": float(hys_margin),
        "hysteresis_hold_rate": float(total_hys_holds / max(total_hys_decisions, 1)),
        "energy_margin_mean": float(total_margin_sum / max(total_margin_count, 1)),
        "refine_trigger_rate": float(total_refine_triggers / max(total_margin_count, 1)),
    }

    # Best-effort: add refinement diagnostics for continuous control.
    if str(OmegaConf.select(cfg, "env.name", default="")) == "dmc":
        try:
            out["refine_enabled"] = bool(OmegaConf.select(cfg, "interface.refine.enabled", default=True))
            out["refine_method"] = str(OmegaConf.select(cfg, "interface.refine.method", default="mirror"))
            out["refine_eta"] = float(OmegaConf.select(cfg, "interface.refine.eta", default=10.0))
            out["refine_temperature"] = float(OmegaConf.select(cfg, "interface.refine.temperature", default=1.0))
            out["refine_resample"] = bool(OmegaConf.select(cfg, "interface.refine.resample", default=False))
        except Exception:
            out["refine_enabled"] = False
    return out
