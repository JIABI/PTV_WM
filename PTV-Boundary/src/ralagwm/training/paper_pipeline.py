"""Paper-scale training orchestration helpers."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from ralagwm.audit.ensemble import AuditEnsemble
from ralagwm.audit.heads import build_audit_head_for_obs
from ralagwm.baselines import PolicyWM, RankWM, ReconWM, ValueWM
from ralagwm.data.replay import ReplayBuffer
from ralagwm.envs import make_env
from ralagwm.models.ralag_wm import RALAGWM
from ralagwm.training.checkpointing import load_checkpoint, save_checkpoint
from ralagwm.training.loops import (
    collect_rollout,
    select_action_from_baseline,
    train_baseline_step,
)
from ralagwm.training.world_model_loop import (
    evaluate_deploy_policy,
    train_world_model_epoch,
    warmstart_replay,
)
from ralagwm.utils.io import dump_json, load_yaml, resolve_path

LOGGER = logging.getLogger("ralagwm.paper")

BASELINES = {
    "recon_wm": ReconWM,
    "value_wm": ValueWM,
    "policy_wm": PolicyWM,
    "rank_wm": RankWM,
}


def _get(cfg: Any, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _repo_path(*parts: str) -> Path:
    return _repo_root().joinpath(*parts)


def _get_device(cfg: Any) -> torch.device:
    requested = str(_get(cfg, "device", "cpu"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _env_label(env_cfg: Any) -> str:
    return str(_get(env_cfg, "label", _get(env_cfg, "name", _get(env_cfg, "env_id", "env"))))


def _resolve_manifest_path(path: str | Path) -> Path:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest / preset file not found: {path}")
    return path


def _infer_flat_obs_dim(env_cfg: Any, env: Any | None = None) -> int:
    obs_type = str(_get(env_cfg, "obs_type", "proprio"))
    if env is not None:
        spec = getattr(env, "spec", None)
        obs_shape = tuple(getattr(spec, "observation_shape", ()) or ()) if spec is not None else ()
        if obs_shape:
            if obs_type == "image":
                prod = 1
                for v in obs_shape:
                    prod *= int(v)
                return int(prod)
            if len(obs_shape) == 1:
                return int(obs_shape[0])
            prod = 1
            for v in obs_shape:
                prod *= int(v)
            return int(prod)
    if obs_type == "image":
        size = int(_get(env_cfg, "obs_size", _get(env_cfg, "obs_dim", 84)))
        channels = int(_get(env_cfg, "image_channels", 3))
        return channels * size * size
    return int(_get(env_cfg, "obs_dim", 16))


def _infer_obs_shape(env: Any, cfg: Any) -> tuple[int, ...]:
    """
    Infer observation shape from env.spec when available, otherwise from config/reset.

    For image observations, this function must preserve frame stacking information,
    so stacked Atari observations become e.g. (4, 84, 84, 3) rather than (84, 84, 3).
    """
    spec = getattr(env, "spec", None)
    if spec is not None and hasattr(spec, "obs_shape"):
        shape = getattr(spec, "obs_shape")
        if shape is not None:
            shape = tuple(int(v) for v in shape)
            if len(shape) > 0:
                return shape

    obs_type = str(_get(cfg.env, "obs_type", "proprio"))
    if obs_type == "image":
        size = int(_get(cfg.env, "obs_size", _get(cfg.env, "obs_dim", 84)))
        channels = 1 if bool(_get(cfg.env, "grayscale_obs", False)) else 3
        frame_stack = int(_get(cfg.env, "frame_stack", 1))

        # preserve frame stack as its own dimension if > 1
        if frame_stack > 1:
            return (frame_stack, size, size, channels)
        return (size, size, channels)

    if _get(cfg.env, "obs_dim", None) is not None:
        return (int(_get(cfg.env, "obs_dim")),)

    obs, _ = env.reset(seed=int(_get(cfg, "seed", 0)))
    if isinstance(obs, torch.Tensor):
        return tuple(int(v) for v in obs.shape)
    return tuple(int(v) for v in getattr(obs, "shape", (16,)))


def _infer_image_spec(obs_shape: tuple[int, ...]) -> tuple[int, int]:
    """

    Infer (image_size, image_channels) from image-like observation shape.

    Supported:

        - HWC          -> (H, C)

        - CHW          -> (H, C)

        - THWC         -> (H, T*C)

        - TCHW         -> (H, T*C)

    Examples:

        (84, 84, 3)       -> (84, 3)

        (3, 84, 84)       -> (84, 3)

        (4, 84, 84, 3)    -> (84, 12)

        (4, 3, 84, 84)    -> (84, 12)

    """

    if len(obs_shape) == 3:

        # HWC

        if obs_shape[-1] <= 16:

            h, w, c = obs_shape

            if h != w:
                raise ValueError(f"Expected square image, got {obs_shape}")

            return int(h), int(c)

        # CHW

        if obs_shape[0] <= 16:

            c, h, w = obs_shape

            if h != w:
                raise ValueError(f"Expected square image, got {obs_shape}")

            return int(h), int(c)

        raise ValueError(f"Unsupported 3D image obs shape: {obs_shape}")

    if len(obs_shape) == 4:

        # THWC

        if obs_shape[-1] <= 16:

            t, h, w, c = obs_shape

            if h != w:
                raise ValueError(f"Expected square stacked image, got {obs_shape}")

            return int(h), int(t * c)

        # TCHW

        if obs_shape[1] <= 16:

            t, c, h, w = obs_shape

            if h != w:
                raise ValueError(f"Expected square stacked image, got {obs_shape}")

            return int(h), int(t * c)

        raise ValueError(f"Unsupported 4D stacked image obs shape: {obs_shape}")

    raise ValueError(f"Expected image obs shape with 3 or 4 dims, got {obs_shape}")


def _infer_action_spec(env: Any, cfg: Any) -> tuple[str, int, int]:
    """
    Returns:
        action_type: "discrete" | "continuous"
        action_dim: model output dimension
        num_actions: number of valid discrete actions or chart/action budget proxy
    """
    spec = getattr(env, "spec", None)
    action_type = str(getattr(spec, "action_type", _get(cfg.env, "action_type", "discrete")))

    if action_type == "discrete":
        if hasattr(env, "action_space") and hasattr(env.action_space, "n"):
            num_actions = int(env.action_space.n)
        else:
            num_actions = int(getattr(spec, "num_actions", _get(cfg.env, "num_actions", 2)))

        # 对离散域，baseline / deploy / decoder 都应该对齐到合法动作数
        action_dim = num_actions
        return action_type, action_dim, num_actions

    action_dim = int(
        getattr(
            spec,
            "action_dim",
            _get(cfg.env, "action_dim", _get(cfg.env, "num_actions", 1)),
        )
    )

    num_actions = int(
        getattr(
            spec,
            "num_actions",
            _get(
                cfg.env,
                "num_actions",
                max(int(_get(cfg.chart, "chart_budget", 8)), action_dim),
            ),
        )
    )

    return action_type, action_dim, num_actions


def _prepare_obs_for_audit(obs: torch.Tensor, obs_type: str) -> torch.Tensor:
    """
    - image: keep layout; CNN head handles BCHW/HWC conversion internally
    - proprio: flatten to [B, D]
    """
    if obs_type == "image":
        return obs.float()
    if obs.dim() > 2:
        return obs.view(obs.shape[0], -1).float()
    return obs.float()


def _continuous_actions_to_score_targets(actions: torch.Tensor, num_actions: int) -> torch.Tensor:
    """
    Map continuous actions [B, action_dim] into score targets [B, num_actions].
    This is a scaffold proxy target for continuous-domain audit training.
    """
    if actions.dim() == 1:
        actions = actions.unsqueeze(-1)
    if actions.dim() > 2:
        actions = actions.view(actions.shape[0], -1)

    actions = torch.tanh(actions.float())
    _, act_dim = actions.shape

    if num_actions == act_dim:
        return actions

    idx = torch.linspace(0, act_dim - 1, num_actions, device=actions.device)
    lo = idx.floor().long()
    hi = idx.ceil().long()
    alpha = (idx - lo.float()).unsqueeze(0)

    lo_val = actions[:, lo]
    hi_val = actions[:, hi]
    return (1.0 - alpha) * lo_val + alpha * hi_val


def _build_audit_targets(
    batch: dict[str, torch.Tensor],
    action_type: str,
    num_actions: int,
) -> tuple[str, torch.Tensor]:
    """
    Returns:
        loss_kind: "ce" or "mse"
        target tensor
    """
    actions = batch["actions"]

    if action_type == "discrete":
        if actions.dim() > 1:
            actions = actions.view(actions.shape[0], -1)[:, 0]
        target = actions.long().clamp(min=0, max=num_actions - 1)
        return "ce", target

    target = _continuous_actions_to_score_targets(actions, num_actions=num_actions)
    return "mse", target


def _build_audit_ensemble_for_env(env: Any, cfg: Any, device: torch.device) -> AuditEnsemble:
    obs_shape = _infer_obs_shape(env, cfg)
    _, _, num_actions = _infer_action_spec(env, cfg)

    heads = [
        build_audit_head_for_obs(obs_shape=obs_shape, num_actions=num_actions)
        for _ in range(int(_get(cfg.audit, "num_audits", 3)))
    ]
    ensemble = AuditEnsemble(
        heads,
        trim_ratio=float(_get(cfg.audit, "trim_ratio", 0.1)),
    ).to(device)
    return ensemble


def apply_overrides(cfg: Any, overrides: dict[str, Any]) -> Any:
    out = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    for section, vals in overrides.items():
        target = out.setdefault(section, {})
        for k, v in vals.items():
            if isinstance(v, dict) and isinstance(target.get(k), dict):
                target[k].update(v)
            else:
                target[k] = v
    return out


def load_training_presets(
    path: str | Path = "inputs/manifests/paper_training_presets.yaml",
) -> dict[str, Any]:
    resolved = _resolve_manifest_path(path)
    return load_yaml(resolved).get("presets", {})


def build_domain_cfg(base_cfg: Any, domain: dict[str, Any], presets: dict[str, Any]) -> Any:
    from ralagwm.evaluation.evaluator import env_cfg_from_domain

    cfg = env_cfg_from_domain(base_cfg, domain)
    preset = presets.get(domain.get("group", ""), {})
    if preset:
        cfg = apply_overrides(cfg, preset)
    return cfg


def checkpoint_dir(run_name: str) -> Path:
    p = _repo_path("outputs", "checkpoints", run_name)
    p.mkdir(parents=True, exist_ok=True)
    return p


def metrics_dir(run_name: str) -> Path:
    p = _repo_path("outputs", "metrics", run_name)
    p.mkdir(parents=True, exist_ok=True)
    return p


def train_audit_for_domain(cfg: Any, run_name: str) -> Path:
    """
    Train a multi-audit ensemble for one domain.

    - image observations -> CNN audit heads
    - proprio observations -> MLP audit heads
    - discrete domains -> CE targets
    - continuous domains -> interpolated score-target proxy
    """
    env = make_env(cfg.env)
    device = _get_device(cfg)

    action_type, action_dim, num_actions = _infer_action_spec(env, cfg)
    obs_type = str(_get(cfg.env, "obs_type", "proprio"))

    model = _build_audit_ensemble_for_env(env, cfg, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(_get(cfg.trainer, "lr", 1e-3)))

    replay = ReplayBuffer(capacity=int(_get(cfg.trainer, "replay_capacity", 5000)))

    warm = collect_rollout(
        env,
        replay,
        policy_fn=lambda obs: env.sample_random_action(),
        episodes=int(_get(cfg.trainer, "warmup_episodes", 4)),
        max_steps=int(_get(cfg.env, "max_episode_steps", 100)),
        seed=int(_get(cfg, "seed", 0)),
    )

    hist: list[dict[str, Any]] = []
    max_steps = int(_get(cfg.runtime, "max_steps", 10))
    batch_size = int(_get(cfg.trainer, "batch_size", 16))

    for step in range(max_steps):
        batch = replay.sample_tensors(batch_size=batch_size, device=device)
        x = _prepare_obs_for_audit(batch["obs"], obs_type=obs_type)
        loss_kind, target = _build_audit_targets(batch, action_type=action_type, num_actions=num_actions)

        optimizer.zero_grad()
        out = model(x)
        scores = out.consensus_scores

        if loss_kind == "ce":
            loss = F.cross_entropy(scores, target)
        else:
            loss = F.mse_loss(scores, target.float())

        loss.backward()
        optimizer.step()

        row = {
            "step": step,
            "loss": float(loss.item()),
            "replay_size": len(replay),
            "loss_kind": loss_kind,
            "action_type": action_type,
            "num_actions": num_actions,
            "action_dim": action_dim,
        }
        hist.append(row)

        LOGGER.info(
            "[audit] domain=%s step=%d/%d loss=%.4f replay=%d kind=%s",
            _env_label(cfg.env),
            step + 1,
            max_steps,
            float(loss.item()),
            len(replay),
            loss_kind,
        )

    ckpt = checkpoint_dir(run_name) / "audit_ensemble.pt"
    save_checkpoint(
        str(ckpt),
        model,
        optimizer,
        extra={"warmup": warm, "history": hist},
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    dump_json(metrics_dir(run_name) / "train_audit.json", {"warmup": warm, "history": hist})
    env.close()
    return ckpt


def train_ralag_for_domain(cfg: Any, run_name: str, audit_ckpt: str | Path | None = None) -> Path:
    env = make_env(cfg.env)
    device = _get_device(cfg)

    obs_type = str(_get(cfg.env, "obs_type", "proprio"))
    obs_shape = _infer_obs_shape(env, cfg)
    obs_dim = _infer_flat_obs_dim(cfg.env, env)
    action_type, action_dim, num_actions = _infer_action_spec(env, cfg)

    if obs_type == "image":
        if len(obs_shape) == 4:
            # stacked image, e.g. (T, H, W, C) or (T, C, H, W)
            if obs_shape[-1] in (1, 3, 4):  # THWC
                frame_stack, image_size, _, base_channels = obs_shape
                image_channels = int(frame_stack * base_channels)
            elif obs_shape[1] in (1, 3, 4):  # TCHW
                frame_stack, base_channels, image_size, _ = obs_shape
                image_channels = int(frame_stack * base_channels)
            else:
                raise ValueError(f"Unsupported stacked image obs shape: {obs_shape}")
        elif len(obs_shape) == 3:
            image_size, image_channels = _infer_image_spec(obs_shape)
        else:
            raise ValueError(f"Unsupported image obs shape: {obs_shape}")
    else:
        image_size, image_channels = 84, 1

    model = RALAGWM(
        obs_type=obs_type,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=int(_get(cfg.model.encoder, "hidden_dim", 128)),
        latent_dim=int(_get(cfg.model.bottleneck, "latent_dim", 64)),
        backbone_kind="gru",
        deploy_kind=str(_get(cfg.deploy, "kind", "linear")),
        image_size=image_size,
        image_channels=image_channels,
        chart_mode=(
            "discrete"
            if action_type == "discrete"
            else ("highdim_continuous" if action_dim > 16 else "continuous")
        ),
        chart_budget=int(_get(cfg.chart, "chart_budget", min(8, max(2, num_actions)))),
        pool_budget=int(_get(cfg.chart, "pool_budget", max(16, num_actions))),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(_get(cfg.trainer, "lr", 1e-3)))

    audit = _build_audit_ensemble_for_env(env, cfg, device=device)
    if audit_ckpt is not None and Path(audit_ckpt).exists():
        load_checkpoint(str(audit_ckpt), audit, strict=False)
    audit.eval()

    replay = ReplayBuffer(capacity=int(_get(cfg.trainer, "replay_capacity", 5000)))
    warm = warmstart_replay(
        env,
        replay,
        episodes=int(_get(cfg.trainer, "warmup_episodes", 2)),
        max_steps=int(_get(cfg.env, "max_episode_steps", 100)),
        seed=int(_get(cfg, "seed", 0)),
    )

    hist: list[dict[str, Any]] = []
    max_epochs = int(_get(cfg.runtime, "max_steps", 10))

    for epoch in range(max_epochs):
        summary = train_world_model_epoch(
            model=model,
            optimizer=optimizer,
            env=env,
            replay=replay,
            audit_ensemble=audit,
            batch_size=int(_get(cfg.trainer, "batch_size", 8)),
            gradient_steps=int(_get(cfg.trainer, "gradient_steps", 4)),
            collection_episodes=int(_get(cfg.trainer, "collection_episodes", 1)),
            max_steps=int(_get(cfg.env, "max_episode_steps", 100)),
            kl_weight=float(_get(cfg.model.bottleneck, "kl_weight", 1e-3)),
            device=str(device),
            seed=int(_get(cfg, "seed", 0)) + epoch * 100,
        )

        eval_metrics = evaluate_deploy_policy(
            env,
            model,
            episodes=int(_get(cfg.runtime, "eval_episodes", 2)),
            max_steps=int(_get(cfg.env, "max_episode_steps", 100)),
            device=str(device),
            seed=int(_get(cfg, "seed", 0)) + 10000 + epoch,
        )

        row = {
            "epoch": epoch,
            **summary.__dict__,
            **{f"eval_{k}": v for k, v in eval_metrics.items()},
            "action_type": action_type,
            "action_dim": action_dim,
            "num_actions": num_actions,
            "image_size": image_size,
            "image_channels": image_channels,
        }
        hist.append(row)

        LOGGER.info(
            "[ralag] domain=%s epoch=%d/%d loss=%.4f eval_return=%.3f flip=%.3f",
            _env_label(cfg.env),
            epoch + 1,
            max_epochs,
            row.get("train_loss", 0.0),
            row.get("eval_mean_return", 0.0),
            row.get("eval_flip_proxy", 0.0),
        )

    ckpt = checkpoint_dir(run_name) / "ralag_wm.pt"
    save_checkpoint(
        str(ckpt),
        model,
        optimizer,
        extra={"warmup": warm, "history": hist, "final": hist[-1] if hist else {}},
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    dump_json(
        metrics_dir(run_name) / "train_ralag_wm.json",
        {"warmup": warm, "history": hist, "final": hist[-1] if hist else {}},
    )
    env.close()
    return ckpt


def _baseline_target(
    baseline_name: str,
    batch: dict[str, torch.Tensor],
    action_dim: int,
    num_actions: int,
) -> torch.Tensor:
    obs = batch["obs"]
    actions = batch["actions"]

    if baseline_name == "recon_wm":
        return batch["next_obs"].reshape(obs.shape[0], -1).float()

    if baseline_name == "value_wm":
        return batch["rewards"].unsqueeze(-1).repeat(1, max(num_actions, action_dim)).float()

    if baseline_name == "policy_wm":
        if actions.dim() == 1:
            return actions
        if actions.dim() == 2 and actions.shape[1] == 1:
            return actions.squeeze(-1)
        return actions.float()

    if actions.dim() == 1:
        actions = actions.unsqueeze(-1)
    if actions.dim() > 2:
        actions = actions.view(actions.shape[0], -1)
    if actions.shape[1] == 1:
        base = actions.float().repeat(1, max(num_actions, action_dim))
    else:
        base = actions.float()

    scores = _continuous_actions_to_score_targets(base, max(num_actions, action_dim))
    return torch.argsort(scores, dim=-1).float()


def train_baseline_for_domain(cfg: Any, run_name: str, baseline_name: str) -> Path:
    env = make_env(cfg.env)
    device = _get_device(cfg)

    obs_type = str(_get(cfg.env, "obs_type", "proprio"))
    obs_shape = _infer_obs_shape(env, cfg)
    obs_dim = _infer_flat_obs_dim(cfg.env, env)
    action_type, action_dim, num_actions = _infer_action_spec(env, cfg)

    if obs_type == "image":
        if len(obs_shape) == 4:
            if obs_shape[-1] in (1, 3, 4):
                frame_stack, image_size, _, base_channels = obs_shape
                image_channels = int(frame_stack * base_channels)
            elif obs_shape[1] in (1, 3, 4):
                frame_stack, base_channels, image_size, _ = obs_shape
                image_channels = int(frame_stack * base_channels)
            else:
                raise ValueError(f"Unsupported stacked image obs shape: {obs_shape}")
        elif len(obs_shape) == 3:
            image_size, image_channels = _infer_image_spec(obs_shape)
        else:
            raise ValueError(f"Unsupported image obs shape: {obs_shape}")
    else:
        image_size, image_channels = 84, 1

    model = BASELINES[baseline_name](
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=int(_get(cfg.model.encoder, "hidden_dim", 128)),
        obs_type=obs_type,
        image_size=image_size,
        image_channels=image_channels,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(_get(cfg.trainer, "lr", 1e-3)))
    replay = ReplayBuffer(capacity=int(_get(cfg.trainer, "replay_capacity", 5000)))

    warm = collect_rollout(
        env,
        replay,
        policy_fn=lambda obs: env.sample_random_action(),
        episodes=int(_get(cfg.trainer, "warmup_episodes", 2)),
        max_steps=int(_get(cfg.env, "max_episode_steps", 100)),
        seed=int(_get(cfg, "seed", 0)),
    )

    hist: list[dict[str, Any]] = []
    max_epochs = int(_get(cfg.runtime, "max_steps", 10))

    for epoch in range(max_epochs):
        collect_rollout(
            env,
            replay,
            policy_fn=lambda obs: select_action_from_baseline(model, obs, env, device=str(device)),
            episodes=int(_get(cfg.trainer, "collection_episodes", 1)),
            max_steps=int(_get(cfg.env, "max_episode_steps", 100)),
            seed=int(_get(cfg, "seed", 0)) + epoch * 100,
        )

        batch = replay.sample_tensors(batch_size=int(_get(cfg.trainer, "batch_size", 8)), device=device)
        obs = batch["obs"]

        target = _baseline_target(
            baseline_name=baseline_name,
            batch=batch,
            action_dim=action_dim,
            num_actions=num_actions,
        )

        metrics = train_baseline_step(model, obs, optimizer, target)
        metrics["epoch"] = epoch
        metrics["replay_size"] = len(replay)
        metrics["action_type"] = action_type
        metrics["action_dim"] = action_dim
        metrics["num_actions"] = num_actions
        metrics["image_size"] = image_size
        metrics["image_channels"] = image_channels
        hist.append(metrics)

        LOGGER.info(
            "[%s] domain=%s epoch=%d/%d loss=%.4f replay=%d",
            baseline_name,
            _env_label(cfg.env),
            epoch + 1,
            max_epochs,
            metrics["loss"],
            len(replay),
        )

    ckpt = checkpoint_dir(run_name) / f"{baseline_name}.pt"
    save_checkpoint(
        str(ckpt),
        model,
        optimizer,
        extra={"warmup": warm, "history": hist, "final": hist[-1] if hist else {}},
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    dump_json(
        metrics_dir(run_name) / f"train_{baseline_name}.json",
        {"warmup": warm, "history": hist, "final": hist[-1] if hist else {}},
    )
    env.close()
    return ckpt


def summarize_final_metrics(run_name: str) -> dict[str, Any]:
    """
    Collect final metrics from JSON artifacts under one run.
    """
    out: dict[str, Any] = {}
    mdir = metrics_dir(run_name)

    for path in sorted(mdir.glob("train_*.json")):
        try:
            out[path.stem] = json.loads(path.read_text())
        except Exception:
            out[path.stem] = {"error": f"failed_to_parse:{path.name}"}
    return out


def paper_training_jobs(domains: list[dict[str, Any]], methods: list[str]) -> int:
    """
    Total jobs = 1 audit job + len(methods) model jobs per domain.
    methods is expected to include 'ralag_wm' and/or baseline names.
    """
    return len(domains) * (1 + len(methods))