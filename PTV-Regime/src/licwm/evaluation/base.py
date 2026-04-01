import glob
import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from licwm.data.domain_registry import build_domain_dataset
from licwm.data.collate import collate_multi_agent
from licwm.models.builders import build_model
from licwm.evaluation.predictive import evaluate_predictive
from licwm.evaluation.matched_geometry import evaluate_matched_geometry
from licwm.evaluation.counterfactual import evaluate_counterfactual
from licwm.evaluation.antisteg import evaluate_antisteg
from licwm.evaluation.control import evaluate_control
from licwm.planners.base import build_planner


def _find_existing_checkpoint() -> str | None:
    candidates = []
    for r in sorted(glob.glob("outputs/runs/*")):
        ck = os.path.join(r, "checkpoint_best.pt")
        if os.path.exists(ck):
            candidates.append(ck)
    return sorted(candidates)[-1] if candidates else None


def _bootstrap_train_if_needed(cfg) -> str | None:
    auto_train = bool(cfg.get("auto_train_if_missing_checkpoint", True))
    if not auto_train:
        return None
    from copy import deepcopy
    from licwm.training.engine import run_training

    train_cfg = deepcopy(cfg)
    if hasattr(train_cfg, "evaluator") and hasattr(train_cfg.evaluator, "name"):
        train_cfg.evaluator.name = "predictive"
    if hasattr(train_cfg, "trainer") and hasattr(train_cfg.trainer, "epochs"):
        train_cfg.trainer.epochs = min(int(getattr(train_cfg.trainer, "epochs", 1) or 1), 1)
    if hasattr(train_cfg, "trainer") and hasattr(train_cfg.trainer, "batch_size"):
        train_cfg.trainer.batch_size = min(int(getattr(train_cfg.trainer, "batch_size", 4) or 4), 4)
    if hasattr(train_cfg, "trainer") and hasattr(train_cfg.trainer, "history_len"):
        train_cfg.trainer.history_len = min(int(getattr(train_cfg.trainer, "history_len", 16) or 16), 6)
    if hasattr(train_cfg, "trainer") and hasattr(train_cfg.trainer, "pred_len"):
        train_cfg.trainer.pred_len = min(int(getattr(train_cfg.trainer, "pred_len", 12) or 12), 3)
    if hasattr(train_cfg, "domain") and hasattr(train_cfg.domain, "n_samples"):
        train_cfg.domain.n_samples = min(int(getattr(train_cfg.domain, "n_samples", 128) or 128), 12)
    if hasattr(train_cfg, "domain") and hasattr(train_cfg.domain, "num_agents"):
        train_cfg.domain.num_agents = min(int(getattr(train_cfg.domain, "num_agents", 16) or 16), 8)
    domain_name = getattr(train_cfg.domain, "name", "domain")
    task_name = getattr(train_cfg.task, "name", "task")
    seed = int(getattr(train_cfg, "seed", 0))
    outdir = os.path.abspath(f"outputs/runs/bootstrap_{domain_name}_{task_name}_{seed}")
    train_cfg.output_dir = outdir
    os.makedirs(os.path.dirname(outdir), exist_ok=True)
    produced = run_training(train_cfg)
    ckpt = os.path.join(produced, "checkpoint_best.pt")
    return ckpt if os.path.exists(ckpt) else None


def _load_model(cfg):
    model = build_model(cfg)
    ckpt = cfg.get("checkpoint_path")
    if ckpt is None:
        ckpt = _find_existing_checkpoint()
    if ckpt is None:
        ckpt = _bootstrap_train_if_needed(cfg)
    if ckpt is None:
        raise FileNotFoundError(
            "No checkpoint available. Either set checkpoint_path, run training first, "
            "or leave auto_train_if_missing_checkpoint=true to bootstrap a training run."
        )
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    return model


def _result_meta(cfg):
    return {
        'section': os.environ.get('LICWM_SECTION', 'manual'),
        'stage': 'eval',
        'domain': getattr(cfg.domain, 'name', 'unknown'),
        'task': getattr(cfg.task, 'name', 'unknown'),
        'evaluator': getattr(cfg.evaluator, 'name', 'unknown'),
        'model': getattr(cfg.model, 'name', 'unknown'),
        'ablation': getattr(cfg.ablation, 'name', 'full'),
        'seed': int(getattr(cfg, 'seed', 0)),
        'run_index': os.environ.get('LICWM_RUN_INDEX', '0'),
    }


def _write_result(res, cfg):
    os.makedirs('outputs/aggregates', exist_ok=True)
    os.makedirs('outputs/aggregates/raw', exist_ok=True)
    payload = dict(_result_meta(cfg))
    if isinstance(res, dict):
        payload.update(res)
    legacy = f"outputs/aggregates/eval_{cfg.evaluator.name}.json"
    with open(legacy, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    stem = "{section}__eval_{evaluator}__{domain}__{task}__{run_index}.json".format(**payload)
    raw_path = Path('outputs/aggregates/raw') / stem
    raw_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload


def run_evaluation(cfg):
    ds = build_domain_dataset(cfg, "test")
    loader = DataLoader(ds, batch_size=cfg.trainer.batch_size, shuffle=False, collate_fn=collate_multi_agent)
    model = _load_model(cfg)
    if cfg.evaluator.name == "predictive":
        res = evaluate_predictive(model, loader, cfg)
    elif cfg.evaluator.name == "matched_geometry":
        res = evaluate_matched_geometry(model, loader, cfg)
    elif cfg.evaluator.name == "counterfactual":
        res = evaluate_counterfactual(model, loader, cfg)
    elif cfg.evaluator.name == "antisteg":
        res = evaluate_antisteg(model, loader, cfg)
    elif cfg.evaluator.name == "control":
        res = evaluate_control(model, loader, build_planner(cfg), cfg)
    else:
        raise ValueError(cfg.evaluator.name)
    return _write_result(res, cfg)
