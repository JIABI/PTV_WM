import csv
import json
import os
import platform
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from licwm.data.domain_registry import build_domain_dataset
from licwm.data.collate import collate_multi_agent
from licwm.models.builders import build_model
from licwm.losses.build import compute_loss
from licwm.training.checkpointing import save_checkpoint
from licwm.training.optimizer import build_optimizer
from licwm.training.scheduler import build_scheduler
from licwm.training.callbacks import on_epoch_end
from licwm.utils.seed import seed_everything


def _env_log(outdir: str):
    with open(os.path.join(outdir, "env.json"), "w", encoding="utf-8") as f:
        json.dump({"platform": platform.platform(), "torch": torch.__version__}, f, indent=2)


def _run_epoch(model, loader, cfg, optimizer=None):
    train = optimizer is not None
    model.train(train)
    rows = []
    for batch in loader:
        out = model(batch.obs_hist, batch.action_hist, batch.event_hist, horizon=cfg.trainer.pred_len, teacher_forcing=cfg.trainer.teacher_forcing)
        loss, metrics = compute_loss(batch, out, cfg)
        law_beta = out.law_states["beta"].mean().item() if isinstance(out.law_states["beta"], torch.Tensor) else 0.0
        metrics["law_beta_mean"] = law_beta
        if train:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        rows.append(metrics)
    out = {k: sum(r[k] for r in rows) / max(1, len(rows)) for k in rows[0].keys()}
    return out


def _meta(cfg, outdir: str):
    return {
        'section': os.environ.get('LICWM_SECTION', 'manual'),
        'stage': 'train',
        'domain': getattr(cfg.domain, 'name', 'unknown'),
        'task': getattr(cfg.task, 'name', 'unknown'),
        'model': getattr(cfg.model, 'name', 'unknown'),
        'ablation': getattr(cfg.ablation, 'name', 'full'),
        'seed': int(getattr(cfg, 'seed', 0)),
        'run_dir': os.path.basename(outdir),
    }


def run_training(cfg):
    seed_everything(cfg.seed, cfg.deterministic)
    outdir = os.path.abspath(cfg.output_dir)
    os.makedirs(outdir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(outdir, "config_snapshot.yaml"))
    _env_log(outdir)

    train_ds = build_domain_dataset(cfg, "train")
    val_ds = build_domain_dataset(cfg, "val")
    train_loader = DataLoader(train_ds, batch_size=cfg.trainer.batch_size, shuffle=True, collate_fn=collate_multi_agent)
    val_loader = DataLoader(val_ds, batch_size=cfg.trainer.batch_size, shuffle=False, collate_fn=collate_multi_agent)

    model = build_model(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    best = float("inf")
    per_epoch = []
    meta = _meta(cfg, outdir)
    for epoch in range(cfg.trainer.epochs):
        train_m = _run_epoch(model, train_loader, cfg, optimizer)
        val_m = _run_epoch(model, val_loader, cfg, None)
        row = {**meta, "epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()}, **{f"val_{k}": v for k, v in val_m.items()}}
        per_epoch.append(row)
        on_epoch_end(epoch, train_m)
        save_checkpoint(os.path.join(outdir, "checkpoint_last.pt"), model, optimizer, epoch, row)
        if val_m["total"] < best:
            best = val_m["total"]
            save_checkpoint(os.path.join(outdir, "checkpoint_best.pt"), model, optimizer, epoch, row)
        scheduler.step()

    with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(per_epoch[-1], f, indent=2)
    with open(os.path.join(outdir, "per_epoch_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=per_epoch[0].keys()); w.writeheader(); w.writerows(per_epoch)
    return outdir
