from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .corruption import DiffusionLikeCorruption
from .losses import LossWeights, prediction_loss, semantic_loss, stability_loss
from .metrics import feature_fd, mae_per_sample, mse_per_sample, psnr_from_mse, summarize_tail
from .probes import covariance_conditioning, normal_burden, pathology_score, relative_shift_and_sensitivity, support_deviation
from .targets import TargetSpec, construct_target, reconstruct_x0_from_target, spec_to_dict
from .utils import append_jsonl, ensure_dir, save_json


@dataclass
class RunArtifacts:
    summary_path: Path
    metrics_path: Path
    checkpoint_dir: Path


class EMA:
    def __init__(self, module: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        msd = module.state_dict()
        for k, v in msd.items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def copy_to(self, module: nn.Module) -> None:
        module.load_state_dict(self.shadow, strict=True)


class OneStepTrainer:
    def __init__(
        self,
        model: nn.Module,
        phi_map: nn.Module,
        corruption: DiffusionLikeCorruption,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        output_dir: str | Path,
        loss_weights: LossWeights,
        loss_kind: str = 'mse',
        mixed_precision: bool = False,
        save_every: int = 0,
        ema_decay: float = 0.0,
    ) -> None:
        self.model = model.to(device)
        self.phi_map = phi_map.to(device)
        self.corruption = corruption
        self.optimizer = optimizer
        self.device = device
        self.output_dir = ensure_dir(output_dir)
        self.loss_weights = loss_weights
        self.loss_kind = loss_kind
        self.metrics_path = self.output_dir / 'metrics.jsonl'
        self.ckpt_dir = ensure_dir(self.output_dir / 'checkpoints')
        self.samples_dir = ensure_dir(self.output_dir / 'samples')
        self.mixed_precision = bool(mixed_precision and device.type == 'cuda')
        self.save_every = int(save_every)
        self.ema_model = EMA(self.model, ema_decay) if ema_decay > 0 else None
        self.ema_phi_map = EMA(self.phi_map, ema_decay) if ema_decay > 0 else None
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.mixed_precision)
            self._autocast = lambda: torch.amp.autocast('cuda', enabled=self.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
            self._autocast = lambda: torch.cuda.amp.autocast(enabled=self.mixed_precision)

    def _state(self, x0: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        xt, eps = self.corruption.sample_xt(x0, t)
        return self.corruption.primitives(x0, xt, eps, t)

    def _checkpoint(self, name: str, step: int, best_val: float) -> Path:
        path = self.ckpt_dir / name
        torch.save({
            'model': self.model.state_dict(),
            'phi_map': self.phi_map.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': step,
            'best_val': best_val,
            'ema_model': self.ema_model.shadow if self.ema_model else None,
            'ema_phi_map': self.ema_phi_map.shadow if self.ema_phi_map else None,
        }, path)
        return path

    def load_ema_state(self, ckpt: dict[str, Any]) -> bool:
        ema_model = ckpt.get('ema_model')
        ema_phi = ckpt.get('ema_phi_map', ckpt.get('ema_phi'))
        if not (ema_model and ema_phi):
            return False
        if self.ema_model is None:
            self.ema_model = EMA(self.model, decay=0.999)
        if self.ema_phi_map is None:
            self.ema_phi_map = EMA(self.phi_map, decay=0.999)
        self.ema_model.shadow = {k: v.detach().clone() for k, v in ema_model.items()}
        self.ema_phi_map.shadow = {k: v.detach().clone() for k, v in ema_phi.items()}
        return True

    def _swap_to_ema(self):
        if not (self.ema_model and self.ema_phi_map):
            return None
        backup = (self.model.state_dict(), self.phi_map.state_dict())
        self.ema_model.copy_to(self.model)
        self.ema_phi_map.copy_to(self.phi_map)
        return backup

    def _restore_from_backup(self, backup):
        if backup is not None:
            self.model.load_state_dict(backup[0])
            self.phi_map.load_state_dict(backup[1])

    def train(
        self,
        loader: DataLoader,
        prediction_spec: TargetSpec,
        loss_spec: TargetSpec,
        max_steps: int,
        eval_every: int,
        collapse_threshold: float,
        tail_percentiles: list[int],
        mode: str,
        grad_clip: float = 1.0,
        num_eval_batches: int = 8,
        save_samples: bool = False,
    ) -> dict[str, Any]:
        self.model.train()
        self.phi_map.train()
        step = 0
        losses: list[float] = []
        grad_norms: list[float] = []
        best_val = float('inf')
        best_path = self.ckpt_dir / 'best.pt'
        last_eval: dict[str, Any] = {}
        start = time.time()
        loader_iter = iter(loader)
        while step < max_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            x0 = batch[0] if isinstance(batch, (list, tuple)) else batch
            x0 = x0.to(self.device)
            t = self.corruption.sample_t(x0.shape[0], self.device)
            state = self._state(x0, t)
            pred_target = construct_target(prediction_spec, state)
            loss_target = construct_target(loss_spec, state)

            with self._autocast():
                z_hat = self.model(state['xt'], t)
                mapped = self.phi_map(z_hat, t)
                lp = prediction_loss(z_hat, pred_target, loss_kind=self.loss_kind)
                ls = semantic_loss(mapped, loss_target, loss_kind=self.loss_kind)
                lstab = stability_loss(self.phi_map, z_hat)
                total = self.loss_weights.pred_weight * lp
                if mode != 'coupled':
                    total = total + self.loss_weights.sem_weight * ls
                total = total + self.loss_weights.stab_weight * lstab

            self.optimizer.zero_grad(set_to_none=True)
            if self.mixed_precision:
                self.scaler.scale(total).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.phi_map.parameters()), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.phi_map.parameters()), grad_clip)
                self.optimizer.step()

            if self.ema_model and self.ema_phi_map:
                self.ema_model.update(self.model)
                self.ema_phi_map.update(self.phi_map)

            losses.append(float(total.item()))
            grad_norms.append(float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm))
            append_jsonl({'step': step, 'loss': float(total.item()), 'pred_loss': float(lp.item()), 'sem_loss': float(ls.item()), 'stab_loss': float(lstab.item()), 'grad_norm': grad_norms[-1]}, self.metrics_path)

            if self.save_every > 0 and (step + 1) % self.save_every == 0:
                self._checkpoint(f'step_{step+1:06d}.pt', step + 1, best_val)
            if (step + 1) % eval_every == 0 or step + 1 == max_steps:
                backup = self._swap_to_ema()
                last_eval = self.evaluate(loader, prediction_spec, loss_spec, collapse_threshold, tail_percentiles, max_batches=num_eval_batches, save_samples=save_samples, step=step + 1)
                self._restore_from_backup(backup)
                save_json(last_eval, self.output_dir / 'last_eval.json')
                self._checkpoint('last.pt', step + 1, best_val)
                val_key = last_eval['quality']['mse']
                if val_key < best_val:
                    best_val = val_key
                    self._checkpoint('best.pt', step + 1, best_val)
            step += 1

        # always finalize last checkpoint once at end
        self._checkpoint('last.pt', max_steps, best_val)
        summary = {
            'mode': mode,
            'prediction_spec': spec_to_dict(prediction_spec),
            'loss_spec': spec_to_dict(loss_spec),
            'trainability': {
                'converged': bool(best_val < collapse_threshold),
                'diverged': False,
                'time_to_threshold': next((i for i, l in enumerate(losses) if l < collapse_threshold), None),
                'collapse_rate': float((torch.tensor(losses) > collapse_threshold).float().mean().item()),
            },
            'quality': last_eval.get('quality', {}),
            'tail': last_eval.get('tail', {}),
            'pathology': last_eval.get('pathology', {}),
            'runtime_sec': time.time() - start,
            'best_checkpoint': str(best_path),
            'grad_var': float(torch.tensor(grad_norms).var().item()) if len(grad_norms) > 1 else 0.0,
        }
        pathology = dict(summary['pathology'])
        pathology.setdefault('support_pix', 0.0)
        pathology.setdefault('support_perc', 0.0)
        pathology.setdefault('support_ssl', 0.0)
        pathology.setdefault('support_deviation', 0.0)
        pathology.setdefault('rho_nor', 0.0)
        pathology.setdefault('normal_burden', pathology['rho_nor'])
        pathology.setdefault('conditioning', 0.0)
        pathology.setdefault('covariance_conditioning', pathology['conditioning'])
        pathology.setdefault('relative_shift', 0.0)
        pathology.setdefault('prediction_sensitivity', 0.0)
        pathology['early_grad_var'] = summary['grad_var']
        pathology['pathology_score'] = pathology_score(pathology)
        summary['pathology'] = pathology
        save_json(summary, self.output_dir / 'summary.json')
        return summary

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        prediction_spec: TargetSpec,
        loss_spec: TargetSpec,
        collapse_threshold: float,
        tail_percentiles: list[int],
        max_batches: int = 8,
        save_samples: bool = False,
        step: int = 0,
    ) -> dict[str, Any]:
        self.model.eval()
        self.phi_map.eval()
        x0_all, recon_all, zhat_all, target_all = [], [], [], []
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x0 = batch[0] if isinstance(batch, (list, tuple)) else batch
            x0 = x0.to(self.device)
            t = self.corruption.sample_t(x0.shape[0], self.device)
            state = self._state(x0, t)
            z_hat = self.model(state['xt'], t)
            x0_hat = reconstruct_x0_from_target(prediction_spec, z_hat, state)
            pred_target = construct_target(prediction_spec, state)
            x0_all.append(x0)
            recon_all.append(x0_hat)
            zhat_all.append(z_hat)
            target_all.append(pred_target)

        x0_cat = torch.cat(x0_all, dim=0)
        recon_cat = torch.cat(recon_all, dim=0).clamp(-1, 1)
        zhat_cat = torch.cat(zhat_all, dim=0)
        target_cat = torch.cat(target_all, dim=0)
        mse = mse_per_sample(x0_cat, recon_cat)
        mae = mae_per_sample(x0_cat, recon_cat)
        psnr = psnr_from_mse(mse)
        pathology = {
            **support_deviation(target_cat, x0_cat),
            **normal_burden(target_cat, x0_cat),
            **covariance_conditioning(target_cat),
            **relative_shift_and_sensitivity(zhat_cat, target_cat, x0_cat),
        }
        pathology['pathology_score'] = pathology_score(pathology)
        tail = summarize_tail(mse, collapse_threshold, tail_percentiles)

        if save_samples:
            self._save_comparison_samples(x0_cat[:16], recon_cat[:16], self.samples_dir / f'samples_step_{step:06d}.png')

        return {
            'quality': {
                'mse': float(mse.mean().item()),
                'mae': float(mae.mean().item()),
                'psnr': float(psnr.mean().item()),
                'feature_fd': feature_fd(x0_cat, recon_cat),
            },
            'tail': tail,
            'pathology': pathology,
        }

    def _save_comparison_samples(self, x0: torch.Tensor, x0_hat: torch.Tensor, path: Path) -> None:
        pair = torch.cat([x0.detach().cpu(), x0_hat.detach().cpu()], dim=0).clamp(-1, 1)
        try:
            from torchvision.utils import save_image
            nrow = max(1, int(math.sqrt(x0.shape[0])))
            save_image((pair + 1.0) / 2.0, path, nrow=nrow)
            return
        except Exception:
            pass

        images = ((pair + 1.0) * 127.5).to(torch.uint8)
        n = images.shape[0]
        cols = max(1, int(math.sqrt(x0.shape[0])))
        rows = int(np.ceil(n / cols))
        c, h, w = images.shape[1:]
        grid = torch.zeros(c, rows * h, cols * w, dtype=torch.uint8)
        for i in range(n):
            r, col = divmod(i, cols)
            grid[:, r * h:(r + 1) * h, col * w:(col + 1) * w] = images[i]
        arr = grid.permute(1, 2, 0).numpy()
        if c == 1:
            arr = arr[:, :, 0]
        from PIL import Image
        Image.fromarray(arr).save(path)
