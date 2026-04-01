from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ralagwm.chart.generator import BICChartGenerator
from ralagwm.data.batch import ModelOutputs, RALAGBatch
from ralagwm.models.backbones import build_backbone
from ralagwm.models.bottleneck import GeometryBottleneck
from ralagwm.models.chart_state_predictor import ChartStatePredictor
from ralagwm.models.deploy_heads import build_deploy_head
from ralagwm.models.encoders import build_encoder
from ralagwm.models.geometry_decoder import GeometryDecoder
from ralagwm.models.refiner import SelectiveBoundaryRefiner


class RALAGWM(nn.Module):
    def __init__(
        self,
        obs_type: str = 'proprio',
        obs_dim: int = 16,
        action_dim: int = 6,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        backbone_kind: str = 'gru',
        deploy_kind: str = 'linear',
        image_size: int = 84,
        image_channels: int = 3,
        chart_mode: str = 'discrete',
        chart_budget: int = 8,
        pool_budget: int = 16,
        chart_tau_delta: float = 0.1,
        chart_knn: int = 3,
    ) -> None:
        super().__init__()
        self.obs_type = str(obs_type)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.image_size = int(image_size)
        self.image_channels = int(image_channels)
        self.chart_mode = str(chart_mode)
        self.chart_budget = int(chart_budget)
        self.pool_budget = int(pool_budget)
        self.action_type = 'discrete' if self.chart_mode == 'discrete' else 'continuous'
        self.coord_dim = 8 if self.action_type == 'discrete' else max(self.action_dim, 1)

        self.encoder = build_encoder(
            obs_type=self.obs_type,
            obs_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,
            image_size=self.image_size,
            image_channels=self.image_channels,
        )
        self.backbone = build_backbone(kind=backbone_kind, hidden_dim=self.hidden_dim)
        self.bottleneck = GeometryBottleneck(input_dim=self.hidden_dim, latent_dim=self.latent_dim)
        self.chart_state_predictor = ChartStatePredictor(
            input_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            coord_dim=self.coord_dim,
            pool_budget=self.pool_budget,
            action_type=self.action_type,
            chart_mode=self.chart_mode,
        )
        self.chart_generator = BICChartGenerator(
            chart_budget=self.chart_budget,
            feature_dim=self.coord_dim + 1,
            knn=chart_knn,
            tau_delta=chart_tau_delta,
            pool_budget=self.pool_budget,
            mode=self.chart_mode,
        )
        self.geometry_decoder = GeometryDecoder(
            input_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            chart_budget=self.chart_budget,
            coord_dim=self.coord_dim,
        )
        self.refiner = SelectiveBoundaryRefiner(
            input_dim=self.hidden_dim,
            action_dim=self.action_dim,
            chart_budget=self.chart_budget,
            coord_dim=self.coord_dim,
        )
        self.deploy_head = build_deploy_head(
            kind=deploy_kind,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            chart_budget=self.chart_budget,
            coord_dim=self.coord_dim,
        )
        aux_out = self.obs_dim if self.obs_type != 'image' else self.image_channels * self.image_size * self.image_size
        self.aux_future_head = nn.Linear(self.latent_dim, aux_out)

    def _prepare_input(self, obs: Any) -> tuple[torch.Tensor, Any | None]:
        if isinstance(obs, RALAGBatch):
            return obs.obs, obs
        if hasattr(obs, 'obs'):
            return obs.obs, obs
        return obs, None

    def _build_deploy_outputs(self, features: torch.Tensor, geom, chart, chart_state) -> tuple[torch.Tensor, torch.Tensor]:
        local_logits = self.deploy_head(features, geom, chart, chart_state)
        action_mask = chart.metadata.get('action_mask') if isinstance(chart.metadata, dict) else None
        if self.action_type == 'discrete':
            full_logits = torch.full((local_logits.shape[0], self.action_dim), -1e9, device=local_logits.device, dtype=local_logits.dtype)
            valid_actions = chart.actions.long().clamp(min=0, max=self.action_dim - 1)
            fill_logits = local_logits if action_mask is None else local_logits.masked_fill(~action_mask, -1e9)
            full_logits.scatter_(1, valid_actions, fill_logits)
            selected_action = torch.argmax(full_logits, dim=-1)
            return full_logits, selected_action

        masked_logits = local_logits if action_mask is None else local_logits.masked_fill(~action_mask, -1e9)
        best = torch.argmax(masked_logits, dim=-1)
        selected = torch.gather(chart.coords[..., : self.action_dim], 1, best[:, None, None].expand(-1, 1, min(chart.coords.shape[-1], self.action_dim))).squeeze(1)
        if selected.shape[-1] < self.action_dim:
            pad = torch.zeros(selected.shape[0], self.action_dim - selected.shape[-1], device=selected.device, dtype=selected.dtype)
            selected = torch.cat([selected, pad], dim=-1)
        return masked_logits, selected

    def forward(self, obs: Any) -> ModelOutputs:
        obs_tensor, batch = self._prepare_input(obs)
        enc = self.encoder(obs_tensor)
        h = self.backbone(enc)
        z, kl_stats = self.bottleneck(h)
        pred_chart_state = self.chart_state_predictor(z)
        pred_chart = self.chart_generator.generate(pred_chart_state, pred_chart_state.boundary_saliency, pred_chart_state.uncertainty)
        pred_geom = self.geometry_decoder(z, pred_chart_state, pred_chart)
        refined_geom, refine_gate = self.refiner(h, pred_geom, pred_chart_state, pred_chart)
        deploy_logits, selected_action = self._build_deploy_outputs(h, refined_geom, pred_chart, pred_chart_state)
        pred_next_obs = self.aux_future_head(z)
        outputs = ModelOutputs(
            latent=z,
            pred_chart_state=pred_chart_state,
            pred_chart=pred_chart,
            pred_geometry=pred_geom,
            refined_geometry=refined_geom,
            refinement_gate=refine_gate,
            auxiliary={'deploy_logits': deploy_logits, 'pred_next_obs': pred_next_obs},
            losses={},
            metadata={'action_type': self.action_type, 'chart_mode': self.chart_mode},
        )
        if isinstance(kl_stats, dict):
            outputs.losses.update(kl_stats)
        outputs.deploy_logits = deploy_logits
        outputs.logits = deploy_logits
        outputs.scores = deploy_logits
        outputs.selected_action = selected_action
        if batch is not None and getattr(batch, 'chart_state', None) is not None:
            outputs.metadata['oracle_chart_available'] = True
        return outputs
