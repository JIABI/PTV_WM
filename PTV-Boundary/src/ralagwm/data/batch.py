from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch

Tensor = torch.Tensor


def _move(x: Any, device: torch.device | str) -> Any:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _move(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_move(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_move(v, device) for v in x)
    if hasattr(x, 'to') and callable(getattr(x, 'to')) and not isinstance(x, torch.nn.Module):
        try:
            return x.to(device)
        except Exception:
            return x
    return x


@dataclass
class AuditScores:
    raw_scores: Tensor
    consensus_scores: Tensor
    disagreement: Tensor

    def to(self, device: torch.device | str) -> 'AuditScores':
        return AuditScores(
            raw_scores=_move(self.raw_scores, device),
            consensus_scores=_move(self.consensus_scores, device),
            disagreement=_move(self.disagreement, device),
        )


@dataclass
class ChartState:
    anchor_action: Tensor
    metric_matrix: Tensor
    boundary_saliency: Tensor
    uncertainty: Tensor
    action_coords: Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> 'ChartState':
        return ChartState(
            anchor_action=_move(self.anchor_action, device),
            metric_matrix=_move(self.metric_matrix, device),
            boundary_saliency=_move(self.boundary_saliency, device),
            uncertainty=_move(self.uncertainty, device),
            action_coords=_move(self.action_coords, device),
            metadata=_move(self.metadata, device),
        )


@dataclass
class BICChart:
    actions: Tensor
    coords: Tensor
    edges: Tensor
    weights: Tensor
    info_matrix: Tensor
    selected_indices: Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> 'BICChart':
        return BICChart(
            actions=_move(self.actions, device),
            coords=_move(self.coords, device),
            edges=_move(self.edges, device),
            weights=_move(self.weights, device),
            info_matrix=_move(self.info_matrix, device),
            selected_indices=_move(self.selected_indices, device),
            metadata=_move(self.metadata, device),
        )


@dataclass
class RALAGGeometry:
    centered_scores: Tensor
    margin: Tensor
    edge_sensitivity: Tensor
    top_action_index: Tensor
    boundary_risk: Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> 'RALAGGeometry':
        return RALAGGeometry(
            centered_scores=_move(self.centered_scores, device),
            margin=_move(self.margin, device),
            edge_sensitivity=_move(self.edge_sensitivity, device),
            top_action_index=_move(self.top_action_index, device),
            boundary_risk=_move(self.boundary_risk, device),
            metadata=_move(self.metadata, device),
        )


@dataclass
class ModelOutputs:
    latent: Tensor | None = None
    pred_chart_state: ChartState | None = None
    pred_chart: BICChart | None = None
    pred_geometry: RALAGGeometry | None = None
    refined_geometry: RALAGGeometry | None = None
    refinement_gate: Tensor | None = None
    auxiliary: dict[str, Tensor] = field(default_factory=dict)
    losses: dict[str, Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # compatibility aliases
    @property
    def bottleneck_latent(self):
        return self.latent

    @property
    def predicted_chart_state(self):
        return self.pred_chart_state

    @property
    def predicted_chart(self):
        return self.pred_chart

    @property
    def predicted_geometry(self):
        return self.pred_geometry

    def to(self, device: torch.device | str) -> 'ModelOutputs':
        return ModelOutputs(
            latent=_move(self.latent, device),
            pred_chart_state=_move(self.pred_chart_state, device),
            pred_chart=_move(self.pred_chart, device),
            pred_geometry=_move(self.pred_geometry, device),
            refined_geometry=_move(self.refined_geometry, device),
            refinement_gate=_move(self.refinement_gate, device),
            auxiliary=_move(self.auxiliary, device),
            losses=_move(self.losses, device),
            metadata=_move(self.metadata, device),
        )


@dataclass
class Transition:
    obs: Tensor | Any
    action: Tensor | Any
    reward: Tensor | Any
    next_obs: Tensor | Any
    done: Tensor | Any
    info: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> 'Transition':
        return Transition(
            obs=_move(self.obs, device),
            action=_move(self.action, device),
            reward=_move(self.reward, device),
            next_obs=_move(self.next_obs, device),
            done=_move(self.done, device),
            info=_move(self.info, device),
        )


@dataclass
class RALAGBatch:
    obs: Tensor
    action: Tensor
    next_obs: Tensor
    next_action: Tensor | None = None
    chart_state: ChartState | None = None
    chart: BICChart | None = None
    geometry_target: RALAGGeometry | None = None
    done: Tensor | None = None
    reward: Tensor | None = None
    audit_scores: AuditScores | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> 'RALAGBatch':
        return RALAGBatch(
            obs=_move(self.obs, device),
            action=_move(self.action, device),
            next_obs=_move(self.next_obs, device),
            next_action=_move(self.next_action, device),
            chart_state=_move(self.chart_state, device),
            chart=_move(self.chart, device),
            geometry_target=_move(self.geometry_target, device),
            done=_move(self.done, device),
            reward=_move(self.reward, device),
            audit_scores=_move(self.audit_scores, device),
            metadata=_move(self.metadata, device),
        )

    @property
    def batch_size(self) -> int:
        return int(self.obs.shape[0])

    def with_metadata(self, **kwargs: Any) -> 'RALAGBatch':
        meta = dict(self.metadata)
        meta.update(kwargs)
        return replace(self, metadata=meta)
