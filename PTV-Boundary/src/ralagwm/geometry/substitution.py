from __future__ import annotations

from dataclasses import replace, is_dataclass

import torch

from ralagwm.typing import ChartState, RALAGGeometry


def primitive_wise_substitution(
    pred: RALAGGeometry,
    target: RALAGGeometry,
    substitute_scores: bool = False,
    substitute_margin: bool = False,
    substitute_edge: bool = False,
) -> RALAGGeometry:
    return RALAGGeometry(
        centered_scores=target.centered_scores if substitute_scores else pred.centered_scores,
        margin=target.margin if substitute_margin else pred.margin,
        edge_sensitivity=target.edge_sensitivity if substitute_edge else pred.edge_sensitivity,
        top_action_index=target.top_action_index if (substitute_scores or substitute_margin) else pred.top_action_index,
        boundary_risk=target.boundary_risk if (substitute_scores or substitute_margin or substitute_edge) else pred.boundary_risk,
    )


def state_wise_substitution(pred_state: ChartState | None, target_state: ChartState | None) -> ChartState | None:
    if target_state is None:
        return pred_state
    if is_dataclass(target_state):
        return replace(target_state)
    return target_state


def dose_response_substitution(pred: RALAGGeometry, target: RALAGGeometry, lam: float) -> RALAGGeometry:
    lam_t = torch.tensor(lam, device=pred.centered_scores.device, dtype=pred.centered_scores.dtype)
    n=min(pred.centered_scores.reshape(-1).numel(), target.centered_scores.reshape(-1).numel())
    pc=pred.centered_scores.reshape(-1)[:n]
    tc=target.centered_scores.reshape(-1)[:n]
    centered=(1-lam_t)*pc + lam_t*tc if n>0 else pc
    pm=pred.margin.reshape(-1); tm=target.margin.reshape(-1); m=min(pm.numel(), tm.numel()); margin=((1-lam_t)*pm[:m] + lam_t*tm[:m]) if m>0 else pm
    n = min(pred.edge_sensitivity.numel(), target.edge_sensitivity.numel())
    edge_pred = pred.edge_sensitivity.reshape(-1)[:n]
    edge_tgt = target.edge_sensitivity.reshape(-1)[:n]
    edge = (1 - lam_t) * edge_pred + lam_t * edge_tgt if n > 0 else edge_pred
    pr=pred.boundary_risk.reshape(-1); tr=target.boundary_risk.reshape(-1); r=min(pr.numel(), tr.numel()); risk=((1-lam_t)*pr[:r] + lam_t*tr[:r]) if r>0 else pr
    top = target.top_action_index if lam >= 0.5 else pred.top_action_index
    return RALAGGeometry(centered_scores=centered, margin=margin, edge_sensitivity=edge, top_action_index=top, boundary_risk=risk)
