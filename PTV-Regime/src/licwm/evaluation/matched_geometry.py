from __future__ import annotations
import itertools
import torch
from licwm.metrics.matched_geometry import geometry_distance, law_separation, response_separation


def evaluate_matched_geometry(model, loader, cfg):
    model.eval()
    threshold_quantile = getattr(cfg.evaluator, "geom_quantile", 0.35)
    triplets = []
    with torch.no_grad():
        for b in loader:
            out = model(b.obs_hist, b.action_hist, b.event_hist, horizon=min(cfg.trainer.pred_len, 2), teacher_forcing=True)
            geom = b.obs_hist[:, -1]
            law = out.law_states["beta"][:, -1]
            resp = out.pred_obs[:, -1]
            for i, j in itertools.combinations(range(len(geom)), 2):
                triplets.append((
                    geometry_distance(geom[i], geom[j]),
                    law_separation(law[i], law[j]),
                    response_separation(resp[i], resp[j]),
                ))
    if not triplets:
        return {"geometry_match_quality": 0.0, "law_separation": 0.0, "response_separation": 0.0, "num_pairs": 0}
    pairs = torch.tensor(triplets)
    geom_cut = torch.quantile(pairs[:, 0], threshold_quantile)
    matched = pairs[pairs[:, 0] <= geom_cut]
    if matched.numel() == 0:
        matched = pairs
    return {
        "geometry_match_quality": float((1.0 / (1e-6 + matched[:, 0])).mean()),
        "law_separation": float(matched[:, 1].mean()),
        "response_separation": float(matched[:, 2].mean()),
        "num_pairs": int(matched.shape[0]),
    }
