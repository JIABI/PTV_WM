import torch
from licwm.metrics.antisteg import tv_law, hf_law

def evaluate_antisteg(model, loader, cfg):
    with torch.no_grad():
        laws = []
        for b in loader:
            out = model(b.obs_hist, b.action_hist, b.event_hist, horizon=cfg.trainer.pred_len)
            laws.append(torch.cat([out.law_states["rho"], out.law_states["beta"], out.law_states["tau"]], dim=-1))
    law = torch.cat(laws)
    return {"tv_law": tv_law(law), "hf_ratio": hf_law(law)}
