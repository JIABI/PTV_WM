def event_alignment_score(event_hist, error_t):
    return float((event_hist.sum(dim=-1) * error_t).mean().item())
