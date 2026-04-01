def evaluate_capacity(results: list[dict]):
    return {"capacity_delta": max(r["rollout_ade"] for r in results) - min(r["rollout_ade"] for r in results)}
