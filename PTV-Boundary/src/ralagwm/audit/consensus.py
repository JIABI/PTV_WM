"""Consensus operations for multi-audit local boundary reference."""
from __future__ import annotations

import torch

from ralagwm.typing import AuditScores


def trimmed_mean_consensus(raw_scores: torch.Tensor, trim_ratio: float = 0.25) -> torch.Tensor:
    n = int(raw_scores.shape[0])
    trim = int(n * float(trim_ratio))
    sorted_scores, _ = torch.sort(raw_scores, dim=0)
    kept = sorted_scores[trim:n-trim] if n - 2 * trim > 0 else sorted_scores
    return kept.mean(dim=0)


def variance_disagreement(raw_scores: torch.Tensor) -> torch.Tensor:
    return raw_scores.var(dim=0, unbiased=False)


def build_audit_scores(raw_scores: torch.Tensor, trim_ratio: float = 0.25) -> AuditScores:
    consensus = trimmed_mean_consensus(raw_scores, trim_ratio=trim_ratio)
    disagreement = variance_disagreement(raw_scores)
    return AuditScores(raw_scores=raw_scores, consensus_scores=consensus, disagreement=disagreement)
