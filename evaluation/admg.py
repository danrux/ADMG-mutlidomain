"""Evaluation protocol for acyclic directed mixed graphs."""

from __future__ import annotations

from typing import Any

from evaluate import compute_admg_metrics
from graphs import ADMG


def evaluate_admg(truth: ADMG, estimate: ADMG) -> dict[str, Any]:
    return compute_admg_metrics(
        truth.directed,
        truth.bidirected,
        estimate.directed,
        estimate.bidirected,
    )


__all__ = ["compute_admg_metrics", "evaluate_admg"]
