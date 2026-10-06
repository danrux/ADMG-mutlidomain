"""Evaluation protocols for directed and weighted directed graphs."""

from __future__ import annotations

from typing import Any

from evaluate import (
    collect_true_est_edges,
    compute_graph_metrics,
    compute_graph_metrics_with_reverse,
    compute_mse,
)
from graphs import DirectedGraph


def evaluate_dag(truth: DirectedGraph, estimate: DirectedGraph) -> dict[str, Any]:
    """Evaluate directed structure, preserving the repository's legacy metrics."""
    return compute_graph_metrics_with_reverse(truth.adjacency, estimate.adjacency)


def evaluate_weighted_dag(
    truth: DirectedGraph,
    estimate: DirectedGraph,
) -> dict[str, Any]:
    """Evaluate structure and weights for a weighted directed estimate."""
    metrics = evaluate_dag(truth, estimate)
    metrics.update(compute_mse(truth.adjacency, estimate.adjacency))
    return metrics


__all__ = [
    "collect_true_est_edges",
    "compute_graph_metrics",
    "compute_graph_metrics_with_reverse",
    "compute_mse",
    "evaluate_dag",
    "evaluate_weighted_dag",
]
