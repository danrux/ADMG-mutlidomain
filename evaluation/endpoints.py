"""Shared endpoint-aware metrics for CPDAGs and PAGs."""

from __future__ import annotations

from typing import Any

import numpy as np

from graphs import EndpointGraph, EndpointMark


def _binary_metrics(truth: np.ndarray, estimate: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum(truth & estimate))
    fp = int(np.sum(~truth & estimate))
    fn = int(np.sum(truth & ~estimate))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "shd": int(np.sum(truth != estimate)),
    }


def evaluate_endpoint_graph(
    truth: EndpointGraph,
    estimate: EndpointGraph,
) -> dict[str, Any]:
    """Evaluate adjacencies once per pair and marks once per endpoint."""
    if truth.kind != estimate.kind:
        raise ValueError(f"Cannot compare {truth.kind.value} with {estimate.kind.value}.")
    if truth.endpoints.shape != estimate.endpoints.shape:
        raise ValueError("Truth and estimate must contain the same nodes.")

    upper = np.triu(np.ones(truth.endpoints.shape, dtype=bool), k=1)
    truth_edges = (truth.endpoints != 0)[upper]
    estimated_edges = (estimate.endpoints != 0)[upper]
    adjacency = _binary_metrics(truth_edges, estimated_edges)

    true_endpoint_mask = truth.endpoints != 0
    endpoint_total = int(np.sum(true_endpoint_mask))
    endpoint_correct = int(
        np.sum((truth.endpoints == estimate.endpoints) & true_endpoint_mask)
    )
    endpoint_errors = int(np.sum(truth.endpoints != estimate.endpoints))
    endpoint_marks = {}
    off_diagonal = ~np.eye(truth.endpoints.shape[0], dtype=bool)
    for mark in (EndpointMark.TAIL, EndpointMark.ARROW, EndpointMark.CIRCLE):
        endpoint_marks[mark.name.lower()] = _binary_metrics(
            (truth.endpoints == mark) & off_diagonal,
            (estimate.endpoints == mark) & off_diagonal,
        )
    return {
        "adjacency": adjacency,
        "endpoint_correct": endpoint_correct,
        "endpoint_total": endpoint_total,
        "endpoint_accuracy": endpoint_correct / (endpoint_total + 1e-12),
        "endpoint_errors": endpoint_errors,
        "endpoint_marks": endpoint_marks,
    }
