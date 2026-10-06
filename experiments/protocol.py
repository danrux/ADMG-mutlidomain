"""Reusable typed execution protocol for new baseline implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data.types import GeneratedDataset
from evaluation import evaluate_graph
from graphs import Graph
from methods.base import Baseline, BaselineResult


@dataclass(frozen=True)
class EvaluatedRun:
    result: BaselineResult
    metrics: dict[str, Any]


def fit_and_evaluate(
    baseline: Baseline,
    dataset: GeneratedDataset,
    truth_graph: Graph,
    config: Any,
    *,
    T: int,
    n: int,
    xn: int,
) -> EvaluatedRun:
    """Run any typed baseline and dispatch its graph-specific evaluator."""
    result = baseline.fit(dataset, config, T=T, n=n, xn=xn)
    metrics = evaluate_graph(truth_graph, result.graph)
    return EvaluatedRun(result=result, metrics=metrics)
