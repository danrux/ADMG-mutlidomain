"""Dispatch evaluation by graph semantics rather than baseline name."""

from __future__ import annotations

from typing import Any, Callable

from evaluation.admg import evaluate_admg
from evaluation.cpdag import evaluate_cpdag
from evaluation.dag import evaluate_dag, evaluate_weighted_dag
from evaluation.pag import evaluate_pag
from graphs import Graph, GraphKind


EVALUATOR_REGISTRY: dict[GraphKind, Callable[[Any, Any], dict[str, Any]]] = {
    GraphKind.DAG: evaluate_dag,
    GraphKind.WEIGHTED_DAG: evaluate_weighted_dag,
    GraphKind.CPDAG: evaluate_cpdag,
    GraphKind.PAG: evaluate_pag,
    GraphKind.ADMG: evaluate_admg,
}


def evaluate_graph(truth: Graph, estimate: Graph) -> dict[str, Any]:
    """Evaluate an estimate using the protocol implied by its representation."""
    try:
        evaluator = EVALUATOR_REGISTRY[estimate.kind]
    except KeyError as exc:
        raise ValueError(f"No evaluator registered for {estimate.kind!r}.") from exc
    return evaluator(truth, estimate)
