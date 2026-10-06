"""Evaluation protocol for completed partially directed acyclic graphs."""

from evaluation.endpoints import evaluate_endpoint_graph

evaluate_cpdag = evaluate_endpoint_graph

__all__ = ["evaluate_cpdag"]
