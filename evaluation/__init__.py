"""Graph-semantic evaluation API.

The root-level :mod:`evaluate` module remains available for compatibility with
existing methods and scripts.
"""

from evaluation.registry import EVALUATOR_REGISTRY, evaluate_graph

__all__ = ["EVALUATOR_REGISTRY", "evaluate_graph"]
