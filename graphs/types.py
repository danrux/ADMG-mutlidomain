"""Canonical graph representations exchanged by baselines and evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np


class GraphKind(str, Enum):
    DAG = "dag"
    WEIGHTED_DAG = "weighted_dag"
    CPDAG = "cpdag"
    PAG = "pag"
    ADMG = "admg"


class EndpointMark(IntEnum):
    NONE = 0
    TAIL = 1
    ARROW = 2
    CIRCLE = 3


def _validate_square(matrix: np.ndarray, name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got {matrix.shape}.")


@dataclass(frozen=True)
class DirectedGraph:
    adjacency: np.ndarray
    weighted: bool = False

    def __post_init__(self) -> None:
        _validate_square(self.adjacency, "adjacency")

    @property
    def kind(self) -> GraphKind:
        return GraphKind.WEIGHTED_DAG if self.weighted else GraphKind.DAG


@dataclass(frozen=True)
class ADMG:
    directed: np.ndarray
    bidirected: np.ndarray
    directed_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_square(self.directed, "directed")
        _validate_square(self.bidirected, "bidirected")
        if self.directed.shape != self.bidirected.shape:
            raise ValueError("Directed and bidirected matrices must have equal shapes.")
        if self.directed_weights is not None:
            _validate_square(self.directed_weights, "directed_weights")
            if self.directed_weights.shape != self.directed.shape:
                raise ValueError("Directed weights must match the graph shape.")

    @property
    def kind(self) -> GraphKind:
        return GraphKind.ADMG


@dataclass(frozen=True)
class EndpointGraph:
    """CPDAG/PAG represented by endpoint marks.

    ``endpoints[i, j]`` is the mark at node ``i`` on the edge between ``i``
    and ``j``. For example, i -> j has TAIL at [i, j] and ARROW at [j, i].
    """

    endpoints: np.ndarray
    graph_kind: GraphKind

    def __post_init__(self) -> None:
        _validate_square(self.endpoints, "endpoints")
        if self.graph_kind not in (GraphKind.CPDAG, GraphKind.PAG):
            raise ValueError("EndpointGraph kind must be CPDAG or PAG.")
        allowed = {mark.value for mark in EndpointMark}
        values = set(np.unique(self.endpoints).tolist())
        if not values.issubset(allowed):
            raise ValueError(f"Unknown endpoint marks: {sorted(values - allowed)}")
        present = self.endpoints != EndpointMark.NONE
        if not np.array_equal(present, present.T):
            raise ValueError("An endpoint must be present at both ends of every edge.")

    @property
    def kind(self) -> GraphKind:
        return self.graph_kind


Graph = DirectedGraph | ADMG | EndpointGraph
