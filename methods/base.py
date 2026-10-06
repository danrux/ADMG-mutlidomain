"""Common baseline contract plus an adapter for existing estimators."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

import numpy as np

from data.types import GeneratedDataset
from graphs import DirectedGraph, Graph


@dataclass
class BaselineResult:
    graph: Graph
    runtime_seconds: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)

    @property
    def graph_kind(self):
        return self.graph.kind


class Baseline(Protocol):
    name: str

    def fit(
        self,
        dataset: GeneratedDataset,
        config: Any,
        *,
        T: int,
        n: int,
        xn: int,
    ) -> BaselineResult:
        ...


@dataclass(frozen=True)
class LegacyBaselineAdapter:
    """Adapt an existing ``estimate`` function without changing its inputs."""

    name: str
    estimator: Any

    def fit(
        self,
        dataset: GeneratedDataset,
        config: Any,
        *,
        T: int,
        n: int,
        xn: int,
    ) -> BaselineResult:
        # Keep this call equivalent to the old runner. In particular, Z_list
        # remains available to legacy methods until they are migrated away
        # from simulation-only diagnostics.
        start = perf_counter()
        raw_estimate, auxiliary = self.estimator(
            dataset.domains,
            dataset.latent_samples,
            T,
            n,
            xn,
            config,
        )

        # Import lazily because some legacy methods import evaluate.py.
        from evaluate import find_P_and_prune_fast

        weighted_adjacency = find_P_and_prune_fast(
            np.linalg.inv(raw_estimate),
            threshold=config.prune_threshold,
            relative_threshold=(config.prune_threshold != "auto"),
            strategy=config.prune_strategy,
            verbose=False,
        )
        return BaselineResult(
            graph=DirectedGraph(weighted_adjacency, weighted=True),
            runtime_seconds=perf_counter() - start,
            diagnostics=auxiliary,
            artifacts={"A_hat_raw": raw_estimate},
        )
