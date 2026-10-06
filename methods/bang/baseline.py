"""Typed BANG baseline adapter."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any
import warnings

import numpy as np

from data.types import GeneratedDataset
from graphs import ADMG
from methods.bang.algorithm import fit_bang
from methods.base import BaselineResult


def _observations(dataset: GeneratedDataset, mode: str) -> np.ndarray:
    if not dataset.domains:
        raise ValueError("BANG requires at least one data domain.")
    dimensions = {np.asarray(domain).shape[1] for domain in dataset.domains}
    if len(dimensions) != 1:
        raise ValueError("Every domain passed to BANG must have the same columns.")
    if mode == "pooled":
        return np.vstack(dataset.domains)
    if mode == "reference":
        return np.asarray(dataset.domains[-1])
    raise ValueError(f"Unknown BANG domain mode {mode!r}.")


@dataclass(frozen=True)
class BANGBaseline:
    """Estimate a bow-free ADMG from non-Gaussian linear observations."""

    name: str = "BANG"

    def fit(
        self,
        dataset: GeneratedDataset,
        config: Any,
        *,
        T: int,
        n: int,
        xn: int,
    ) -> BaselineResult:
        del T, n, xn
        mode = getattr(config, "bang_domain_mode", "reference")
        observed = _observations(dataset, mode)
        if mode == "pooled" and len(dataset.domains) > 1:
            warnings.warn(
                "Pooled BANG assumes every domain follows the same observational "
                "SEM; heterogeneous scaling, masks, or interventions can violate "
                "its moment restrictions.",
                UserWarning,
                stacklevel=2,
            )
        if dataset.metadata.get("generator_kwargs", {}).get("noise_type") == "gauss":
            warnings.warn(
                "BANG assumes non-Gaussian errors; use --noise-type exp or gumbel.",
                UserWarning,
                stacklevel=2,
            )
        if dataset.metadata.get("dgp") == "nonlinear_mask":
            warnings.warn(
                "BANG assumes a linear SEM, but the selected DGP is nonlinear_mask.",
                UserWarning,
                stacklevel=2,
            )

        start = perf_counter()
        fitted = fit_bang(
            observed,
            degree=getattr(config, "bang_moment_degree", 3),
            level=getattr(config, "bang_level", 0.01),
            restriction=getattr(config, "bang_restriction", 1),
            max_set_size=getattr(config, "bang_max_set_size", None),
            max_iterations=getattr(config, "bang_max_iterations", 10_000),
            condition_limit=getattr(config, "bang_condition_limit", 1e12),
            verbose=getattr(config, "bang_verbose", False),
        )
        runtime = perf_counter() - start
        directed_threshold = getattr(config, "bang_directed_threshold", 0.0)
        if directed_threshold < 0:
            raise ValueError("--bang-directed-threshold must be nonnegative.")
        weights = fitted.direct_effect.copy()
        weights[np.abs(weights) <= directed_threshold] = 0.0
        directed = (np.abs(weights) > 0).astype(int)
        total_effect = np.linalg.inv(np.eye(weights.shape[0]) - weights)

        return BaselineResult(
            graph=ADMG(
                directed=directed,
                bidirected=fitted.bidirected,
                directed_weights=weights,
            ),
            runtime_seconds=runtime,
            diagnostics={
                "domain_mode": mode,
                "n_samples": int(observed.shape[0]),
                "n_features": int(observed.shape[1]),
                "moment_degree": int(getattr(config, "bang_moment_degree", 3)),
                "level": float(getattr(config, "bang_level", 0.01)),
                "restriction": int(getattr(config, "bang_restriction", 1)),
                "independence_tests": fitted.tests,
                "numerical_failures": fitted.numerical_failures,
                "maximum_set_size": fitted.maximum_set_size,
            },
            artifacts={
                "A_hat_raw": total_effect,
                "raw_directed_weights": fitted.direct_effect,
                "omega_hat": fitted.omega,
                "residuals": fitted.residuals,
            },
        )
