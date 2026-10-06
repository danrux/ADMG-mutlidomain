"""LiNGAM baseline producing a directed mixed graph."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from data.types import GeneratedDataset
from evaluate import estimate_bidirected_from_Z_hat_list, select_bidirected_observations
from graphs import ADMG
from methods.base import BaselineResult


def select_base_domain(dataset: GeneratedDataset, mode: str = "auto") -> tuple[int, str]:
    """Select the domain used for residual-correlation estimation.

    Mask and hard-intervention DGPs append an unmasked/non-intervened reference
    domain. Other DGPs do not define such a domain, so ``auto`` falls back to
    the final domain and records that choice in diagnostics.
    """
    if not dataset.domains:
        raise ValueError("LiNGAM requires at least one data domain.")
    if mode == "last":
        return len(dataset.domains) - 1, "explicit-last"
    if mode != "auto":
        raise ValueError(f"Unknown LiNGAM base-domain mode {mode!r}.")

    dgp = dataset.metadata.get("dgp")
    parameters = dataset.domain_parameters
    if dgp in ("mask", "nonlinear_mask"):
        for index in range(len(parameters) - 1, -1, -1):
            if np.all(np.asarray(parameters[index]) == 1.0):
                return index, "all-ones-mask"
        raise ValueError("The mask DGP contains no fully unmasked base domain.")
    if dgp == "hard_intervention":
        for index in range(len(parameters) - 1, -1, -1):
            if len(parameters[index]) == 0:
                return index, "non-intervention"
        raise ValueError("The hard-intervention DGP contains no reference domain.")
    return len(dataset.domains) - 1, "last-domain-fallback"


def recover_residuals(observations: np.ndarray, directed_weights: np.ndarray) -> np.ndarray:
    """Recover structural residuals using B[i,j] for the edge j -> i."""
    observations = np.asarray(observations, dtype=float)
    directed_weights = np.asarray(directed_weights, dtype=float)
    if observations.ndim != 2:
        raise ValueError("Base-domain observations must be a two-dimensional array.")
    if directed_weights.shape != (observations.shape[1], observations.shape[1]):
        raise ValueError(
            "LiNGAM adjacency shape does not match the observed dimension: "
            f"{directed_weights.shape} versus {observations.shape[1]}."
        )
    centered = observations - observations.mean(axis=0, keepdims=True)
    unmixing = np.eye(observations.shape[1]) - directed_weights
    return centered @ unmixing.T


@dataclass(frozen=True)
class LiNGAMBaseline:
    """Fit DirectLiNGAM to concatenated domains and return an ADMG."""

    name: str = "LiNGAM"

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
        try:
            import lingam
        except ImportError as exc:
            raise ImportError(
                "LiNGAM requires the lingam package. Install it with "
                "'python -m pip install -r requirements.txt'."
            ) from exc

        dimensions = {domain.shape[1] for domain in dataset.domains}
        if len(dimensions) != 1:
            raise ValueError("Every domain passed to LiNGAM must have the same columns.")
        concatenated = np.vstack(dataset.domains).astype(float, copy=False)
        if not np.isfinite(concatenated).all():
            raise ValueError("LiNGAM does not accept NaN or infinite observations.")

        start = perf_counter()
        model = lingam.DirectLiNGAM(
            random_state=getattr(config, "lingam_random_state", None),
            measure=getattr(config, "lingam_measure", "pwling"),
        )
        model.fit(concatenated)
        raw_directed = np.asarray(model.adjacency_matrix_, dtype=float)

        directed_threshold = getattr(config, "lingam_directed_threshold", 0.0)
        if directed_threshold < 0:
            raise ValueError("--lingam-directed-threshold must be nonnegative.")
        directed_weights = raw_directed.copy()
        directed_weights[np.abs(directed_weights) <= directed_threshold] = 0.0
        np.fill_diagonal(directed_weights, 0.0)

        bidirected_observations, base_index, base_reason = (
            select_bidirected_observations(
                dataset.domains,
                dataset.domain_parameters,
                dataset.metadata.get("dgp"),
            )
        )
        z_hat_base = recover_residuals(bidirected_observations, directed_weights)
        bidirected_threshold = getattr(config, "bidir_threshold", 0.05)
        if bidirected_threshold < 0:
            raise ValueError("--bidir-threshold must be nonnegative.")
        bidirected = estimate_bidirected_from_Z_hat_list(
            z_hat_base,
            threshold=bidirected_threshold,
        )
        runtime = perf_counter() - start

        directed = (np.abs(directed_weights) > 0).astype(int)
        mixing = np.linalg.inv(np.eye(directed.shape[0]) - directed_weights)
        return BaselineResult(
            graph=ADMG(
                directed=directed,
                bidirected=bidirected,
                directed_weights=directed_weights,
            ),
            runtime_seconds=runtime,
            diagnostics={
                "causal_order": [int(value) for value in model.causal_order_],
                "n_samples": int(concatenated.shape[0]),
                "n_features": int(concatenated.shape[1]),
                "base_domain_index": None if base_index is None else int(base_index),
                "base_domain_reason": base_reason,
                "base_domain_samples": int(len(z_hat_base)),
                "directed_threshold": float(directed_threshold),
                "bidirected_threshold": float(bidirected_threshold),
            },
            artifacts={
                "raw_directed_weights": raw_directed,
                "z_hat_base": z_hat_base,
                "A_hat_raw": mixing,
                "model": model,
            },
        )
