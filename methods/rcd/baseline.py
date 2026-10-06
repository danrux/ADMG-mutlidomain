"""Typed adapter for the official LiNGAM RCD implementation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any
import warnings

import numpy as np

from data.types import GeneratedDataset
from graphs import ADMG
from methods.base import BaselineResult


def rcd_adjacency_to_admg(
    adjacency: np.ndarray,
    *,
    directed_threshold: float = 0.0,
) -> tuple[ADMG, np.ndarray]:
    """Convert RCD's finite/NaN adjacency encoding to a native ADMG.

    RCD uses ``adjacency[child, parent]`` for a directed coefficient and places
    NaN at both entries of a node pair inferred to share a latent confounder.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"RCD must return a square adjacency matrix, got {adjacency.shape}.")
    if directed_threshold < 0.0:
        raise ValueError("--rcd-directed-threshold must be nonnegative.")
    if np.isinf(adjacency).any():
        raise ValueError("RCD returned an infinite adjacency coefficient.")

    bidirected = np.logical_or(np.isnan(adjacency), np.isnan(adjacency.T))
    np.fill_diagonal(bidirected, False)
    weights = np.nan_to_num(adjacency, nan=0.0).copy()
    weights[bidirected] = 0.0
    weights[np.abs(weights) <= directed_threshold] = 0.0
    np.fill_diagonal(weights, 0.0)
    directed = np.abs(weights) > 0.0
    return (
        ADMG(
            directed=directed.astype(int),
            bidirected=bidirected.astype(int),
            directed_weights=weights,
        ),
        adjacency.copy(),
    )


def _select_observations(dataset: GeneratedDataset, mode: str) -> np.ndarray:
    if not dataset.domains:
        raise ValueError("RCD requires at least one data domain.")
    dimensions = {np.asarray(domain).shape[1] for domain in dataset.domains}
    if len(dimensions) != 1:
        raise ValueError("Every domain passed to RCD must have the same columns.")
    if mode == "reference":
        observations = dataset.domains[-1]
    elif mode == "pooled":
        observations = np.vstack(dataset.domains)
    else:
        raise ValueError(f"Unknown RCD domain mode {mode!r}.")
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 2 or observations.shape[0] < 2:
        raise ValueError("RCD requires a two-dimensional matrix with at least two rows.")
    if not np.isfinite(observations).all():
        raise ValueError("RCD does not accept NaN or infinite observations.")
    return observations


@dataclass(frozen=True)
class RCDBaseline:
    """Discover directed effects and shared latent confounders with RCD."""

    name: str = "RCD"

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
                "RCD requires the lingam package. Install it with "
                "'python -m pip install -r requirements.txt'."
            ) from exc

        mode = getattr(config, "rcd_domain_mode", "reference")
        observations = _select_observations(dataset, mode)
        if mode == "pooled" and len(dataset.domains) > 1:
            warnings.warn(
                "Pooled RCD assumes all rows follow one linear non-Gaussian SEM; "
                "heterogeneous scaling, masks, or interventions can violate its tests.",
                UserWarning,
                stacklevel=2,
            )
        noise_type = dataset.metadata.get("generator_kwargs", {}).get("noise_type")
        if noise_type == "gauss":
            warnings.warn(
                "RCD assumes non-Gaussian errors; use --noise-type exp or gumbel.",
                UserWarning,
                stacklevel=2,
            )
        if dataset.metadata.get("dgp") == "nonlinear_mask":
            warnings.warn(
                "RCD assumes a linear SEM, but the selected DGP is nonlinear_mask.",
                UserWarning,
                stacklevel=2,
            )

        max_samples = getattr(config, "rcd_max_samples", 300)
        original_samples = len(observations)
        if max_samples is not None:
            if max_samples < 2:
                raise ValueError("--rcd-max-samples must be at least 2.")
            if original_samples > max_samples:
                indices = np.random.choice(original_samples, max_samples, replace=False)
                observations = observations[indices]

        start = perf_counter()
        model = lingam.RCD(
            max_explanatory_num=getattr(config, "rcd_max_explanatory_num", 2),
            cor_alpha=getattr(config, "rcd_cor_alpha", 0.01),
            ind_alpha=getattr(config, "rcd_ind_alpha", 0.01),
            shapiro_alpha=getattr(config, "rcd_shapiro_alpha", 0.01),
            MLHSICR=getattr(config, "rcd_mlhsicr", False),
            bw_method=getattr(config, "rcd_bw_method", "mdbs"),
            independence=getattr(config, "rcd_independence", "hsic"),
            ind_corr=getattr(config, "rcd_ind_corr", 0.5),
        )
        model.fit(observations)
        graph, raw_adjacency = rcd_adjacency_to_admg(
            model.adjacency_matrix_,
            directed_threshold=getattr(config, "rcd_directed_threshold", 0.0),
        )
        runtime = perf_counter() - start
        mixing = np.linalg.inv(np.eye(graph.directed.shape[0]) - graph.directed_weights)
        ancestors = [sorted(int(value) for value in values) for values in model.ancestors_list_]
        return BaselineResult(
            graph=graph,
            runtime_seconds=runtime,
            diagnostics={
                "domain_mode": mode,
                "original_samples": int(original_samples),
                "n_samples": int(observations.shape[0]),
                "n_features": int(observations.shape[1]),
                "max_explanatory_num": int(getattr(config, "rcd_max_explanatory_num", 2)),
                "cor_alpha": float(getattr(config, "rcd_cor_alpha", 0.01)),
                "ind_alpha": float(getattr(config, "rcd_ind_alpha", 0.01)),
                "shapiro_alpha": float(getattr(config, "rcd_shapiro_alpha", 0.01)),
                "independence": getattr(config, "rcd_independence", "hsic"),
                "lingam_version": getattr(lingam, "__version__", "unknown"),
                "ancestors": ancestors,
            },
            artifacts={
                "A_hat_raw": mixing,
                "raw_adjacency": raw_adjacency,
                "raw_directed_weights": graph.directed_weights,
                "model": model,
            },
        )
