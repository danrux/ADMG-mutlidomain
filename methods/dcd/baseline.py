"""Typed adapter for Differentiable Causal Discovery."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any
import warnings

import numpy as np

from data.types import GeneratedDataset
from graphs import ADMG
from methods.base import BaselineResult


def _fit_dcd(observations: np.ndarray, **kwargs):
    try:
        from methods.dcd.algorithm import fit_dcd
    except ImportError as exc:
        if exc.name == "autograd":
            raise ImportError(
                "DCD requires autograd. Install it with "
                "'python -m pip install -r requirements.txt'."
            ) from exc
        raise
    return fit_dcd(observations, **kwargs)


def _select_observations(dataset: GeneratedDataset, mode: str) -> np.ndarray:
    if not dataset.domains:
        raise ValueError("DCD requires at least one data domain.")
    dimensions = {np.asarray(domain).shape[1] for domain in dataset.domains}
    if len(dimensions) != 1:
        raise ValueError("Every domain passed to DCD must have the same columns.")
    if mode == "reference":
        return np.asarray(dataset.domains[-1], dtype=float)
    if mode == "pooled":
        return np.vstack(dataset.domains).astype(float, copy=False)
    raise ValueError(f"Unknown DCD domain mode {mode!r}.")


def _standardize_observations(
    observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return column-standardized observations and their original-unit moments."""
    means = np.mean(observations, axis=0)
    scales = np.std(observations, axis=0, ddof=0)
    scale_tolerance = np.finfo(float).eps * np.maximum(
        1.0, np.max(np.abs(observations), axis=0)
    )
    constant = scales <= scale_tolerance
    if np.any(constant):
        columns = np.flatnonzero(constant).tolist()
        raise ValueError(
            "DCD cannot standardize constant or numerically constant columns: "
            f"{columns}."
        )
    return (observations - means) / scales, means, scales


@dataclass(frozen=True)
class DCDBaseline:
    """Estimate an ancestral, arid, or bow-free ADMG."""

    name: str = "DCD"

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
        mode = getattr(config, "dcd_domain_mode", "reference")
        observations = _select_observations(dataset, mode)
        if mode == "pooled" and len(dataset.domains) > 1:
            warnings.warn(
                "Pooled DCD assumes all domains share one Gaussian linear SEM; "
                "scaling, masks, and interventions generally violate that model.",
                UserWarning,
                stacklevel=2,
            )
        noise_type = dataset.metadata.get("generator_kwargs", {}).get("noise_type")
        if noise_type not in (None, "gauss"):
            warnings.warn(
                "DCD is derived for a Gaussian linear SEM with correlated errors; "
                f"the selected DGP uses {noise_type!r} noise.",
                UserWarning,
                stacklevel=2,
            )
        if dataset.metadata.get("dgp") == "nonlinear_mask":
            warnings.warn(
                "DCD assumes a linear SEM, but the selected DGP is nonlinear_mask.",
                UserWarning,
                stacklevel=2,
            )

        configured_seed = getattr(config, "dcd_random_state", None)
        internal_seed = (
            int(configured_seed)
            if configured_seed is not None
            else int(np.random.randint(0, np.iinfo(np.int32).max))
        )
        max_samples = getattr(config, "dcd_max_samples", None)
        if max_samples is not None:
            if max_samples < 2:
                raise ValueError("--dcd-max-samples must be at least 2.")
            if len(observations) > max_samples:
                rng = np.random.default_rng(internal_seed)
                indices = rng.choice(len(observations), size=max_samples, replace=False)
                observations = observations[indices]

        standardize = bool(getattr(config, "dcd_standardize", False))
        if standardize:
            observations, input_means, input_scales = _standardize_observations(
                observations
            )
        else:
            input_means = np.mean(observations, axis=0)
            input_scales = np.ones(observations.shape[1], dtype=float)

        start = perf_counter()
        fitted = _fit_dcd(
            observations,
            admg_class=getattr(config, "dcd_admg_class", "bowfree"),
            sparsity=getattr(config, "dcd_lambda", 0.05),
            threshold=getattr(config, "dcd_threshold", 0.05),
            max_iterations=getattr(config, "dcd_max_iterations", 100),
            constraint_tolerance=getattr(config, "dcd_h_tol", 1e-8),
            rho_max=getattr(config, "dcd_rho_max", 1e16),
            num_restarts=getattr(config, "dcd_num_restarts", 5),
            ricf_increment=getattr(config, "dcd_ricf_increment", 1),
            ricf_tolerance=getattr(config, "dcd_ricf_tol", 1e-4),
            ricf_refit_iterations=getattr(config, "dcd_ricf_refit_iterations", 100),
            optimizer_max_iterations=getattr(config, "dcd_optimizer_max_iterations", 15000),
            coefficient_bound=getattr(config, "dcd_coefficient_bound", 4.0),
            jitter=getattr(config, "dcd_jitter", 1e-8),
            random_state=internal_seed,
            verbose=getattr(config, "dcd_verbose", False),
        )
        runtime = perf_counter() - start
        # DCD is fit in standardized coordinates, but downstream coefficient and
        # covariance metrics use the DGP's original units.  With the repository's
        # [child, parent] convention, B_original[i, j] = s_i B_std[i, j] / s_j.
        directed_weights = (
            input_scales[:, None]
            * fitted.directed_weights
            / input_scales[None, :]
        )
        error_covariance = (
            input_scales[:, None]
            * fitted.error_covariance
            * input_scales[None, :]
        )
        mixing = np.linalg.inv(np.eye(fitted.directed.shape[0]) - directed_weights)
        return BaselineResult(
            graph=ADMG(
                directed=fitted.directed,
                bidirected=fitted.bidirected,
                directed_weights=directed_weights,
            ),
            runtime_seconds=runtime,
            diagnostics={
                "domain_mode": mode,
                "standardized": standardize,
                "input_means": input_means.tolist(),
                "input_scales": input_scales.tolist(),
                "n_samples": int(observations.shape[0]),
                "n_features": int(observations.shape[1]),
                "admg_class": getattr(config, "dcd_admg_class", "bowfree"),
                "bic": fitted.bic,
                "converged": fitted.converged,
                "selected_restart": fitted.restart,
                "constraint": fitted.constraint,
                "optimizer_failures": fitted.optimizer_failures,
                "random_state": internal_seed,
            },
            artifacts={
                "A_hat_raw": mixing,
                "raw_directed_weights": directed_weights,
                "omega_hat": error_covariance,
            },
        )
