"""Thin adapters around the repository's existing representation learners."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .graph import GraphSplit, edge_matrices


REQUESTED_METHODS = ("diagGMM", "MuDo-nll")


def dataset_digest(domains: tuple[np.ndarray, ...] | list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for values in domains:
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.dtype.str.encode())
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The existing DiagGMM and MuDo-nll implementations require torch."
        ) from exc
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _method_estimator(name: str):
    if name == "diagGMM":
        from methods.diagGMM import estimate
    elif name == "MuDo-nll":
        from methods.MuDo_nll import estimate
    else:
        raise ValueError(f"Unsupported causalAssembly method {name!r}")
    return estimate


def _bidirected_from_residuals(
    residuals: np.ndarray,
    *,
    policy: str,
    regime_indices: list[int],
    alpha: float,
    minimum_correlation: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    from evaluate import estimate_bidirected_from_Z_hat_list

    threshold_edges = estimate_bidirected_from_Z_hat_list(
        residuals, threshold=minimum_correlation
    )
    n_samples, dimension = residuals.shape
    correlations = np.corrcoef(residuals, rowvar=False)
    p_values = np.ones((dimension, dimension), dtype=float)
    significant = np.zeros((dimension, dimension), dtype=int)
    for left in range(dimension):
        for right in range(left + 1, dimension):
            rho = float(np.clip(correlations[left, right], -0.999999, 0.999999))
            z = abs(np.arctanh(rho)) * math.sqrt(max(1, n_samples - 3))
            p_value = math.erfc(z / math.sqrt(2.0))
            p_values[left, right] = p_values[right, left] = p_value
            if threshold_edges[left, right] and p_value < alpha:
                significant[left, right] = significant[right, left] = 1
    return significant, {
        "residual_policy": policy,
        "regime_indices": regime_indices,
        "sample_count": int(n_samples),
        "correlations": correlations,
        "p_values": p_values,
    }


def _residual_bidirected(
    decoder: np.ndarray,
    dependence_domains: tuple[np.ndarray, ...],
    method: str,
    sample_source: str,
    alpha: float,
    minimum_correlation: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    from evaluate import get_aligned_z_hat

    selected = dependence_domains if method == "diagGMM" else dependence_domains[:1]
    policy = (
        f"mixed-regime-{sample_source}"
        if method == "diagGMM"
        else f"single-observational-{sample_source}"
    )
    residuals = np.vstack(get_aligned_z_hat(decoder, list(selected)))
    return _bidirected_from_residuals(
        residuals,
        policy=policy,
        regime_indices=list(range(len(selected))),
        alpha=alpha,
        minimum_correlation=minimum_correlation,
    )


def align_mixing_columns(true: np.ndarray, estimate: np.ndarray):
    """Hungarian cosine matching followed by least-squares column scaling."""
    from scipy.optimize import linear_sum_assignment

    true_norm = np.linalg.norm(true, axis=0)
    estimate_norm = np.linalg.norm(estimate, axis=0)
    similarity = np.abs(true.T @ estimate) / (
        true_norm[:, None] * estimate_norm[None, :] + 1e-15
    )
    true_rows, estimated_columns = linear_sum_assignment(-similarity)
    aligned = np.zeros_like(estimate, dtype=float)
    scales = np.zeros(true.shape[1], dtype=float)
    matched = np.zeros(true.shape[1], dtype=float)
    assignment = []
    for true_column, estimated_column in zip(true_rows, estimated_columns):
        source = estimate[:, estimated_column]
        scale = float(source @ true[:, true_column] / (source @ source + 1e-15))
        aligned[:, true_column] = scale * source
        scales[true_column] = scale
        matched[true_column] = similarity[true_column, estimated_column]
        assignment.append([int(true_column), int(estimated_column)])
    diagnostics = {
        "assignment_true_to_estimated": assignment,
        "absolute_cosine_similarity": matched.tolist(),
        "mean_absolute_cosine_similarity": float(matched.mean()),
        "column_scales": scales.tolist(),
    }
    return aligned, diagnostics


def postprocess_directed(inverse: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    """Normalize the recovered inverse, then apply the declared edge policy."""
    from evaluate import (
        _permute_and_normalize,
        adaptive_threshold,
        find_P_and_prune_fast,
    )

    if config.directed_postprocessing == "dag_pruned":
        return find_P_and_prune_fast(
            inverse,
            threshold=config.directed_threshold,
            relative_threshold=config.directed_threshold != "auto",
            strategy=config.prune_strategy,
            verbose=False,
        )
    normalized, _, _ = _permute_and_normalize(inverse.copy())
    weighted_b = np.eye(normalized.shape[0]) - normalized
    np.fill_diagonal(weighted_b, 0.0)
    if config.directed_threshold == "auto":
        cutoff = adaptive_threshold(weighted_b)
    else:
        maximum = float(np.max(np.abs(weighted_b)))
        cutoff = float(config.directed_threshold) * maximum
    weighted_b[np.abs(weighted_b) < cutoff] = 0.0
    return weighted_b


def evaluate_linear_oracle(dataset, split: GraphSplit, config: ExperimentConfig) -> dict[str, Any]:
    """Measure the finite-sample ceiling of the configured dependence test."""
    from evaluate import compute_admg_metrics

    truth_directed, truth_bidirected = edge_matrices(split)
    disturbance_domains = (
        dataset.composite_disturbances_train
        if config.dependence_sample_source == "training"
        else dataset.composite_disturbances_heldout
    )
    results: dict[str, Any] = {}
    for method in REQUESTED_METHODS:
        selected = disturbance_domains if method == "diagGMM" else disturbance_domains[:1]
        policy = (
            f"mixed-regime-{config.dependence_sample_source}"
            if method == "diagGMM"
            else f"single-observational-{config.dependence_sample_source}"
        )
        bidirected, diagnostics = _bidirected_from_residuals(
            np.vstack(selected),
            policy=policy,
            regime_indices=list(range(len(selected))),
            alpha=config.alpha,
            minimum_correlation=config.bidirected_min_abs_correlation,
        )
        metrics = compute_admg_metrics(
            truth_directed,
            truth_bidirected,
            truth_directed,
            bidirected,
        )
        metrics["overall_admg_shd"] = int(
            metrics["directed"]["shd"] + metrics["bidirected"]["shd"]
        )
        metrics["residual_policy"] = diagnostics["residual_policy"]
        metrics["sample_count"] = diagnostics["sample_count"]
        results[method] = metrics
    return results


def fit_and_evaluate_method(
    name: str,
    dataset,
    split: GraphSplit,
    config: ExperimentConfig,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Fit an existing learner and evaluate its declared directed-edge policy."""
    from evaluate import compute_admg_metrics, is_dag

    if name not in config.methods:
        raise ValueError(f"No method settings configured for {name}")
    train = list(dataset.train_domains)
    heldout = tuple(dataset.heldout_domains)
    train_hash = dataset_digest(train)
    heldout_hash = dataset_digest(heldout)
    if any(np.shares_memory(x, y) for x in train for y in heldout):
        raise AssertionError("Held-out observations must not alias representation-learning data")
    _seed_everything(seed)
    settings = asdict(config.methods[name])
    if settings["exact_components"] is None:
        settings["exact_components"] = config.n_regimes
    method_config = SimpleNamespace(**settings)
    estimator = _method_estimator(name)
    decoder, auxiliary = estimator(
        train,
        None,  # never expose oracle disturbances/latents to either method
        config.n_regimes,
        len(split.observed_nodes),
        len(split.observed_nodes),
        method_config,
    )
    if dataset_digest(train) != train_hash:
        raise AssertionError(f"{name} mutated the frozen shared dataset")
    inverse = np.linalg.inv(decoder)
    weighted_b = postprocess_directed(inverse, config)
    dependence_domains = (
        tuple(train)
        if config.dependence_sample_source == "training"
        else heldout
    )
    bidirected, residual_diagnostics = _residual_bidirected(
        decoder,
        dependence_domains,
        name,
        config.dependence_sample_source,
        config.alpha,
        config.bidirected_min_abs_correlation,
    )
    truth_directed, truth_bidirected = edge_matrices(split)
    metrics = compute_admg_metrics(
        truth_directed,
        truth_bidirected,
        weighted_b != 0,
        bidirected,
    )
    metrics["overall_edge_shd"] = int(
        metrics["directed"]["shd"] + metrics["bidirected"]["shd"]
    )
    metrics["directed_postprocessing"] = config.directed_postprocessing
    metrics["directed_is_dag"] = bool(is_dag(weighted_b))
    if metrics["directed_is_dag"]:
        metrics["overall_admg_shd"] = metrics["overall_edge_shd"]
    metrics["dataset_sha256"] = train_hash
    metrics["heldout_dataset_sha256"] = heldout_hash
    metrics["heldout_used_for_representation_learning"] = False
    metrics["dependence_sample_source"] = config.dependence_sample_source
    if config.setup == "linear":
        aligned, matching = align_mixing_columns(dataset.true_mixing, decoder)
        b_error = weighted_b - dataset.observed_coefficients
        metrics["linear"] = {
            "aligned_A_relative_frobenius_error": float(
                np.linalg.norm(aligned - dataset.true_mixing)
                / (np.linalg.norm(dataset.true_mixing) + 1e-15)
            ),
            "B_rmse": float(np.sqrt(np.mean(b_error**2))),
            "B_relative_frobenius_error": float(
                np.linalg.norm(b_error)
                / (np.linalg.norm(dataset.observed_coefficients) + 1e-15)
            ),
            "column_matching": matching,
        }
    scalar_auxiliary = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in auxiliary.items()
        if value is None or isinstance(value, (str, bool, int, float, np.generic))
    }
    metrics["fit_diagnostics"] = scalar_auxiliary
    metrics["residual_diagnostics"] = {
        key: value for key, value in residual_diagnostics.items()
        if not isinstance(value, np.ndarray)
    }
    arrays = {
        "A_hat_raw": np.asarray(decoder),
        "B_hat": weighted_b,
        "bidirected_hat": bidirected,
        "residual_correlations": residual_diagnostics["correlations"],
        "residual_p_values": residual_diagnostics["p_values"],
    }
    return metrics, arrays
