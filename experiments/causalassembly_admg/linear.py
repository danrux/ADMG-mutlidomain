"""Linear SEM on the fixed causalAssembly Station-1/2 topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .graph import GraphSplit
from .interventions import InterventionDesign


@dataclass(frozen=True)
class LinearDataset:
    train_domains: tuple[np.ndarray, ...]
    heldout_domains: tuple[np.ndarray, ...]
    observed_nodes: tuple[str, ...]
    full_nodes: tuple[str, ...]
    coefficient_matrix: np.ndarray
    observed_coefficients: np.ndarray
    true_mixing: np.ndarray
    composite_disturbances_train: tuple[np.ndarray, ...]
    composite_disturbances_heldout: tuple[np.ndarray, ...]
    intervention_parameters: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]


def generate_coefficients(graph, nodes: tuple[str, ...], config: ExperimentConfig, seed: int):
    """Generate one fixed coefficient matrix; rows are children, columns parents."""
    rng = np.random.default_rng(seed)
    index = {node: i for i, node in enumerate(nodes)}
    coefficients = np.zeros((len(nodes), len(nodes)), dtype=float)
    for parent, child in sorted(graph.edges):
        magnitude = rng.uniform(config.coefficient_min, config.coefficient_max)
        coefficients[index[child], index[parent]] = magnitude * rng.choice((-1.0, 1.0))
    return coefficients


def _primitive_noise(rng, shape, distribution: str, scale: float) -> np.ndarray:
    if distribution in {"gaussian", "normal"}:
        return rng.normal(scale=scale, size=shape)
    if distribution == "laplace":
        return rng.laplace(scale=scale / np.sqrt(2.0), size=shape)
    if distribution == "uniform":
        return rng.uniform(-np.sqrt(3.0) * scale, np.sqrt(3.0) * scale, size=shape)
    if distribution in {"student", "student_t"}:
        return rng.standard_t(df=5, size=shape) * scale * np.sqrt(3.0 / 5.0)
    raise ValueError(f"Unknown noise distribution {distribution!r}")


def _sample_sem(
    coefficients: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    noise_distribution: str,
    noise_scale: float,
    interventions: dict[int, tuple[float, float]],
    intervention_mechanism: str = "replace",
) -> np.ndarray:
    values = np.zeros((n_samples, coefficients.shape[0]), dtype=float)
    primitive = _primitive_noise(
        rng, values.shape, noise_distribution, noise_scale
    )
    # ``nodes`` are topologically ordered, so incoming nonzeros occur earlier.
    for child in range(coefficients.shape[0]):
        if child in interventions:
            mean, scale = interventions[child]
            if intervention_mechanism == "noise_variance":
                if mean != 0.0:
                    raise ValueError("Noise-variance interventions must have zero mean")
                # Preserve the parent mechanism and change only this primitive
                # noise's standard deviation. All variables remain zero mean
                # in the population because the SEM has no intercepts.
                values[:, child] = (
                    values @ coefficients[child] + primitive[:, child] * scale
                )
            else:
                values[:, child] = rng.normal(mean, scale, size=n_samples)
        else:
            values[:, child] = values @ coefficients[child] + primitive[:, child]
    return values


def _domain_summary(values: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    mean = values.mean(axis=0)
    variance = values.var(axis=0)
    ref_mean = reference.mean(axis=0)
    ref_variance = reference.var(axis=0)
    standardized_mean_shift = np.linalg.norm(
        (mean - ref_mean) / np.sqrt(ref_variance + 1e-12)
    ) / np.sqrt(values.shape[1])
    log_variance_shift = np.linalg.norm(
        np.log((variance + 1e-12) / (ref_variance + 1e-12))
    ) / np.sqrt(values.shape[1])
    return {
        "mean": mean.tolist(),
        "variance": variance.tolist(),
        "standardized_mean_shift": float(standardized_mean_shift),
        "rms_log_variance_shift": float(log_variance_shift),
    }


def generate_linear_dataset(
    graph,
    split: GraphSplit,
    design: InterventionDesign,
    config: ExperimentConfig,
    seed: int,
) -> LinearDataset:
    """Generate frozen domains with the configured hidden intervention mechanism."""
    import networkx as nx

    nodes = tuple(nx.lexicographical_topological_sort(graph, key=str))
    coefficients = generate_coefficients(graph, nodes, config, seed)
    node_index = {node: i for i, node in enumerate(nodes)}
    observed_index = [node_index[node] for node in split.observed_nodes]
    hidden_index = [node_index[node] for node in split.hidden_nodes]
    b_xx = coefficients[np.ix_(observed_index, observed_index)]
    true_mixing = np.linalg.inv(np.eye(len(observed_index)) - b_xx)

    # The extra independent child stream draws one variance factor per target
    # and regime, without perturbing the existing sample streams.
    seeds = np.random.SeedSequence(seed).spawn(2 * config.n_regimes + 1)
    variance_rng = np.random.default_rng(seeds[-1])
    observational_full = _sample_sem(
        coefficients,
        config.n_per_regime,
        np.random.default_rng(seeds[0]),
        config.noise_distribution,
        config.noise_scale,
        {},
        config.interventions.mechanism,
    )
    hidden_mean = observational_full[:, hidden_index].mean(axis=0)
    hidden_std = observational_full[:, hidden_index].std(axis=0, ddof=1)
    hidden_stats = {
        node: (hidden_mean[i], max(hidden_std[i], 1e-6))
        for i, node in enumerate(split.hidden_nodes)
    }

    train_full: list[np.ndarray] = []
    heldout_full: list[np.ndarray] = []
    intervention_records: list[dict[str, Any]] = []
    for regime in design.regimes:
        replacements: dict[int, tuple[float, float]] = {}
        details: dict[str, Any] = {}
        if regime.index:
            for target_offset, target in enumerate(regime.targets):
                if config.interventions.mechanism == "noise_variance":
                    if config.interventions.variance_multiplier_distribution == "uniform":
                        variance_multiplier = float(variance_rng.uniform(
                            config.interventions.variance_multiplier_min,
                            config.interventions.variance_multiplier_max,
                        ))
                        scale_multiplier = float(np.sqrt(variance_multiplier))
                    else:
                        scale_multiplier = config.interventions.scale_multipliers[regime.strength]
                        variance_multiplier = float(scale_multiplier**2)
                    replacements[node_index[target]] = (0.0, float(scale_multiplier))
                    details[target] = {
                        "mechanism": "primitive_noise_variance",
                        "family": config.noise_distribution,
                        "mean": 0.0,
                        "base_standard_deviation": float(config.noise_scale),
                        "standard_deviation": float(config.noise_scale * scale_multiplier),
                        "variance_multiplier": variance_multiplier,
                    }
                else:
                    scale_multiplier = config.interventions.scale_multipliers[regime.strength]
                    base_mean, base_std = hidden_stats[target]
                    direction = -1.0 if (regime.index + target_offset) % 2 else 1.0
                    mean = base_mean + direction * config.interventions.mean_shift_multipliers[regime.strength] * base_std
                    scale = base_std * scale_multiplier
                    replacements[node_index[target]] = (float(mean), float(scale))
                    details[target] = {
                        "mechanism": "replacement",
                        "family": "Normal",
                        "mean": float(mean),
                        "standard_deviation": float(scale),
                        "base_mean": float(base_mean),
                        "base_standard_deviation": float(base_std),
                    }
        train = observational_full if regime.index == 0 else _sample_sem(
            coefficients,
            config.n_per_regime,
            np.random.default_rng(seeds[2 * regime.index]),
            config.noise_distribution,
            config.noise_scale,
            replacements,
            config.interventions.mechanism,
        )
        heldout = _sample_sem(
            coefficients,
            config.n_heldout_per_regime,
            np.random.default_rng(seeds[2 * regime.index + 1]),
            config.noise_distribution,
            config.noise_scale,
            replacements,
            config.interventions.mechanism,
        )
        train_full.append(train)
        heldout_full.append(heldout)
        intervention_records.append(
            {"regime": regime.index, "strength": regime.strength, "targets": details}
        )

    train_domains = tuple(values[:, observed_index] for values in train_full)
    heldout_domains = tuple(values[:, observed_index] for values in heldout_full)
    residual_operator = (np.eye(len(observed_index)) - b_xx).T
    disturbances_train = tuple(values @ residual_operator for values in train_domains)
    disturbances_heldout = tuple(values @ residual_operator for values in heldout_domains)
    total_effect = np.linalg.inv(np.eye(len(nodes)) - coefficients)[
        np.ix_(observed_index, hidden_index)
    ]
    per_regime = [
        _domain_summary(values, train_domains[0]) for values in train_domains
    ]
    disturbance_summaries = [
        {
            "mean": values.mean(axis=0).tolist(),
            "variance": values.var(axis=0).tolist(),
        }
        for values in disturbances_train
    ]
    diagnostics = {
        "condition_number_A_true": float(np.linalg.cond(true_mixing)),
        "composite_disturbance_covariance_observational": np.cov(
            disturbances_train[0], rowvar=False
        ).tolist(),
        "per_regime_observed": per_regime,
        "per_regime_composite_disturbance": disturbance_summaries,
        "hidden_to_observed_total_effect": total_effect.tolist(),
        "observed_structural_coefficients_invariant": True,
        "intervention_mechanism": config.interventions.mechanism,
        "variance_multiplier_distribution": config.interventions.variance_multiplier_distribution,
        "population_means_zero": config.interventions.mechanism == "noise_variance",
        "dataset_column_order": list(split.observed_nodes),
    }
    # There is one B_XX shared by construction; this explicit assertion guards refactors.
    for _ in design.regimes:
        np.testing.assert_array_equal(b_xx, coefficients[np.ix_(observed_index, observed_index)])
    return LinearDataset(
        train_domains,
        heldout_domains,
        split.observed_nodes,
        nodes,
        coefficients,
        b_xx,
        true_mixing,
        disturbances_train,
        disturbances_heldout,
        tuple(intervention_records),
        diagnostics,
    )
