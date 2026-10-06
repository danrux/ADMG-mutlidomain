"""Original nonlinear causalAssembly/DRF data generation."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .graph import CausalAssemblyGraph, GraphSplit
from .interventions import InterventionDesign


DRF_SETUP_MESSAGE = (
    "causalAssembly DRF fitting requires a working R installation visible to rpy2 "
    "and the R package 'drf'. Install R, set R_HOME if needed, then run "
    "R -e \"install.packages('drf', repos='https://cloud.r-project.org')\"."
)


@dataclass(frozen=True)
class NonlinearDataset:
    train_domains: tuple[np.ndarray, ...]
    heldout_domains: tuple[np.ndarray, ...]
    observed_nodes: tuple[str, ...]
    intervention_parameters: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]


@dataclass
class FittedDRFSystem:
    graph: Any
    empirical_data: Any
    fit_row_count: int
    full_source_graph_node_count: int
    fit_data_sha256: str = ""
    empirical_data_sha256: str = ""


def generation_fingerprint(config: ExperimentConfig, split: GraphSplit,
                           design: InterventionDesign, seed: int) -> str:
    """Identify the data-generating choices, independently of learner settings."""
    values = config.as_dict()
    choices = {key: values[key] for key in (
        "setup", "n_regimes", "n_per_regime", "n_heldout_per_regime",
        "base_seed", "drf_fit_samples", "drf_smoothed_sources", "split",
        "interventions",
    )}
    choices.update(seed=seed, hidden_nodes=split.hidden_nodes,
                   observed_nodes=split.observed_nodes, design=design.as_dict())
    encoded = json.dumps(choices, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _values_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _write_regime_checkpoint(path: Path, train: np.ndarray, heldout: np.ndarray,
                             metadata: dict[str, Any]) -> None:
    """Publish one completed regime atomically, so interruption cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    metadata = {**metadata, "values_sha256": _values_digest(train, heldout)}
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, train=train, heldout=heldout,
                                metadata_json=json.dumps(metadata, sort_keys=True))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_regime_checkpoint(path: Path, expected: dict[str, Any],
                            config: ExperimentConfig,
                            observed_nodes: tuple[str, ...]):
    with np.load(path, allow_pickle=False) as archive:
        train = np.asarray(archive["train"])
        heldout = np.asarray(archive["heldout"])
        metadata = json.loads(str(archive["metadata_json"].item()))
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Checkpoint {path} has mismatched {key}; refusing to mix regimes")
    if metadata.get("observed_nodes") != list(observed_nodes):
        raise ValueError(f"Checkpoint {path} has a different observed column order")
    dimension = len(observed_nodes)
    if train.shape != (config.n_per_regime, dimension) or heldout.shape != (
        config.n_heldout_per_regime, dimension
    ):
        raise ValueError(f"Checkpoint {path} has an unexpected sample shape")
    if not np.issubdtype(train.dtype, np.number) or not np.issubdtype(heldout.dtype, np.number):
        raise ValueError(f"Checkpoint {path} does not contain numeric samples")
    if not np.isfinite(train).all() or not np.isfinite(heldout).all():
        raise ValueError(f"Checkpoint {path} contains nonfinite samples")
    if _values_digest(train, heldout) != metadata.get("values_sha256"):
        raise ValueError(f"Checkpoint {path} failed its sample hash check")
    return train, heldout, metadata["intervention_parameters"]


def build_ancestor_closed_production_line(
    context: CausalAssemblyGraph,
    split: GraphSplit,
):
    """Retain Stations 1+2 only after verifying they contain every ancestor of X.

    Station 2 alone would remove hidden Station-1 causes. Downstream Stations
    3-5 need no mechanisms if no path can lead back into the observed suffix.
    """
    import networkx as nx
    from causalAssembly.models_dag import ProductionLineGraph

    full_graph = context.full_graph
    relevant_nodes = set(context.station12_graph.nodes)
    observed_nodes = set(split.observed_nodes)
    if set(split.hidden_nodes) | observed_nodes != relevant_nodes:
        raise ValueError("The hidden/observed split must partition Stations 1+2")
    if not nx.is_directed_acyclic_graph(full_graph):
        raise ValueError("The full causalAssembly ground truth must be a DAG")
    outside_ancestors = set().union(
        *(nx.ancestors(full_graph, node) for node in observed_nodes)
    ) - relevant_nodes
    incoming_edges = [
        (source, target) for source, target in full_graph.edges
        if source not in relevant_nodes and target in relevant_nodes
    ]
    if outside_ancestors or incoming_edges:
        raise ValueError(
            "Stations 1+2 are not ancestor-closed for the observed suffix; "
            f"outside ancestors={sorted(outside_ancestors)}, "
            f"incoming edges={incoming_edges}"
        )

    reduced = ProductionLineGraph()
    for station in ("Station1", "Station2"):
        reduced.new_cell(station).graph = context.assembly_line.cells[station].graph.copy()
    within_cell_edges = set(reduced.graph.edges)
    crossing_edges = [
        edge for edge in context.station12_graph.edges if edge not in within_cell_edges
    ]
    reduced.connect_across_cells_manually(edges=crossing_edges)
    if (set(reduced.nodes) != relevant_nodes
        or set(reduced.edges) != set(context.station12_graph.edges)):
        raise AssertionError("Reduced production line does not reproduce Stations 1+2")
    return reduced


def fit_relevant_drf_system(
    context: CausalAssemblyGraph,
    split: GraphSplit,
    config: ExperimentConfig,
    seed: int,
) -> FittedDRFSystem:
    """Fit original DRFs on the complete ancestor-closed Station-1/2 system."""
    relevant_line = build_ancestor_closed_production_line(context, split)
    try:
        from causalAssembly.drf_fitting import fit_drf
        from causalAssembly.models_dag import ProductionLineGraph
        import rpy2.robjects as ro
    except Exception as exc:  # rpy2 raises non-ImportError exceptions without R
        raise RuntimeError(f"{DRF_SETUP_MESSAGE} Original error: {exc}") from exc
    try:
        data = ProductionLineGraph.get_data()
        if not set(relevant_line.nodes).issubset(data.columns):
            raise AssertionError("DRF fit data does not cover every relevant node")
        # Drop downstream columns before fit_drf makes its own data copy, and
        # retain only relevant empirical data for intervention quantiles.
        empirical_data = data.loc[:, relevant_line.nodes].copy()
        fit_data = empirical_data
        if config.drf_fit_samples is not None and config.drf_fit_samples < len(empirical_data):
            fit_data = empirical_data.sample(
                n=config.drf_fit_samples, random_state=seed, replace=False
            )
        # DRF tree construction is randomized in R, independently of the
        # NumPy generator used for intervention sampling and row selection.
        ro.r["set.seed"](int(seed))
        relevant_line.random_state = np.random.default_rng(seed)
        relevant_line.drf = fit_drf(relevant_line, data=fit_data)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"DRF fitting failed. {DRF_SETUP_MESSAGE} Original error: {exc}") from exc
    if set(relevant_line.drf) != set(relevant_line.nodes):
        raise AssertionError("DRFs were not fit for every relevant causalAssembly node")
    return FittedDRFSystem(
        relevant_line, empirical_data, len(fit_data), len(context.full_graph.nodes),
        _values_digest(fit_data.to_numpy(dtype=float)),
        _values_digest(empirical_data.to_numpy(dtype=float)),
    )


def _fresh_regime_graph(fitted_graph):
    """Fresh intervention registry per regime, sharing read-only fitted mechanisms."""
    # ProductionLineGraph.copy() copies topology, not the large fitted R forests.
    # Sampling only reads the forests; interventions live in each graph's own
    # registry, so sharing this dictionary is both safe and memory bounded.
    regime_graph = fitted_graph.copy()
    regime_graph.drf = fitted_graph.drf
    # Defensive clearing matters because causalAssembly keys interventions only
    # by target set and therefore overwrites repeated target sets.
    regime_graph.mutilated_dags = {}
    regime_graph.interventional_drf = {}
    return regime_graph


def _observed_summary(values: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    variance = values.var(axis=0)
    reference_variance = reference.var(axis=0)
    return {
        "mean": values.mean(axis=0).tolist(),
        "variance": variance.tolist(),
        "standardized_mean_shift": float(
            np.linalg.norm(
                (values.mean(axis=0) - reference.mean(axis=0))
                / np.sqrt(reference_variance + 1e-12)
            )
            / np.sqrt(values.shape[1])
        ),
        "rms_log_variance_shift": float(
            np.linalg.norm(
                np.log((variance + 1e-12) / (reference_variance + 1e-12))
            )
            / np.sqrt(values.shape[1])
        ),
    }


def generate_nonlinear_dataset(
    fitted: FittedDRFSystem,
    split: GraphSplit,
    design: InterventionDesign,
    config: ExperimentConfig,
    seed: int,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> NonlinearDataset:
    """Sample hidden interventions from the ancestor-complete fitted DRF system."""
    try:
        from sympy.stats import Normal
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("sympy is required for causalAssembly soft interventions") from exc
    hidden = set(split.hidden_nodes)
    if set(fitted.graph.nodes) != hidden | set(split.observed_nodes):
        raise AssertionError("Nonlinear fit must cover exactly the Station-1/2 system")
    if any(not set(regime.targets) <= hidden for regime in design.regimes):
        raise AssertionError("Nonlinear interventions must target hidden nodes only")
    empirical = fitted.empirical_data
    base_stats: dict[str, tuple[float, float, float, float]] = {}
    for target in design.selected_targets:
        values = np.asarray(empirical[target], dtype=float)
        q10, q90 = np.quantile(values, [0.1, 0.9])
        base_stats[target] = (
            float(np.median(values)),
            float(max(np.std(values, ddof=1), 1e-6)),
            float(q10),
            float(q90),
        )

    child_seeds = np.random.SeedSequence(seed).spawn(config.n_regimes)
    train: list[np.ndarray] = []
    heldout: list[np.ndarray] = []
    intervention_records: list[dict[str, Any]] = []
    fingerprint = generation_fingerprint(config, split, design, seed)
    for regime in design.regimes:
        checkpoint_path = (
            checkpoint_dir / f"regime_{regime.index:03d}.npz"
            if checkpoint_dir is not None else None
        )
        expected = {
            "generation_fingerprint": fingerprint,
            "regime": regime.index,
            "seed": seed,
            "strength": regime.strength,
            "targets": list(regime.targets),
            "drf_fit_row_count": fitted.fit_row_count,
            "fit_data_sha256": fitted.fit_data_sha256,
            "empirical_data_sha256": fitted.empirical_data_sha256,
        }
        if checkpoint_path is not None and checkpoint_path.exists():
            if not resume:
                raise FileExistsError(
                    f"Checkpoint {checkpoint_path} already exists; use --resume-generation"
                )
            values, remaining, parameters = _read_regime_checkpoint(
                checkpoint_path, expected, config, split.observed_nodes
            )
            train.append(values)
            heldout.append(remaining)
            intervention_records.append(parameters)
            continue
        graph = _fresh_regime_graph(fitted.graph)
        graph.random_state = np.random.default_rng(child_seeds[regime.index])
        details: dict[str, Any] = {}
        if regime.index:
            replacements = {}
            for target_offset, target in enumerate(regime.targets):
                median, empirical_std, q10, q90 = base_stats[target]
                robust_span = max(q90 - q10, empirical_std, 1e-6)
                direction = -1.0 if (regime.index + target_offset) % 2 else 1.0
                mean = median + direction * 0.5 * config.interventions.mean_shift_multipliers[regime.strength] * robust_span
                scale = max(
                    0.35 * robust_span * config.interventions.scale_multipliers[regime.strength],
                    1e-6,
                )
                # With only one effective Station-1 target, the strength/sign
                # cycle would otherwise repeat identical intervention laws.
                # An optional, configured trend makes all 17 intervention
                # domains distinct without changing the original benchmark.
                scale *= 1.0 + config.interventions.regime_scale_trend * (
                    (regime.index - 1) / max(1, config.n_regimes - 2)
                )
                replacements[target] = Normal(
                    f"regime_{regime.index}_{target}", mean, scale
                )
                details[target] = {
                    "family": "Normal",
                    "mean": float(mean),
                    "standard_deviation": float(scale),
                    "empirical_median": median,
                    "empirical_standard_deviation": empirical_std,
                    "empirical_q10": q10,
                    "empirical_q90": q90,
                }
            graph.intervene_on(nodes_values=replacements)
            frame = graph.sample_from_interventional_drf(
                which_intervention=0,
                size=config.n_per_regime + config.n_heldout_per_regime,
                smoothed=config.drf_smoothed_sources,
            )
        else:
            frame = graph.sample_from_drf(
                size=config.n_per_regime + config.n_heldout_per_regime,
                smoothed=config.drf_smoothed_sources,
            )
        # Sampling used all causally relevant Station-1/2 mechanisms; only now
        # hide H before passing the observed suffix to representation learning.
        values = frame.loc[:, split.observed_nodes].to_numpy(dtype=float)
        training = values[: config.n_per_regime]
        remaining = values[config.n_per_regime :]
        parameters = {"regime": regime.index, "strength": regime.strength, "targets": details}
        if checkpoint_path is not None:
            _write_regime_checkpoint(
                checkpoint_path, training, remaining,
                {**expected, "observed_nodes": list(split.observed_nodes),
                 "intervention_parameters": parameters},
            )
        train.append(training)
        heldout.append(remaining)
        intervention_records.append(parameters)
    diagnostics = {
        "fit_graph_node_count": len(fitted.graph.nodes),
        "full_source_graph_node_count": fitted.full_source_graph_node_count,
        "fit_scope": "ancestor_closed_stations_1_2",
        "fit_graph_covers_hidden_and_observed": True,
        "drf_fit_row_count": fitted.fit_row_count,
        "fit_data_sha256": fitted.fit_data_sha256,
        "empirical_data_sha256": fitted.empirical_data_sha256,
        "dataset_column_order": list(split.observed_nodes),
        "per_regime_observed": [
            _observed_summary(values, train[0]) for values in train
        ],
    }
    return NonlinearDataset(
        tuple(train),
        tuple(heldout),
        split.observed_nodes,
        tuple(intervention_records),
        diagnostics,
    )
