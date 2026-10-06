"""causalAssembly graph loading, latent projection, and structural split selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
from itertools import combinations
from typing import Any, Iterable

import numpy as np

from .config import SplitConfig


def _networkx():
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - dependency message
        raise RuntimeError("Install requirements-causalassembly.txt (networkx is missing).") from exc
    return nx


@dataclass(frozen=True)
class CausalAssemblyGraph:
    assembly_line: Any
    full_graph: Any
    station12_graph: Any
    station1_nodes: tuple[str, ...]
    station2_nodes: tuple[str, ...]
    causalassembly_version: str


@dataclass(frozen=True)
class GraphSplit:
    station2_topological_order: tuple[str, ...]
    cutoff: int
    hidden_nodes: tuple[str, ...]
    observed_nodes: tuple[str, ...]
    directed_edges: tuple[tuple[str, str], ...]
    bidirected_edges: tuple[tuple[str, str], ...]
    eligible_intervention_targets: tuple[str, ...]
    diagnostics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["directed_edges"] = [list(edge) for edge in self.directed_edges]
        value["bidirected_edges"] = [list(edge) for edge in self.bidirected_edges]
        return value


def load_causalassembly_graph() -> CausalAssemblyGraph:
    """Load the official graph and retain Stations 1+2 as one causal system."""
    try:
        from causalAssembly.models_dag import ProductionLineGraph
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "causalAssembly is unavailable. Install requirements-causalassembly.txt."
        ) from exc
    nx = _networkx()
    line = ProductionLineGraph.get_ground_truth()
    full = line.graph.copy()
    if not nx.is_directed_acyclic_graph(full):
        raise ValueError("causalAssembly ground truth is not a DAG")
    station1 = tuple(line.cells["Station1"].nodes)
    station2_set = set(line.cells["Station2"].nodes)
    station2_graph = full.subgraph(station2_set)
    station2 = tuple(nx.lexicographical_topological_sort(station2_graph, key=str))
    station12 = full.subgraph((*station1, *station2)).copy()
    return CausalAssemblyGraph(
        assembly_line=line,
        full_graph=full,
        station12_graph=station12,
        station1_nodes=station1,
        station2_nodes=station2,
        causalassembly_version=importlib.metadata.version("causalAssembly"),
    )


def _hidden_only_observed_reach(graph, start: str, observed: set[str]) -> set[str]:
    """Observed endpoints reachable from ``start`` with hidden internal nodes."""
    reached: set[str] = set()
    stack = list(graph.successors(start))
    visited: set[str] = set()
    while stack:
        node = stack.pop()
        if node in observed:
            reached.add(node)
            continue
        if node in visited:
            continue
        visited.add(node)
        stack.extend(graph.successors(node))
    return reached


def latent_projection(
    full_dag,
    hidden_nodes: Iterable[str],
    observed_nodes: Iterable[str],
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    """Return the DAG latent projection on observed nodes.

    A directed edge is induced by a directed path with only latent internal
    nodes. A bidirected edge is induced by a latent common ancestor whose two
    directed branches reach the observed endpoints through latent nodes. This
    formulation supports bows and hidden mediator chains.
    """
    nx = _networkx()
    hidden = set(hidden_nodes)
    observed_order = tuple(observed_nodes)
    observed = set(observed_order)
    if hidden & observed:
        raise ValueError("hidden_nodes and observed_nodes must be disjoint")
    if hidden | observed != set(full_dag.nodes):
        missing = set(full_dag.nodes) - (hidden | observed)
        extra = (hidden | observed) - set(full_dag.nodes)
        raise ValueError(f"hidden/observed partition mismatch; missing={missing}, extra={extra}")
    if not nx.is_directed_acyclic_graph(full_dag):
        raise ValueError("latent projection requires a DAG")

    directed: set[tuple[str, str]] = set()
    for source in observed_order:
        for target in _hidden_only_observed_reach(full_dag, source, observed):
            if target != source:
                directed.add((source, target))

    reachable_from_hidden = {
        node: _hidden_only_observed_reach(full_dag, node, observed) for node in hidden
    }
    bidirected: set[tuple[str, str]] = set()
    position = {node: index for index, node in enumerate(observed_order)}
    for endpoints in reachable_from_hidden.values():
        for left, right in combinations(endpoints, 2):
            edge = (left, right) if position[left] < position[right] else (right, left)
            bidirected.add(edge)
    return (
        tuple(sorted(directed, key=lambda edge: (position[edge[0]], position[edge[1]]))),
        tuple(sorted(bidirected, key=lambda edge: (position[edge[0]], position[edge[1]]))),
    )


def _split_diagnostics(graph, station1, station2_order, cutoff: int) -> dict[str, Any]:
    nx = _networkx()
    hidden = tuple((*station1, *station2_order[:cutoff]))
    observed = tuple(station2_order[cutoff:])
    hidden_set, observed_set = set(hidden), set(observed)
    projection_directed, projection_bidirected = latent_projection(
        graph, hidden, observed
    )
    direct_observed = tuple(sorted(
        edge for edge in graph.edges if edge[0] in observed_set and edge[1] in observed_set
    ))
    crossing = tuple(sorted(
        edge for edge in graph.edges if edge[0] in hidden_set and edge[1] in observed_set
    ))
    descendant_sets = {
        node: sorted(set(nx.descendants(graph, node)) & observed_set)
        for node in hidden
    }
    eligible = tuple(sorted(node for node, descendants in descendant_sets.items() if descendants))
    violating = tuple(
        (observed_node, hidden_node)
        for observed_node in observed
        for hidden_node in hidden
        if nx.has_path(graph, observed_node, hidden_node)
    )
    return {
        "cutoff": cutoff,
        "hidden_prefix_fraction": cutoff / len(station2_order),
        "n_hidden": len(hidden),
        "n_observed": len(observed),
        "n_direct_observed_edges": len(direct_observed),
        "n_projected_directed_edges": len(projection_directed),
        "n_hidden_to_observed_edges": len(crossing),
        "n_true_bidirected_edges": len(projection_bidirected),
        "n_eligible_targets": len(eligible),
        "eligible_targets": list(eligible),
        "observed_descendants": descendant_sets,
        "ordering_violations": [list(edge) for edge in violating],
        "direct_observed_edges": [list(edge) for edge in direct_observed],
        "hidden_to_observed_edges": [list(edge) for edge in crossing],
    }


def select_graph_split(context: CausalAssemblyGraph, config: SplitConfig) -> GraphSplit:
    """Choose the configured graph-only hidden/observed Station-1/2 split."""
    station2 = context.station2_nodes
    m = len(station2)
    if config.mode == "station1_only":
        if not context.station1_nodes:
            raise ValueError("station1_only split requires Station-1 nodes")
        # A zero cutoff is intentional: Station 1 is hidden, all of Station 2
        # is observed, and no Station-2 mechanism may be intervened on.
        candidates = [
            _split_diagnostics(context.station12_graph, context.station1_nodes, station2, 0)
        ]
    else:
        low = max(1, int(np.ceil(config.minimum_fraction * m)))
        high = min(m - config.minimum_observed, int(np.floor(config.maximum_fraction * m)))
        if low > high:
            raise ValueError("No cutoff satisfies the configured fraction/observed-size bounds")
        candidates = [
            _split_diagnostics(context.station12_graph, context.station1_nodes, station2, k)
            for k in range(low, high + 1)
        ]

    def deficits(row: dict[str, Any]) -> tuple[int, ...]:
        return (
            max(0, config.minimum_observed - row["n_observed"]),
            max(0, config.minimum_observed_edges - row["n_direct_observed_edges"]),
            max(0, config.minimum_hidden_to_observed_edges - row["n_hidden_to_observed_edges"]),
            max(0, config.minimum_bidirected_edges - row["n_true_bidirected_edges"]),
            max(0, config.minimum_eligible_targets - row["n_eligible_targets"]),
            len(row["ordering_violations"]),
        )

    chosen = min(
        candidates,
        key=lambda row: (
            int(any(deficits(row))),
            sum(deficits(row)),
            abs(row["hidden_prefix_fraction"] - config.hidden_prefix_fraction),
            -row["n_direct_observed_edges"],
            -row["n_true_bidirected_edges"],
            row["cutoff"],
        ),
    )
    if chosen["ordering_violations"]:
        raise AssertionError("Observed Station-2 nodes may not be ancestors of hidden nodes")
    if config.mode == "station1_only" and not chosen["eligible_targets"]:
        raise ValueError("No Station-1 node affects the observed Station-2 system")
    cutoff = int(chosen["cutoff"])
    hidden = tuple((*context.station1_nodes, *station2[:cutoff]))
    observed = tuple(station2[cutoff:])
    directed, bidirected = latent_projection(context.station12_graph, hidden, observed)
    diagnostics = {
        **chosen,
        "split_mode": config.mode,
        "selection_is_fully_nondegenerate": not any(deficits(chosen)),
        "selection_deficits": list(deficits(chosen)),
        "candidate_diagnostics": candidates,
    }
    return GraphSplit(
        station2_topological_order=station2,
        cutoff=cutoff,
        hidden_nodes=hidden,
        observed_nodes=observed,
        directed_edges=directed,
        bidirected_edges=bidirected,
        eligible_intervention_targets=tuple(chosen["eligible_targets"]),
        diagnostics=diagnostics,
    )


def edge_matrices(split: GraphSplit) -> tuple[np.ndarray, np.ndarray]:
    index = {node: i for i, node in enumerate(split.observed_nodes)}
    directed = np.zeros((len(index), len(index)), dtype=int)
    bidirected = np.zeros_like(directed)
    for source, target in split.directed_edges:
        directed[index[target], index[source]] = 1
    for left, right in split.bidirected_edges:
        i, j = index[left], index[right]
        bidirected[i, j] = bidirected[j, i] = 1
    return directed, bidirected
