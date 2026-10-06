"""PAG-based endpoint recovery for DCD ADMG estimates."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
from typing import Any

import numpy as np

from graphs import ADMG, EndpointGraph, EndpointMark, GraphKind


def _rates(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    tp = int(np.sum(target & prediction))
    fp = int(np.sum(~target & prediction))
    target_count = int(np.sum(target))
    predicted_count = int(np.sum(prediction))
    return {
        "tp": tp,
        "fp": fp,
        "target": target_count,
        "predicted": predicted_count,
        "tpr": float(tp / target_count) if target_count else 0.0,
        "fdr": float(fp / predicted_count) if predicted_count else 0.0,
    }


def _ancestor_matrix(directed: np.ndarray) -> np.ndarray:
    """Return ``ancestors[descendant, ancestor]`` including self ancestry."""
    n = directed.shape[0]
    ancestors = np.eye(n, dtype=bool) | (directed != 0)
    for intermediate in range(n):
        ancestors |= ancestors[:, intermediate, None] & ancestors[None, intermediate, :]
    return ancestors


def _has_inducing_path(
    source: int,
    target: int,
    directed: np.ndarray,
    bidirected: np.ndarray,
    ancestors: np.ndarray,
) -> bool:
    """Match the inducing-path search used by the original DCD converter."""
    if directed[source, target] or directed[target, source] or bidirected[source, target]:
        return True

    ancestors_of_endpoints = ancestors[source] | ancestors[target]
    siblings = bidirected != 0
    children = directed[:, source] != 0
    stack = list(np.flatnonzero((siblings[source] | children) & ancestors_of_endpoints))
    visited: set[int] = set()

    while stack:
        if target in stack:
            return True
        # The target is a parent of any currently reachable collider.
        if any(directed[node, target] for node in stack):
            return True
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(
            int(sibling)
            for sibling in np.flatnonzero(siblings[node] & ancestors_of_endpoints)
            if int(sibling) not in visited
        )
    return False


def maximal_ancestral_projection(graph: ADMG) -> ADMG:
    """Project an ADMG to the MAG used by the original DCD PAG converter."""
    directed = np.asarray(graph.directed) != 0
    bidirected = (np.asarray(graph.bidirected) != 0)
    bidirected = bidirected | bidirected.T
    n = directed.shape[0]
    ancestors = _ancestor_matrix(directed)
    mag_directed = np.zeros((n, n), dtype=np.int8)
    mag_bidirected = np.zeros((n, n), dtype=np.int8)

    for first in range(n):
        for second in range(first + 1, n):
            if not _has_inducing_path(first, second, directed, bidirected, ancestors):
                continue
            if ancestors[second, first] and first != second:
                # first -> second; repository convention is [child, parent].
                mag_directed[second, first] = 1
            elif ancestors[first, second] and first != second:
                mag_directed[first, second] = 1
            else:
                mag_bidirected[first, second] = mag_bidirected[second, first] = 1

    return ADMG(directed=mag_directed, bidirected=mag_bidirected)


def _mag_to_latent_dag(mag: ADMG):
    """Create a DAG whose observed latent projection is ``mag``."""
    try:
        from causallearn.graph.Dag import Dag
        from causallearn.graph.GraphNode import GraphNode
    except ImportError as exc:
        raise ImportError(
            "PAG evaluation for DCD requires causal-learn. Install "
            "requirements.txt."
        ) from exc

    n = mag.directed.shape[0]
    observed = [GraphNode(f"X{index}") for index in range(n)]
    bidirected_pairs = [
        (first, second)
        for first in range(n)
        for second in range(first + 1, n)
        if mag.bidirected[first, second] or mag.bidirected[second, first]
    ]
    latent = [GraphNode(f"L{first}_{second}") for first, second in bidirected_pairs]
    dag = Dag(observed + latent)

    for child, parent in np.argwhere(np.asarray(mag.directed) != 0):
        dag.add_directed_edge(observed[int(parent)], observed[int(child)])
    for latent_node, (first, second) in zip(latent, bidirected_pairs):
        dag.add_directed_edge(latent_node, observed[first])
        dag.add_directed_edge(latent_node, observed[second])
    return dag, observed, latent


def admg_to_pag(graph: ADMG) -> EndpointGraph:
    """Convert an ADMG to the PAG of its maximal ancestral projection."""
    from causallearn.graph.Endpoint import Endpoint
    from causallearn.utils.DAG2PAG import dag2pag

    mag = maximal_ancestral_projection(graph)
    dag, observed, latent = _mag_to_latent_dag(mag)
    # causal-learn currently prints a blank line and may print rule diagnostics.
    with redirect_stdout(io.StringIO()):
        native_pag = dag2pag(dag, islatent=latent, isselection=[])

    node_indices = {
        node.get_name(): index for index, node in enumerate(native_pag.get_nodes())
    }
    endpoints = np.zeros((len(observed), len(observed)), dtype=np.int8)
    mark_map = {
        Endpoint.NULL.value: EndpointMark.NONE,
        Endpoint.TAIL.value: EndpointMark.TAIL,
        Endpoint.ARROW.value: EndpointMark.ARROW,
        Endpoint.CIRCLE.value: EndpointMark.CIRCLE,
    }
    for first in range(len(observed)):
        for second in range(len(observed)):
            native_mark = int(
                native_pag.graph[
                    node_indices[f"X{first}"],
                    node_indices[f"X{second}"],
                ]
            )
            if native_mark not in mark_map:
                raise ValueError(f"Unsupported combined PAG endpoint value {native_mark}.")
            endpoints[first, second] = mark_map[native_mark].value
    return EndpointGraph(endpoints=endpoints, graph_kind=GraphKind.PAG)


def evaluate_dcd_pag_recovery(
    truth: ADMG,
    estimate: ADMG,
    *,
    truth_pag: EndpointGraph | None = None,
    estimate_pag: EndpointGraph | None = None,
) -> dict[str, Any]:
    """Use exact ADMG skeletons and PAG arrowheads/tails as in DCD Table 2."""
    upper = np.triu(np.ones(truth.directed.shape, dtype=bool), k=1)
    true_skeleton = (
        (truth.directed != 0)
        | (truth.directed.T != 0)
        | (truth.bidirected != 0)
        | (truth.bidirected.T != 0)
    )[upper]
    estimated_skeleton = (
        (estimate.directed != 0)
        | (estimate.directed.T != 0)
        | (estimate.bidirected != 0)
        | (estimate.bidirected.T != 0)
    )[upper]

    truth_pag = admg_to_pag(truth) if truth_pag is None else truth_pag
    estimate_pag = admg_to_pag(estimate) if estimate_pag is None else estimate_pag
    off_diagonal = ~np.eye(truth.directed.shape[0], dtype=bool)
    return {
        "skeleton": _rates(true_skeleton, estimated_skeleton),
        "arrowhead": _rates(
            (truth_pag.endpoints == EndpointMark.ARROW)[off_diagonal],
            (estimate_pag.endpoints == EndpointMark.ARROW)[off_diagonal],
        ),
        "tail": _rates(
            (truth_pag.endpoints == EndpointMark.TAIL)[off_diagonal],
            (estimate_pag.endpoints == EndpointMark.TAIL)[off_diagonal],
        ),
    }


__all__ = ["admg_to_pag", "evaluate_dcd_pag_recovery", "maximal_ancestral_projection"]
