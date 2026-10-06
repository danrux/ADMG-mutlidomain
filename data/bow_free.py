"""Shared sampling helpers for bow-free linear ADMG ground truth."""

from __future__ import annotations

import numpy as np

import utils as ut


def simulate_bow_free_er_dag(
    n: int,
    edge_count: int,
    bidirected: np.ndarray,
) -> np.ndarray:
    """Sample an ER-style DAG with no adjacency on a bidirected pair."""
    order = np.random.permutation(n)
    eligible = [
        (order[child_position], order[parent_position])
        for parent_position in range(n)
        for child_position in range(parent_position + 1, n)
        if not bidirected[order[child_position], order[parent_position]]
    ]
    if edge_count > len(eligible):
        raise ValueError(
            "Cannot generate a bow-free observed DAG with "
            f"{edge_count} edges: only {len(eligible)} eligible node pairs remain."
        )
    skeleton = np.zeros((n, n), dtype=int)
    if edge_count:
        selected = np.random.choice(len(eligible), size=edge_count, replace=False)
        for index in np.atleast_1d(selected):
            child, parent = eligible[int(index)]
            skeleton[child, parent] = 1
    return skeleton


def sample_bow_free_linear_structures(
    n: int,
    latent_edge_count: int,
    observed_edge_count: int,
    graph_type: str,
    *,
    max_attempts: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample latent/observed DAGs whose induced ADMG contains no bows.

    The returned matrices are ``latent_skeleton, latent_weights,
    observed_skeleton, observed_weights`` and use the repository's
    ``[child, parent]`` convention.
    """
    if graph_type != "ER":
        raise ValueError(
            "bow_free_ground_truth currently supports only graph_type='ER'."
        )
    for _ in range(max_attempts):
        latent_skeleton = ut.simulate_dag(n, latent_edge_count, graph_type)
        latent_weights = ut.simulate_parameter(latent_skeleton)
        _, bidirected = ut.get_admg_of_X(
            latent_skeleton,
            np.zeros((n, n)),
        )
        eligible_pairs = (
            n * (n - 1) // 2
            - int(np.count_nonzero(np.triu(bidirected, 1)))
        )
        if observed_edge_count <= eligible_pairs:
            observed_skeleton = simulate_bow_free_er_dag(
                n,
                observed_edge_count,
                bidirected,
            )
            observed_weights = ut.simulate_parameter(observed_skeleton)
            return (
                latent_skeleton,
                latent_weights,
                observed_skeleton,
                observed_weights,
            )
    raise RuntimeError(
        "Could not sample a latent DAG compatible with the requested bow-free "
        f"observed-edge and confounder densities after {max_attempts} attempts."
    )
