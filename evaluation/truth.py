"""Build graph-specific evaluation references from DGP ground truth."""

from __future__ import annotations

import numpy as np

import utils as ut
from data.types import GeneratedDataset
from graphs import ADMG, EndpointGraph, EndpointMark, GraphKind


def admg_reference_from_dataset(dataset: GeneratedDataset) -> ADMG:
    """Return the observed ADMG implied by a generated dataset.

    In the auxiliary DGP, the pooled marginal distribution treats the shared
    auxiliary variable as a common cause of every latent component.  Its
    bidirected graph is therefore complete even though the conditional
    covariance within each auxiliary-value group is diagonal.
    """
    weights = dataset.truth.observed_adjacency
    directed = (np.abs(weights) > 0).astype(int)
    if dataset.metadata.get("dgp") == "auxillary":
        bidirected = np.ones(directed.shape, dtype=int)
        np.fill_diagonal(bidirected, 0)
    else:
        _, bidirected = ut.get_admg_of_X(
            dataset.truth.latent_skeleton,
            weights,
        )
    return ADMG(
        directed=directed,
        bidirected=bidirected,
        directed_weights=weights,
    )


def pag_reference_from_dataset(dataset: GeneratedDataset) -> EndpointGraph:
    """Represent the true observed ADMG with PAG endpoint marks.

    This is a fully oriented reference graph, not an equivalence-class oracle.
    If the DGP contains a bow on a node pair, the directed edge takes priority
    because a single PAG edge cannot encode directed and bidirected edges at
    the same time.
    """
    admg = admg_reference_from_dataset(dataset)
    directed = admg.directed
    bidirected = admg.bidirected
    n = directed.shape[0]
    endpoints = np.zeros((n, n), dtype=np.int8)

    for i in range(n):
        for j in range(i + 1, n):
            # Repository adjacency convention: directed[i, j] means j -> i.
            if directed[i, j]:
                endpoints[i, j] = EndpointMark.ARROW
                endpoints[j, i] = EndpointMark.TAIL
            elif directed[j, i]:
                endpoints[i, j] = EndpointMark.TAIL
                endpoints[j, i] = EndpointMark.ARROW
            elif bidirected[i, j]:
                endpoints[i, j] = EndpointMark.ARROW
                endpoints[j, i] = EndpointMark.ARROW

    return EndpointGraph(endpoints=endpoints, graph_kind=GraphKind.PAG)
