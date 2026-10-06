import os
import numpy as np
import torch
import utils as ut
import igraph as ig
from data.bow_free import sample_bow_free_linear_structures


def generate_data(T=4, n=2, xn=2, N=1000, graph_dense=1, confounder_dense=0.5, graph_type="ER", noise_type="gauss", test_vio=False, bow_free_ground_truth=False, **_):
    """Generate ``T`` scaled domains followed by an unscaled base domain.

    The base domain uses an all-ones scaling vector and is appended last, matching
    the reference-domain convention used by the mask and hard-intervention DGPs.
    """

    Ds = [np.random.uniform(0.2, 1, size=n) for _ in range(T)]
    Ds.append(np.ones(n))

    sigma_vec = np.random.uniform(0.2, 0.5, size=n)
    sigma = np.diag(sigma_vec)

    latent_edges = int(n * confounder_dense)
    observed_edges = int(n * graph_dense)
    if bow_free_ground_truth:
        skeleton, B, skeleton_X, A = sample_bow_free_linear_structures(
            n, latent_edges, observed_edges, graph_type
        )
    else:
        skeleton = ut.simulate_dag(n, latent_edges, graph_type)
        B = ut.simulate_parameter(skeleton)
        skeleton_X = ut.simulate_dag(n, observed_edges, graph_type)
        A = ut.simulate_parameter(skeleton_X)
    I_B_inv = np.linalg.inv(np.eye(n) - B)
    I_A_inv = np.linalg.inv(np.eye(n) - A)

    X_list, Z_list = [], []
    for t in range(T + 1):
        if noise_type == "gauss":
            E_t = np.random.multivariate_normal(np.zeros(n), sigma @ sigma.T, size=N)
        elif noise_type == "exp":
            E_t = (np.random.exponential(scale=1.0, size=(N, n)) - 1.0) * sigma_vec
        elif noise_type == "gumbel":
            E_t = np.random.gumbel(loc=0.0, scale=1.0, size=(N, n)) * sigma_vec
        else:
            raise ValueError(f"Unsupported noise_type: {noise_type}")

        Z_t = (I_B_inv @ E_t.T).T
        X_t = (I_A_inv @ np.diag(Ds[t]) @ Z_t.T).T

        Z_list.append(Z_t)
        X_list.append(X_t)

    if test_vio:
        # Stack the scaled domains and the base domain: ((T + 1) * N, n).
        Z = np.vstack(Z_list)

        theo_var = np.diag(I_B_inv @ np.diag(sigma_vec ** 2) @ I_B_inv.T)
        # check if variance increases along each causal edge
        g = ig.Graph.Adjacency((skeleton.T > 0).tolist())
        violations = []
        for edge in g.get_edgelist():
            src, dst = edge  # src -> dst
            if theo_var[dst] < theo_var[src]:
                violations.append((src, dst, theo_var[src], theo_var[dst]))
        total_edges = len(g.get_edgelist())
        proportion = len(violations) / total_edges if total_edges > 0 else 0.0
        print(f'variance violations: {len(violations)}/{total_edges} ({proportion:.2%})')

    return X_list, A, B, Ds, sigma_vec, Z_list, skeleton


if __name__ == '__main__':
    for i in range(10):
        print('seed ', i)
        generate_data(T=2, n=10, xn=10, N=2000, graph_dense=2, graph_type="ER", noise_type="gauss", test_vio=True)
