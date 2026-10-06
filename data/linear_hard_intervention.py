import numpy as np
import utils as ut
from data.bow_free import sample_bow_free_linear_structures


def generate_data(T=4, n=2, xn=2, N=1000, graph_dense=1, confounder_dense=0.5,
                  graph_type="ER", noise_type="gauss", interv_dense=0.5,
                  bow_free_ground_truth=False, **kwargs):
    """
    Hard-intervention DGP: X_t = (I-A)^{-1} Z_t

    For each domain t, a subset I_t of nodes receives a hard intervention:
    the structural equation for node i in I_t is replaced by z_i = epsilon_i
    (parent contribution severed, per the "hard int" mechanism in LinearSCM).
    This is implemented by zeroing out row i of B before computing (I-B_t)^{-1}.

    An extra reference domain with no interventions is always appended last.
    Every latent node is guaranteed to be intervened in at least one non-reference domain.
    """
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
    I_A_inv = np.linalg.inv(np.eye(n) - A)

    n_interv = max(1, int(interv_dense * n))
    print(f'  n_interv={n_interv} (intervention density={interv_dense})')
    if T < n:
        raise ValueError(
            f"A {T} x {n} intervention target matrix cannot have full column "
            "rank because the number of domains is smaller than the number "
            "of nodes."
        )
    if n_interv > n:
        raise ValueError(
            f"interv_dense={interv_dense} requests {n_interv} targets per "
            f"domain, but only {n} nodes exist."
        )
    if n > 1 and n_interv == n:
        raise ValueError(
            "Full column rank is impossible when every domain intervenes on "
            "every node: all rows of the target matrix would be identical."
        )

    max_target_draws = 10_000
    for _ in range(max_target_draws):
        if n_interv == 1 and T == n:
            # Each domain gets exactly one unique node via a random permutation.
            perm = np.random.permutation(n)
            Is = [np.array([perm[t]]) for t in range(T)]
        else:
            Is = [
                np.random.choice(n, size=n_interv, replace=False)
                for _ in range(T)
            ]

            # Ensure every latent node is intervened in at least one domain.
            for j in range(n):
                if not any(j in Is[t] for t in range(T)):
                    t_pick = np.random.randint(T)
                    Is[t_pick] = np.append(Is[t_pick], j)

        intervention_targets = np.zeros((T, n), dtype=int)
        for t, targets in enumerate(Is):
            intervention_targets[t, targets] = 1
        intervention_rank = np.linalg.matrix_rank(intervention_targets)
        if intervention_rank == n:
            break
    else:
        raise RuntimeError(
            f"Could not draw a full-column-rank {T} x {n} intervention "
            f"target matrix (required rank {n}) after {max_target_draws} "
            "attempts. Try changing "
            "--interv-dense."
        )

    # Rows correspond to non-reference domains and columns to latent nodes.
    # The appended reference domain is excluded because its all-zero row does
    # not contribute to the column rank.
    print("  intervention target matrix (domains x nodes):")
    print(intervention_targets)
    print(
        "  full column rank: True "
        f"(rank={intervention_rank}, columns={n})"
    )

    # Reference domain: no interventions
    Is.append(np.array([], dtype=int))
    T_total = T + 1

  
    

    X_list, Z_list = [], []
    for t in range(T_total):
        if noise_type == "gauss":
            E_t = np.random.multivariate_normal(np.zeros(n), sigma @ sigma.T, size=N)
        elif noise_type == "exp":
            E_t = (np.random.exponential(scale=1.0, size=(N, n)) - 1.0) * sigma_vec
        elif noise_type == "gumbel":
            E_t = np.random.gumbel(loc=0.0, scale=1.0, size=(N, n)) * sigma_vec
        else:
            raise ValueError(f"Unsupported noise_type: {noise_type}")

        # Hard intervention: zero out rows of B for intervened nodes
        B_t = B.copy()
        if len(Is[t]) > 0:
            B_t[Is[t], :] = 0.0
            # Match the UMNI-CRL experiment: intervention noise variance is
            # reduced to one quarter (equivalently, standard deviation / 2).
            E_t[:, Is[t]] *= 0.5
        I_B_t_inv = np.linalg.inv(np.eye(n) - B_t)

        Z_t = (I_B_t_inv @ E_t.T).T
        X_t = (I_A_inv @ Z_t.T).T

        Z_list.append(Z_t)
        X_list.append(X_t)

    return X_list, A, B, Is, sigma_vec, Z_list, skeleton


if __name__ == '__main__':
    for i in range(5):
        print('seed', i)
        np.random.seed(i)
        X_list, A, B, Is, sigma_vec, Z_list, skeleton = generate_data(
            T=4, n=6, N=1000, graph_dense=1, confounder_dense=0.5,
            graph_type="ER", noise_type="gauss", interv_dense=0.5,
        )
        print(f'  T_total={len(X_list)}, X shape={X_list[0].shape}')
        for t, targets in enumerate(Is):
            print(f'  domain {t}: intervened nodes = {sorted(targets)}')
