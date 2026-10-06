import numpy as np
import utils as ut
from data.bow_free import sample_bow_free_linear_structures


def generate_data(T=4, n=2, xn=2, N=1000, graph_dense=1, confounder_dense=0.5,
                  graph_type="ER", noise_type="gauss", mask_dense=0.5,
                  bow_free_ground_truth=False, **kwargs):
    """
    Binary-mask DGP: X_t = (I-A)^{-1} diag(M_t) Z_t
    where M_t in {0,1}^n is a domain-specific binary mask with mask_dense * n active dims.

    Each latent dimension is guaranteed to be active in at least one domain.
    Returns Ms (list of binary mask vectors) in place of Ds for the 4th return value.
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
    I_B_inv = np.linalg.inv(np.eye(n) - B)
    I_A_inv = np.linalg.inv(np.eye(n) - A)
    

    n_active = max(1, int(mask_dense * n))
    Ms = []
    for t in range(T):
        mask = np.zeros(n)
        mask[np.random.choice(n, size=n_active, replace=False)] = 1.0
        Ms.append(mask)

    # Ensure every latent dimension is active in at least one domain
    for j in range(n):
        if not any(Ms[t][j] for t in range(T)):
            Ms[np.random.randint(T)][j] = 1.0

    # Always append one extra domain with all-ones mask (all dims active)
    Ms.append(np.ones(n))
    
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
            

        Z_t = (np.diag(Ms[t]) @ (I_B_inv @ E_t.T)).T

        
        '''
        var_Z_t = np.sum((I_B_inv ** 2) * (sigma_vec ** 2), axis=1) * Ms[t]
        Cov_Z_t = np.diag(Ms[t]) @ I_B_inv @ np.diag(sigma_vec ** 2) @ I_B_inv.T @ np.diag(Ms[t])
        var_X_t = np.diag(I_A_inv @ Cov_Z_t @ I_A_inv.T)
        print(f"Domain {t}: Var(Z_t) = {var_Z_t}")
        print(f"Domain {t}: Var(X_t) = {var_X_t}")
        '''
        
        X_t = (I_A_inv @ Z_t.T).T

        Z_list.append(Z_t)
        X_list.append(X_t)

    return X_list, A, B, Ms, sigma_vec, Z_list, skeleton
