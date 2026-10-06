import numpy as np
import utils as ut


def _build_decoder(skeleton_X, n, n_layers=2, seed=None):
    """
    Piecewise linear decoder f: R^n -> R^n based on skeleton_X.

    Layer 0 weight matrix is (I-A)^{-1} derived from skeleton_X (same structure as
    the linear case).  Each subsequent layer is a random orthogonal matrix.
    Between layers, leaky-ReLU activations with slopes sampled from [0.5, 1.5] are
    applied:  h = max(neg_slope * h, h).
    """
    rng = np.random.default_rng(seed)

    A = ut.simulate_parameter(skeleton_X)
    W0 = np.linalg.inv(np.eye(n) - A)

    W_layers = [W0]
    for _ in range(n_layers - 1):
        W = rng.normal(size=(n, n))
        W = np.linalg.qr(W.T)[0].T  # orthogonal
        W_layers.append(W)

    neg_slopes = rng.uniform(0.5, 1.5, n_layers)

    def decoder(Z):
        """Z: (N, n) -> X: (N, n)"""
        h = Z @ W_layers[0].T
        for l in range(n_layers - 1):
            h = np.maximum(neg_slopes[l + 1] * h, h)
            h = h @ W_layers[l + 1].T
        return h

    return decoder, A


def generate_data(T=4, n=2, xn=2, N=1000, graph_dense=1, confounder_dense=0.5,
                  graph_type="ER", noise_type="gauss", mask_dense=0.5, n_layers=2,
                  **kwargs):
    """
    Binary-mask DGP with piecewise linear mixing: X_t = f(diag(M_t) Z_t)

    f is a leaky-ReLU network whose first layer weight matrix is (I-A)^{-1} derived
    from skeleton_X, followed by (n_layers-1) random orthogonal layers with leaky-ReLU
    activations between them.  Everything else (latent DAG, masks, noise) is identical
    to linear_mask.py.
    """
    sigma_vec = np.random.uniform(0.2, 0.5, size=n)
    sigma = np.diag(sigma_vec)

    skeleton = ut.simulate_dag(n, int(n * confounder_dense), graph_type)
    B = ut.simulate_parameter(skeleton)
    I_B_inv = np.linalg.inv(np.eye(n) - B)

    skeleton_X = ut.simulate_dag(n, int(n * graph_dense), graph_type)
    decoder, A = _build_decoder(skeleton_X, n, n_layers=n_layers)

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
        X_t = decoder(Z_t)

        Z_list.append(Z_t)
        X_list.append(X_t)

    return X_list, A, B, Ms, sigma_vec, Z_list, skeleton
