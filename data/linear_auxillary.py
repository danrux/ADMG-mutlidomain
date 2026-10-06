"""Linear DGP with an observed auxiliary variable defining five groups.

The latent variables are generated before the observations:

    Z_g ~ N(mu(u_g), Sigma(u_g))
    X_g = (I - A)^{-1} Z_g

Both Gaussian parameters vary smoothly with the group-specific constant u_g.
"""

import numpy as np

import utils as ut


def generate_data(
    T=5,
    n=2,
    xn=2,
    N=1000,
    graph_dense=1,
    confounder_dense=0.5,
    graph_type="ER",
    noise_type="gauss",
    u_values=None,
    covariance_jitter=1e-6,
    **_,
):
    """Generate five auxiliary-variable groups of ``Z`` and then linear ``X``.

    Parameters
    ----------
    T : int
        Retained for compatibility with the other DGPs. This process always
        generates five groups; pass ``u_values`` to choose their constants.
    n : int
        Dimension of both Z and X.
    N : int
        Number of observations in each group.
    u_values : array-like of length 5, optional
        Constant auxiliary-variable values. Defaults to
        ``[-2, -1, 0, 1, 2]``.
    covariance_jitter : float
        Positive diagonal term used to keep every covariance nonsingular.

    Returns
    -------
    X_list, A, B, us, sigma_vec, Z_list, skeleton
        The return layout matches the existing linear DGPs. Since Z is sampled
        directly rather than from a latent SCM, ``B`` and ``skeleton`` are
        zero matrices. The graph-truth adapter represents the shared auxiliary
        variable as a common cause and therefore gives the pooled marginal
        ADMG a complete bidirected graph. ``us`` occupies the usual
        domain-parameter slot, while ``sigma_vec`` contains the marginal
        standard deviations at u=0.
    """
    if noise_type != "gauss":
        raise ValueError("linear_auxillary supports only noise_type='gauss'")
    if n < 1 or N < 1:
        raise ValueError("n and N must be positive")
    if covariance_jitter <= 0:
        raise ValueError("covariance_jitter must be positive")

    us = np.asarray(
        [-2.0, -1.0, 0.0, 1.0, 2.0] if u_values is None else u_values,
        dtype=float,
    )
    if us.shape != (5,):
        raise ValueError("u_values must contain exactly five constants")

    # Shared random functions of u. The diagonal factor construction guarantees
    # that Sigma(u) is diagonal and positive definite for every real u:
    #   mu(u) = mean_intercept + u * mean_slope
    #   Sigma(u) = L(u)L(u)^T + jitter*I,
    #   L(u) = L0 + u*L1.
    mean_intercept = np.random.normal(0.0, 0.5, size=n)
    mean_slope = np.random.normal(0.0, 0.5, size=n)
    L0 = np.diag(np.random.uniform(0.2, 0.5, size=n))
    L1 = np.diag(np.random.normal(0.0, 0.08, size=n))

    skeleton_X = ut.simulate_dag(n, int(n * graph_dense), graph_type)
    A = ut.simulate_parameter(skeleton_X)
    I_A_inv = np.linalg.inv(np.eye(n) - A)

    B = np.zeros((n, n))
    skeleton = np.zeros((n, n))
    sigma_vec = np.sqrt(np.diag(L0 @ L0.T + covariance_jitter * np.eye(n)))

    X_list, Z_list = [], []
    for u in us:
        mean = mean_intercept + u * mean_slope
        factor = L0 + u * L1
        covariance = factor @ factor.T + covariance_jitter * np.eye(n)

        Z = np.random.multivariate_normal(mean, covariance, size=N)
        X = (I_A_inv @ Z.T).T

        Z_list.append(Z)
        X_list.append(X)

    return X_list, A, B, us, sigma_vec, Z_list, skeleton


if __name__ == "__main__":
    np.random.seed(0)
    generated = generate_data(n=4, N=1000)
    X_list, _, _, us, _, Z_list, _ = generated
    for u, Z, X in zip(us, Z_list, X_list):
        print(f"u={u: .1f}: Z shape={Z.shape}, X shape={X.shape}")
