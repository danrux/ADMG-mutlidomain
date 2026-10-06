"""Differentiable causal discovery for linear Gaussian ADMG models.

This is an independent implementation of the differentiable constraints and
augmented-Lagrangian procedure in Bhattacharya et al. (2021).  ``W_directed``
uses the paper/reference orientation: ``W_directed[parent, child]``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import autograd.numpy as anp
from autograd import grad
import numpy as np
from scipy.optimize import minimize
from scipy.special import comb


ArrayPenalty = Callable[[anp.ndarray, anp.ndarray], anp.ndarray]


@dataclass(frozen=True)
class DCDResult:
    directed_weights: np.ndarray
    error_covariance: np.ndarray
    directed: np.ndarray
    bidirected: np.ndarray
    bic: float
    converged: bool
    restart: int
    constraint: float
    optimizer_failures: int


def cycle_constraint(weights: anp.ndarray) -> anp.ndarray:
    """Polynomial acyclicity constraint from the DCD paper."""
    dimension = len(weights)
    matrix = anp.eye(dimension) + weights * weights / dimension
    power = anp.eye(dimension)
    for _ in range(dimension):
        power = anp.dot(power, matrix)
    return anp.trace(power) - dimension


def bow_constraint(directed: anp.ndarray, bidirected: anp.ndarray) -> anp.ndarray:
    """Zero exactly when no node pair has both edge types."""
    dimension = len(directed)
    return anp.sum((directed * directed / dimension) * (bidirected * bidirected / dimension))


def ancestrality_constraint(
    directed: anp.ndarray,
    bidirected: anp.ndarray,
) -> anp.ndarray:
    """Penalize bidirected edges between directed ancestors/descendants."""
    dimension = len(directed)
    directed_squared = directed * directed
    bidirected_squared = bidirected * bidirected
    power = anp.eye(dimension)
    reachability = anp.eye(dimension)
    for degree in range(1, dimension):
        power = anp.dot(power, directed_squared)
        reachability = reachability + power / math.factorial(degree)
    return anp.sum(reachability * bidirected_squared)


def arid_constraint(
    directed: anp.ndarray,
    bidirected: anp.ndarray,
    *,
    directed_scale: float = 1.0,
    bidirected_scale: float = 2.0,
    sharpness: float = float(np.log(5000.0)),
) -> anp.ndarray:
    """Differentiable primal-fixing constraint characterizing arid ADMGs."""
    dimension = len(directed)
    total = 0.0
    for root in range(dimension):
        root_mask = anp.array([1.0 if index == root else 0.0 for index in range(dimension)])
        directed_fixed = directed * directed
        bidirected_fixed = bidirected * bidirected

        for _ in range(dimension - 1):
            power = anp.eye(dimension)
            closure = anp.eye(dimension)
            for degree in range(1, dimension):
                power = anp.dot(power, bidirected_fixed)
                closure = closure + comb(dimension, degree) * (bidirected_scale**degree) * power
            fixability_matrix = closure * directed_fixed
            argument = anp.clip(
                sharpness * (anp.mean(fixability_matrix, axis=1) + root_mask),
                0.0,
                4.0,
            )
            fixability = anp.tanh(argument / 2.0)
            row_mask = anp.vstack([fixability for _ in range(dimension)])
            directed_fixed = directed_fixed * row_mask
            bidirected_fixed = bidirected_fixed * row_mask * row_mask.T

        directed_power = anp.eye(dimension)
        bidirected_power = anp.eye(dimension)
        directed_closure = anp.eye(dimension)
        bidirected_closure = anp.eye(dimension)
        for degree in range(1, dimension):
            directed_power = anp.dot(directed_power, directed_fixed)
            bidirected_power = anp.dot(bidirected_power, bidirected_fixed)
            directed_closure = directed_closure + (
                1.0 / math.factorial(degree)
                + comb(dimension, degree) * directed_scale**degree
            ) * directed_power
            bidirected_closure = bidirected_closure + (
                1.0 / math.factorial(degree)
                + comb(dimension, degree) * bidirected_scale**degree
            ) * bidirected_power
        total = total + anp.sum(
            directed_closure[:, root] * bidirected_closure[:, root]
        ) - 1.0
    return total


PENALTIES: dict[str, ArrayPenalty] = {
    "bowfree": bow_constraint,
    "ancestral": ancestrality_constraint,
    "arid": arid_constraint,
}


def _symmetrize(raw_bidirected: np.ndarray) -> np.ndarray:
    return raw_bidirected + raw_bidirected.T


def _pseudo_variables(
    observations: np.ndarray,
    directed: np.ndarray,
    error_covariance: np.ndarray,
    *,
    jitter: float,
) -> list[np.ndarray]:
    dimension = observations.shape[1]
    residuals = observations - observations @ directed
    pseudo_variables = []
    for vertex in range(dimension):
        retained = [index for index in range(dimension) if index != vertex]
        covariance = error_covariance[np.ix_(retained, retained)]
        covariance = covariance + jitter * np.eye(dimension - 1)
        try:
            solved = np.linalg.solve(covariance, residuals[:, retained].T).T
        except np.linalg.LinAlgError:
            solved = (np.linalg.pinv(covariance) @ residuals[:, retained].T).T
        pseudo_variables.append(np.insert(solved, vertex, 0.0, axis=1))
    return pseudo_variables


def _bounds(dimension: int, coefficient_bound: float) -> list[tuple[float, float]]:
    bounds = []
    for parent in range(dimension):
        for child in range(dimension):
            bounds.append(
                (0.0, 0.0) if parent == child else (-coefficient_bound, coefficient_bound)
            )
    for row in range(dimension):
        for column in range(dimension):
            bounds.append(
                (0.0, 0.0)
                if row <= column
                else (-coefficient_bound, coefficient_bound)
            )
    return bounds


def _objective_factory(
    observations: np.ndarray,
    pseudo_variables: list[np.ndarray],
    dimension: int,
    sample_size: int,
    rho: float,
    alpha: float,
    sparsity: float,
    penalty: ArrayPenalty,
):
    observations_anp = anp.asarray(observations)
    pseudo_anp = [anp.asarray(value) for value in pseudo_variables]

    def objective(parameters):
        directed = anp.reshape(parameters[: dimension * dimension], (dimension, dimension))
        raw_bidirected = anp.reshape(parameters[dimension * dimension :], (dimension, dimension))
        bidirected = raw_bidirected + raw_bidirected.T
        fit_loss = 0.0
        for vertex in range(dimension):
            residual = (
                observations_anp[:, vertex]
                - anp.dot(observations_anp, directed[:, vertex])
                - anp.dot(pseudo_anp[vertex], bidirected[:, vertex])
            )
            fit_loss = fit_loss + 0.5 * anp.sum(residual * residual) / sample_size
        constraint = cycle_constraint(directed) + penalty(directed, bidirected)
        augmented = 0.5 * rho * constraint * constraint + alpha * constraint
        approximate_l0 = anp.sum(
            anp.tanh(0.5 * anp.log(float(sample_size)) * anp.abs(parameters))
        )
        return fit_loss + augmented + sparsity * approximate_l0

    return objective


def _graph_masks(
    directed: np.ndarray,
    bidirected: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    directed_mask = np.abs(directed) > threshold
    np.fill_diagonal(directed_mask, False)
    bidirected_mask = np.abs(bidirected) > threshold
    bidirected_mask = np.logical_or(bidirected_mask, bidirected_mask.T)
    np.fill_diagonal(bidirected_mask, False)
    return directed_mask, bidirected_mask


def refit_admg(
    observations: np.ndarray,
    directed_mask: np.ndarray,
    bidirected_mask: np.ndarray,
    *,
    tolerance: float = 1e-4,
    max_iterations: int = 100,
    jitter: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """RICF-style parameter refit and the reference implementation's BIC."""
    observations = np.asarray(observations, dtype=float)
    sample_size, dimension = observations.shape
    sample_covariance = np.atleast_2d(np.cov(observations, rowvar=False))
    directed = np.zeros((dimension, dimension))
    off_diagonal = sample_covariance.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    diagonal = np.diag(np.maximum(np.diag(sample_covariance), jitter))

    for _ in range(max_iterations):
        old_directed = directed.copy()
        old_error = off_diagonal + diagonal
        for vertex in range(dimension):
            error_covariance = off_diagonal + diagonal
            pseudo = _pseudo_variables(
                observations, directed, error_covariance, jitter=jitter
            )[vertex]
            parents = np.flatnonzero(directed_mask[:, vertex]).tolist()
            siblings = np.flatnonzero(bidirected_mask[:, vertex]).tolist()
            columns = []
            if parents:
                columns.append(observations[:, parents])
            if siblings:
                columns.append(pseudo[:, siblings])
            directed[:, vertex] = 0.0
            off_diagonal[vertex, :] = 0.0
            off_diagonal[:, vertex] = 0.0
            if columns:
                design = np.column_stack(columns)
                coefficients = np.linalg.lstsq(
                    design, observations[:, vertex], rcond=None
                )[0]
                split = len(parents)
                if parents:
                    directed[parents, vertex] = coefficients[:split]
                if siblings:
                    off_diagonal[vertex, siblings] = coefficients[split:]
                    off_diagonal[siblings, vertex] = coefficients[split:]
            structural_residual = observations[:, vertex] - observations @ directed[:, vertex]
            diagonal[vertex, vertex] = max(float(np.var(structural_residual)), jitter)

        delta = np.sum(np.abs(directed - old_directed)) + np.sum(
            np.abs((off_diagonal + diagonal) - old_error)
        )
        if delta < tolerance:
            break

    structural = directed.T
    try:
        mixing = np.linalg.inv(np.eye(dimension) - structural)
        model_covariance = mixing @ (off_diagonal + diagonal) @ mixing.T
        sign, log_determinant = np.linalg.slogdet(model_covariance)
        if sign <= 0 or not np.isfinite(log_determinant):
            raise np.linalg.LinAlgError("Non-positive model covariance")
        likelihood = 0.5 * sample_size * (
            log_determinant
            + np.trace(np.linalg.solve(model_covariance, sample_covariance))
        )
        edge_count = int(directed_mask.sum() + np.triu(bidirected_mask, 1).sum())
        bic = 2.0 * likelihood + np.log(sample_size) * edge_count
    except np.linalg.LinAlgError:
        bic = np.inf
    return directed, off_diagonal + diagonal, float(bic)


def _one_restart(
    observations: np.ndarray,
    *,
    admg_class: str,
    sparsity: float,
    threshold: float,
    max_iterations: int,
    constraint_tolerance: float,
    rho_max: float,
    ricf_increment: int,
    ricf_tolerance: float,
    optimizer_max_iterations: int,
    coefficient_bound: float,
    jitter: float,
    rng: np.random.Generator,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, bool, float, int]:
    sample_size, dimension = observations.shape
    penalty = PENALTIES[admg_class]
    bounds = _bounds(dimension, coefficient_bound)
    directed_hat = rng.uniform(-0.5, 0.5, size=(dimension, dimension))
    np.fill_diagonal(directed_hat, 0.0)
    raw_bidirected_hat = np.zeros((dimension, dimension))
    lower = np.tril_indices(dimension, k=-1)
    raw_bidirected_hat[lower] = rng.uniform(-0.05, 0.05, size=len(lower[0]))
    sample_covariance = np.atleast_2d(np.cov(observations, rowvar=False))
    diagonal_hat = np.diag(np.maximum(np.diag(sample_covariance), jitter))

    rho, alpha, previous_constraint = 1.0, 0.0, np.inf
    ricf_iterations = 1
    converged = False
    optimizer_failures = 0
    final_constraint = np.inf

    for outer_iteration in range(max_iterations):
        while rho < rho_max:
            directed_new = directed_hat.copy()
            raw_bidirected_new = raw_bidirected_hat.copy()
            diagonal_new = diagonal_hat.copy()
            for _ in range(ricf_iterations):
                directed_old = directed_new.copy()
                error_old = _symmetrize(raw_bidirected_new) + diagonal_new
                pseudo = _pseudo_variables(
                    observations,
                    directed_new,
                    error_old,
                    jitter=jitter,
                )
                initial = np.concatenate(
                    [directed_new.ravel(), raw_bidirected_new.ravel()]
                )
                objective = _objective_factory(
                    observations,
                    pseudo,
                    dimension,
                    sample_size,
                    rho,
                    alpha,
                    sparsity,
                    penalty,
                )
                gradient = grad(objective)
                solution = minimize(
                    objective,
                    initial,
                    method="L-BFGS-B",
                    jac=gradient,
                    bounds=bounds,
                    options={"disp": False, "maxiter": optimizer_max_iterations},
                )
                if not solution.success:
                    optimizer_failures += 1
                if not np.isfinite(solution.x).all():
                    optimizer_failures += 1
                    break
                directed_new = solution.x[: dimension * dimension].reshape(
                    dimension, dimension
                )
                raw_bidirected_new = solution.x[dimension * dimension :].reshape(
                    dimension, dimension
                )
                np.fill_diagonal(directed_new, 0.0)
                for vertex in range(dimension):
                    residual = observations[:, vertex] - observations @ directed_new[:, vertex]
                    diagonal_new[vertex, vertex] = max(float(np.var(residual)), jitter)
                error_new = _symmetrize(raw_bidirected_new) + diagonal_new
                delta = np.sum(np.abs(directed_old - directed_new)) + np.sum(
                    np.abs(error_old - error_new)
                )
                if delta < ricf_tolerance:
                    converged = True
                    break

            bidirected_new = _symmetrize(raw_bidirected_new)
            final_constraint = float(
                cycle_constraint(directed_new) + penalty(directed_new, bidirected_new)
            )
            if verbose:
                print(
                    f"DCD iter={outer_iteration} rho={rho:.3g} "
                    f"constraint={final_constraint:.3g}"
                )
            if final_constraint < 0.25 * previous_constraint:
                break
            rho *= 10.0

        directed_hat = directed_new.copy()
        raw_bidirected_hat = raw_bidirected_new.copy()
        diagonal_hat = diagonal_new.copy()
        previous_constraint = final_constraint
        alpha += rho * final_constraint
        ricf_iterations += ricf_increment
        if final_constraint <= constraint_tolerance or rho >= rho_max:
            break

    bidirected_hat = _symmetrize(raw_bidirected_hat)
    directed_mask, bidirected_mask = _graph_masks(
        directed_hat, bidirected_hat, threshold
    )
    return directed_mask, bidirected_mask, converged, final_constraint, optimizer_failures


def fit_dcd(
    observations: np.ndarray,
    *,
    admg_class: str = "bowfree",
    sparsity: float = 0.05,
    threshold: float = 0.05,
    max_iterations: int = 100,
    constraint_tolerance: float = 1e-8,
    rho_max: float = 1e16,
    num_restarts: int = 5,
    ricf_increment: int = 1,
    ricf_tolerance: float = 1e-4,
    ricf_refit_iterations: int = 100,
    optimizer_max_iterations: int = 15000,
    coefficient_bound: float = 4.0,
    jitter: float = 1e-8,
    random_state: int | None = None,
    verbose: bool = False,
) -> DCDResult:
    """Fit DCD and select random restarts with the paper implementation's BIC."""
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 2 or observations.shape[0] < 2 or observations.shape[1] < 2:
        raise ValueError("DCD requires an n-by-p matrix with n >= 2 and p >= 2.")
    if not np.isfinite(observations).all():
        raise ValueError("DCD does not accept NaN or infinite observations.")
    if admg_class not in PENALTIES:
        raise ValueError(f"Unknown DCD ADMG class {admg_class!r}.")
    if sparsity < 0.0 or threshold < 0.0:
        raise ValueError("DCD sparsity and threshold must be nonnegative.")
    if max_iterations < 1 or num_restarts < 1 or ricf_increment < 1:
        raise ValueError("DCD iteration and restart counts must be positive.")
    if not 0.0 < constraint_tolerance < 1.0:
        raise ValueError("DCD constraint tolerance must lie between zero and one.")
    if rho_max <= 1.0 or coefficient_bound <= 0.0 or jitter <= 0.0:
        raise ValueError("DCD rho max, coefficient bound, and jitter must be positive.")

    centered = observations - observations.mean(axis=0, keepdims=True)
    if np.any(centered.std(axis=0) <= np.finfo(float).eps):
        raise ValueError("DCD requires every observed variable to be nonconstant.")
    rng = np.random.default_rng(random_state)
    best = None
    for restart in range(num_restarts):
        directed_mask, bidirected_mask, converged, constraint, failures = _one_restart(
            centered,
            admg_class=admg_class,
            sparsity=sparsity,
            threshold=threshold,
            max_iterations=max_iterations,
            constraint_tolerance=constraint_tolerance,
            rho_max=rho_max,
            ricf_increment=ricf_increment,
            ricf_tolerance=ricf_tolerance,
            optimizer_max_iterations=optimizer_max_iterations,
            coefficient_bound=coefficient_bound,
            jitter=jitter,
            rng=rng,
            verbose=verbose,
        )
        directed_fit, error_covariance, bic = refit_admg(
            centered,
            directed_mask,
            bidirected_mask,
            tolerance=ricf_tolerance,
            max_iterations=ricf_refit_iterations,
            jitter=jitter,
        )
        candidate = (
            bic,
            restart,
            directed_fit,
            error_covariance,
            directed_mask,
            bidirected_mask,
            converged,
            constraint,
            failures,
        )
        if best is None or candidate[0] < best[0]:
            best = candidate

    assert best is not None
    bic, restart, directed_fit, error_covariance, directed_mask, bidirected_mask, converged, constraint, failures = best
    # Convert parent-by-child coefficients to the repository's B[child, parent].
    directed_weights = directed_fit.T
    return DCDResult(
        directed_weights=directed_weights,
        error_covariance=error_covariance,
        directed=directed_mask.T.astype(int),
        bidirected=bidirected_mask.astype(int),
        bic=float(bic),
        converged=bool(converged),
        restart=int(restart),
        constraint=float(constraint),
        optimizer_failures=int(failures),
    )
