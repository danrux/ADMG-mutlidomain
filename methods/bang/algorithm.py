"""Python implementation of the empirical-likelihood BANG procedure.

This module follows Algorithms 1--4 and the reference implementation in
Y. Samuel Wang and Mathias Drton, "Causal Discovery with Unobserved
Confounding and Non-Gaussian Data". The authors' R package ``ngBap`` is MIT
licensed; provenance details are recorded in ``NOTICE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True)
class BANGResult:
    total_effect: np.ndarray
    direct_effect: np.ndarray
    omega: np.ndarray
    directed: np.ndarray
    bidirected: np.ndarray
    residuals: np.ndarray
    tests: int
    numerical_failures: int
    maximum_set_size: int


def moment_restrictions(
    left: np.ndarray,
    right: np.ndarray,
    degree: int,
    restriction: int,
) -> np.ndarray:
    """Construct the moment restrictions used by the reference implementation."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float).reshape(-1)
    if left.ndim == 1:
        left = left[:, None]
    if left.ndim != 2 or len(left) != len(right):
        raise ValueError("Moment-test operands must have the same sample dimension.")
    if degree < 3:
        raise ValueError("BANG moment degree K must be at least 3.")
    if restriction == 1:
        powers: Iterable[int] = range(2, degree)
    elif restriction == 2:
        powers = (degree - 1,)
    else:
        raise ValueError("BANG restriction must be 1 or 2.")

    columns = []
    right_column = right[:, None]
    for power in powers:
        columns.append(np.power(left, power) * right_column)
        columns.append(left * np.power(right_column, power))
    return np.hstack(columns)


def empirical_likelihood_pvalue(
    restrictions: np.ndarray,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """Empirical-likelihood ratio test for a zero vector mean.

    The dual problem is solved with damped Newton iterations. Projecting onto
    the numerical column space and scaling its axes does not change the null
    constraints, but substantially improves stability for higher moments.
    """
    values = np.asarray(restrictions, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Empirical likelihood requires an n-by-q moment matrix.")
    if not np.isfinite(values).all():
        return 0.0

    rms = np.sqrt(np.mean(values * values, axis=0))
    values = values[:, rms > np.finfo(float).eps]
    if values.shape[1] == 0:
        return 1.0
    values = values / np.sqrt(np.mean(values * values, axis=0))[None, :]

    _, singular_values, right_vectors = np.linalg.svd(values, full_matrices=False)
    rank_tolerance = max(values.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.sum(singular_values > rank_tolerance))
    if rank == 0:
        return 1.0
    values = values @ right_vectors[:rank].T
    values /= np.sqrt(np.mean(values * values, axis=0))[None, :]

    sample_size = values.shape[0]
    mean_norm = np.linalg.norm(values.mean(axis=0))
    if mean_norm <= tolerance:
        return 1.0

    multiplier = np.zeros(rank)
    converged = False
    for _ in range(max_iterations):
        denominator = 1.0 + values @ multiplier
        if np.any(denominator <= 0.0):
            return 0.0
        gradient = np.sum(values / denominator[:, None], axis=0)
        if np.linalg.norm(gradient) / sample_size <= tolerance:
            converged = True
            break
        scaled = values / denominator[:, None]
        curvature = scaled.T @ scaled
        try:
            step = np.linalg.solve(curvature, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(curvature, gradient, rcond=None)[0]

        current_objective = float(np.log(denominator).sum())
        directional_gain = float(gradient @ step)
        step_size = 1.0
        accepted = False
        while step_size >= 2.0**-30:
            candidate = multiplier + step_size * step
            candidate_denominator = 1.0 + values @ candidate
            if np.all(candidate_denominator > 1e-12):
                candidate_objective = float(np.log(candidate_denominator).sum())
                if candidate_objective >= current_objective + 1e-4 * step_size * directional_gain:
                    multiplier = candidate
                    accepted = True
                    break
            step_size *= 0.5
        if not accepted:
            break

    if not converged:
        denominator = 1.0 + values @ multiplier
        if np.any(denominator <= 0.0):
            return 0.0
        gradient = np.sum(values / denominator[:, None], axis=0)
        converged = np.linalg.norm(gradient) / sample_size <= 10.0 * tolerance
    if not converged:
        return 0.0

    statistic = max(0.0, 2.0 * float(np.log1p(values @ multiplier).sum()))
    return float(chi2.sf(statistic, df=rank))


class _BANG:
    def __init__(
        self,
        observations: np.ndarray,
        *,
        degree: int,
        level: float,
        restriction: int,
        max_set_size: int | None,
        max_iterations: int,
        condition_limit: float,
        verbose: bool,
    ):
        observations = np.asarray(observations, dtype=float)
        if observations.ndim != 2 or observations.shape[0] < 2:
            raise ValueError("BANG requires an n-by-p observation matrix with n >= 2.")
        if not np.isfinite(observations).all():
            raise ValueError("BANG does not accept NaN or infinite observations.")
        if not 0.0 < level < 1.0:
            raise ValueError("BANG test level must lie strictly between 0 and 1.")
        if degree < 3:
            raise ValueError("BANG moment degree K must be at least 3.")
        if restriction not in (1, 2):
            raise ValueError("BANG restriction must be 1 or 2.")
        if max_set_size is not None and max_set_size < 1:
            raise ValueError("BANG maximum set size must be positive or None.")
        if max_iterations < 1:
            raise ValueError("BANG maximum iterations must be positive.")
        if condition_limit <= 1.0:
            raise ValueError("BANG condition-number limit must be greater than 1.")

        centered = observations - observations.mean(axis=0, keepdims=True)
        scales = centered.std(axis=0, ddof=1)
        if np.any(scales <= np.finfo(float).eps):
            bad = np.where(scales <= np.finfo(float).eps)[0].tolist()
            raise ValueError(f"BANG requires nonconstant variables; constant columns: {bad}.")
        self.original_scales = scales
        self.original = centered / scales[None, :]
        self.errors = self.original.copy()
        self.n, self.p = self.original.shape
        self.degree = degree
        self.level = level
        self.restriction = restriction
        self.max_set_size = max_set_size
        self.max_iterations = max_iterations
        self.condition_limit = condition_limit
        self.verbose = verbose
        self.direct = np.zeros((self.p, self.p))
        self.total = np.eye(self.p)
        self.siblings = np.ones((self.p, self.p), dtype=bool)
        np.fill_diagonal(self.siblings, False)
        self.tests = 0
        self.numerical_failures = 0
        self.maximum_set_size = 0

    def _test(self, left: np.ndarray, right: np.ndarray) -> float:
        self.tests += 1
        moments = moment_restrictions(left, right, self.degree, self.restriction)
        return empirical_likelihood_pvalue(moments)

    def _debiased_state(
        self,
        parents: list[int],
        vertex: int,
        direct: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not parents:
            total = np.linalg.inv(np.eye(self.p) - direct)
            return direct, total, self.original[:, vertex].copy()

        ancestor_mask = np.any(np.abs(self.total[parents, :]) > 1e-12, axis=0)
        ancestors = np.flatnonzero(ancestor_mask)
        omega = np.cov(self.errors, rowvar=False)
        coefficient_matrix = omega[np.ix_(parents, ancestors)] @ self.total[np.ix_(parents, ancestors)].T
        if (
            coefficient_matrix.shape[0] != coefficient_matrix.shape[1]
            or not np.isfinite(coefficient_matrix).all()
            or np.linalg.cond(coefficient_matrix) > self.condition_limit
        ):
            raise np.linalg.LinAlgError("Ill-conditioned BANG debiasing system.")
        rhs = self.errors[:, parents].T @ self.original[:, vertex] / self.n
        coefficients = np.linalg.solve(coefficient_matrix, rhs)

        updated = direct.copy()
        updated[vertex, :] = 0.0
        updated[vertex, parents] = coefficients
        system = np.eye(self.p) - updated
        if np.linalg.cond(system) > self.condition_limit:
            raise np.linalg.LinAlgError("BANG candidate introduces an ill-conditioned system.")
        total = np.round(np.linalg.inv(system), 10)
        residual = self.original[:, vertex] - self.errors[:, ancestors] @ total[vertex, ancestors]
        return updated, total, residual

    def _ancestor_pvalue(self, candidate: list[int], vertex: int) -> float:
        try:
            _, _, residual = self._debiased_state(candidate, vertex, self.direct)
        except np.linalg.LinAlgError:
            self.numerical_failures += 1
            return 0.0
        return self._test(self.errors[:, candidate], residual)

    def _update(self, parents: list[int], vertex: int) -> bool:
        if not parents:
            return False
        updated, total, residual = self._debiased_state(parents, vertex, self.direct)
        changed = not np.allclose(updated, self.direct, atol=1e-12, rtol=0.0)
        self.direct = updated
        self.total = total
        self.errors[:, vertex] = residual
        return changed

    def _ordering(self) -> list[int]:
        ordering = [0]
        total = self.total.copy()
        np.fill_diagonal(total, 0.0)
        for vertex in range(1, self.p):
            descendants = set(np.flatnonzero(np.abs(total[:, vertex]) > 1e-12).tolist())
            position = len(ordering) - 1
            while position >= 0 and ordering[position] not in descendants:
                position -= 1
            ordering.insert(position + 1, vertex)
        return ordering

    def _prune_pvalue(self, parent: int, vertex: int, parents: list[int]) -> float:
        if len(parents) == 1:
            return self._test(self.errors[:, parent], self.original[:, vertex])
        retained = [value for value in parents if value != parent]
        try:
            _, _, residual = self._debiased_state(retained, vertex, self.direct)
        except np.linalg.LinAlgError:
            self.numerical_failures += 1
            return 0.0
        return self._test(self.errors[:, parents], residual)

    def run(self) -> BANGResult:
        set_size = 1
        loop_iterations = 0
        while np.any(self.siblings.sum(axis=1) >= set_size):
            loop_iterations += 1
            if loop_iterations > self.max_iterations:
                raise RuntimeError(
                    "BANG exceeded its iteration limit; increase --bang-max-iterations "
                    "or restrict --bang-max-set-size."
                )
            if self.max_set_size is not None and set_size > self.max_set_size:
                break
            self.maximum_set_size = max(self.maximum_set_size, set_size)
            old_errors = self.errors.copy()

            for vertex in range(self.p):
                possible = np.flatnonzero(self.siblings[vertex]).tolist()
                for candidate in possible:
                    pvalue = self._test(self.errors[:, candidate], self.original[:, vertex])
                    if pvalue > self.level:
                        self.siblings[vertex, candidate] = False
                        self.siblings[candidate, vertex] = False

                descendants = set(np.flatnonzero(np.abs(self.total[:, vertex]) > 1e-12).tolist())
                possible = [
                    value
                    for value in np.flatnonzero(self.siblings[vertex]).tolist()
                    if value not in descendants
                ]
                if len(possible) < set_size:
                    continue

                existing = np.flatnonzero(np.abs(self.direct[vertex]) > 1e-12).tolist()
                certified: set[int] = set()
                for candidate_set in combinations(possible, set_size):
                    conditioning = list(candidate_set) + existing
                    if self._ancestor_pvalue(conditioning, vertex) > self.level:
                        certified.update(conditioning)

                if certified:
                    parents = sorted(certified)
                    if self.verbose:
                        print(f"BANG: certified parents {parents} for node {vertex}")
                    for parent in parents:
                        self.siblings[vertex, parent] = False
                        self.siblings[parent, vertex] = False
                    self._update(parents, vertex)

            if float(np.sum((old_errors - self.errors) ** 2)) > 1e-5:
                set_size = 1
            else:
                set_size += 1

        for vertex in self._ordering():
            parents = np.flatnonzero(np.abs(self.direct[vertex]) > 1e-12).tolist()
            if not parents:
                continue
            retained = [
                parent
                for parent in parents
                if self._prune_pvalue(parent, vertex, parents) < self.level
            ]
            if retained:
                self._update(retained, vertex)
            else:
                self.direct[vertex, :] = 0.0
                self.total = np.linalg.inv(np.eye(self.p) - self.direct)
                self.errors[:, vertex] = self.original[:, vertex]

        # Undo the internal diagonal rescaling: B_original = S B_scaled S^-1.
        direct_effect = (
            self.original_scales[:, None]
            * self.direct
            / self.original_scales[None, :]
        )
        total_effect = np.linalg.inv(np.eye(self.p) - direct_effect)
        residuals = (
            self.errors * self.original_scales[None, :]
        )
        return BANGResult(
            total_effect=total_effect,
            direct_effect=direct_effect,
            omega=np.atleast_2d(np.cov(residuals, rowvar=False)),
            directed=(np.abs(direct_effect) > 0).astype(int),
            bidirected=self.siblings.astype(int),
            residuals=residuals,
            tests=self.tests,
            numerical_failures=self.numerical_failures,
            maximum_set_size=self.maximum_set_size,
        )


def fit_bang(
    observations: np.ndarray,
    *,
    degree: int = 3,
    level: float = 0.01,
    restriction: int = 1,
    max_set_size: int | None = None,
    max_iterations: int = 10_000,
    condition_limit: float = 1e12,
    verbose: bool = False,
) -> BANGResult:
    """Fit BANG using empirical-likelihood moment tests."""
    return _BANG(
        observations,
        degree=degree,
        level=level,
        restriction=restriction,
        max_set_size=max_set_size,
        max_iterations=max_iterations,
        condition_limit=condition_limit,
        verbose=verbose,
    ).run()
