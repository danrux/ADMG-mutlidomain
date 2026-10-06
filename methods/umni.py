"""UMNI-CRL for linear hard interventions.

Adapts the hard-intervention branch of:
  "Linear Causal Representation Learning from Unknown Multi-node Interventions"
  NeurIPS 2024 — https://arxiv.org/abs/2406.05937
  Reference: acarturk-e/score-based-crl/UMNI-CRL/umni_crl.py

Data convention (matches data.linear_hard_intervention):
  X_list[-1]  = reference domain (no interventions), shape (N, xn)
  X_list[:-1] = T intervention domains, shape (N, xn) each
  Requires T >= n at call time and a full-column-rank intervention design.

Score function differences are computed analytically from sample covariances
(valid for the linear Gaussian setting) instead of requiring pre-computed
score samples as in the original code.
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import numpy.typing as npt
from scipy import stats


_MAX_ENCODER_CONDITION = 1e12


# ── inline helpers (from acarturk-e/score-based-crl utils) ───────────────────

def _canonical_weight_batch_generator(kappa: int, n: int, batch_size: int):
    """Yield one representative per equivalent weight class, in batches.

    Since r(w) is quadratic in w, w and -w produce exactly the same matrix.
    Keep the first member of each sign-equivalent pair in the original
    lexicographic search. Other scalar multiples are deliberately retained:
    the algorithm's absolute eigenvalue tolerance can assign them different
    numerical ranks.
    """
    batch = []
    for w in product(range(-kappa, kappa + 1), repeat=n):
        first_nonzero = next((value for value in w if value != 0), None)
        if first_nonzero is None or first_nonzero > 0:
            continue
        batch.append(w)
        if len(batch) == batch_size:
            yield np.asarray(batch, dtype=np.int64)
            batch.clear()
    if batch:
        yield np.asarray(batch, dtype=np.int64)


def _canonicalize_weight(w: npt.NDArray[np.integer]) -> tuple[int, ...] | None:
    """Return a sign-normalized integer weight vector for deduplication."""
    w = np.asarray(w, dtype=np.int64)
    nonzero = np.flatnonzero(w)
    if nonzero.size == 0:
        return None
    if w[nonzero[0]] < 0:
        w = -w
    return tuple(int(value) for value in w)


def _evaluate_weight_batch(
    weights: npt.NDArray[np.integer],
    rx_ij: npt.NDArray[np.floating],
    null_projector: npt.NDArray[np.floating],
    atol_eigv: float,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.int_],
           npt.NDArray[np.floating], int | None]:
    """Evaluate a batch of separating-weight candidates with NumPy kernels."""
    # r_w[k] = sum_ij weights[k,i] weights[k,j] rx_ij[i,j].
    r_w = np.einsum(
        "ki,kj,ijab->kab", weights, weights, rx_ij,
        optimize=True,
    )
    eigval, eigvec = np.linalg.eigh(r_w)
    ranks = np.sum(eigval > atol_eigv, axis=1).astype(np.int64)

    d = eigvec.shape[-1]
    col_mask = np.arange(d)[None, :] >= (d - ranks)[:, None]
    projected = np.einsum(
        "ab,kbc->kac", null_projector, eigvec, optimize=True,
    )
    projected *= col_mask[:, None, :]
    projected_svs = np.linalg.svd(projected, compute_uv=False)
    valid = (ranks > 0) & (np.sum(projected_svs > atol_eigv, axis=1) == 1)
    nonzero_ranks = ranks[ranks > 0]
    best_rank = int(nonzero_ranks.min()) if nonzero_ranks.size else None
    return valid, ranks, eigvec, best_rank


def _search_weight_batches(
    batches,
    rx_ij: npt.NDArray[np.floating],
    null_projector: npt.NDArray[np.floating],
    atol_eigv: float,
    workers: int,
    candidate_validator=None,
):
    """Search ordered batches in parallel while returning the first match."""
    best_rank = None
    with ThreadPoolExecutor(max_workers=workers) as executor:
        batch_iter = iter(batches)
        while True:
            window = []
            for _ in range(workers):
                try:
                    window.append(next(batch_iter))
                except StopIteration:
                    break
            if not window:
                return None, best_rank

            futures = [
                executor.submit(
                    _evaluate_weight_batch,
                    weights,
                    rx_ij,
                    null_projector,
                    atol_eigv,
                )
                for weights in window
            ]
            # Consume in input order, so parallel execution does not change
            # which valid candidate is selected.
            for weights, future in zip(window, futures):
                valid, ranks, eigvec, batch_best_rank = future.result()
                if batch_best_rank is not None:
                    best_rank = (
                        batch_best_rank if best_rank is None
                        else min(best_rank, batch_best_rank)
                    )
                valid_indices = np.flatnonzero(valid)
                for idx_value in valid_indices:
                    idx = int(idx_value)
                    rank = int(ranks[idx])
                    candidate = (
                        weights[idx],
                        rank,
                        eigvec[idx, :, -rank:],
                    )
                    if (candidate_validator is None
                            or candidate_validator(candidate)):
                        return candidate, best_rank


def _divide_by_gcd_cols(w_mat: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    out = np.zeros_like(w_mat)
    for i in range(w_mat.shape[1]):
        col = w_mat[:, i]
        g = math.gcd(*col.tolist())
        if g != 0:
            out[:, i] = -(col // g)
    return out


def _setminus(lst1: list, lst2: list) -> list:
    return [v for v in lst1 if v not in lst2]


def _sample_cov(data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    centered = data - data.mean(axis=0, keepdims=True)
    return centered.T @ centered / data.shape[0]


def _stable_solve(
    matrix: npt.NDArray[np.floating],
    rhs: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Solve a small covariance system, tolerating rank-deficient estimates."""
    if np.linalg.cond(matrix) < 1e12:
        try:
            return np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            pass
    return np.linalg.lstsq(matrix, rhs, rcond=None)[0]


def _encoder_health(
    encoder: npt.NDArray[np.floating],
) -> tuple[int, float, float]:
    """Return numerical rank, condition number, and smallest singular value."""
    singular_values = np.linalg.svd(encoder, compute_uv=False)
    rank = int(np.linalg.matrix_rank(encoder))
    smallest = float(singular_values[-1]) if singular_values.size else 0.0
    condition = float(np.linalg.cond(encoder))
    return rank, condition, smallest


def _partial_corr_suffstat(data: npt.NDArray[np.floating]) -> dict:
    return {"C": np.corrcoef(data.T), "n": data.shape[0]}


def _partial_corr_test(suffstat: dict, i: int, j: int, cond_set: list) -> dict:
    C, n_samp = suffstat["C"], suffstat["n"]
    k = len(cond_set)
    if k == 0:
        r = float(C[i, j])
    else:
        idx = [i, j] + list(cond_set)
        C_sub = C[np.ix_(idx, idx)]
        try:
            C_inv = np.linalg.inv(C_sub)
        except np.linalg.LinAlgError:
            return {"p_value": 1.0}
        r = float(-C_inv[0, 1] / np.sqrt(abs(C_inv[0, 0] * C_inv[1, 1])))
    r = max(-1 + 1e-10, min(1 - 1e-10, r))
    df = n_samp - k - 2
    if df <= 0:
        return {"p_value": 1.0}
    t_stat = r * math.sqrt(df / (1.0 - r * r))
    return {"p_value": float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df)))}


# ── algorithm steps ───────────────────────────────────────────────────────────

def _causal_order(
    rx_ij: npt.NDArray[np.floating],
    atol_eigv: float,
    kappa: int,
    batch_size: int,
    workers: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.int_]]:
    """Identify causal ordering from score-difference covariance matrices.

    rx_ij[j, i] = E[dsx_i dsx_j^T], shape (T, T, n, n), T >= n.
    Returns an n x n encoder and a T x n intervention-weight matrix.
    """

    n_domains, _, n, _ = rx_ij.shape
    if n_domains < n:
        raise ValueError(
            f"#intervention_domains ({n_domains}) must be >= n_latent ({n})"
        )

    h_mat = np.zeros((n, n))
    w_mat = np.zeros((n_domains, n), dtype=np.int64)

    for t in range(n):
        ht_b = np.linalg.qr(h_mat[:t, :].T, "complete").Q
        ht_nullb = ht_b[:, t:]
        ht_nullp = ht_nullb @ ht_nullb.T
        result, best_rank = _search_weight_batches(
            _canonical_weight_batch_generator(kappa, n_domains, batch_size),
            rx_ij,
            ht_nullp,
            atol_eigv,
            workers,
        )

        if result is None:
            rank_text = "none" if best_rank is None else str(best_rank)
            raise RuntimeError(
                f"_causal_order: no separating weight vector for position t={t}; "
                f"smallest nonzero candidate rank was {rank_text} "
                f"(kappa={kappa}, atol_eigv={atol_eigv}). "
                "Try more samples or a larger atol_eigv."
            )

        w, cm_rank, rw_colb = result
        u, _, _ = np.linalg.svd(rw_colb @ rw_colb.T @ ht_nullb)
        h_mat[t, :] = u[:, 0]
        w_mat[:, t] = w
        print(
            "I found a separating weight vector for position "
            f"t={t} (kappa={kappa}, rank={cm_rank})"
        )

    return h_mat, _divide_by_gcd_cols(w_mat)


def _ancestors(
    rx_ij: npt.NDArray[np.floating],
    h_mat_c: npt.NDArray[np.floating],
    w_mat_c: npt.NDArray[np.int_],
    atol_eigv: float,
    batch_size: int,
    workers: int,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.floating], npt.NDArray[np.int_]]:
    """Refine causal order to a full ancestor graph."""
    n_domains, _, n, _ = rx_ij.shape
    if h_mat_c.shape != (n, n) or w_mat_c.shape != (n_domains, n):
        raise ValueError(
            "Incompatible UMNI shapes: "
            f"rx_ij={rx_ij.shape}, h_mat_c={h_mat_c.shape}, "
            f"w_mat_c={w_mat_c.shape}."
        )
    hat_g_s = np.zeros((n, n), dtype=bool)
    h_mat_s = h_mat_c.copy()
    w_mat_s = w_mat_c.copy()

    for t in reversed(range(n)):
        for j in range(t + 1, n):
            if hat_g_s[t, j]:
                continue
            mtj = [i for i in range(j) if i != t and not hat_g_s[t, i]]
            max_w_t = int(np.sum(np.abs(w_mat_s[:, j])))
            max_w_j = int(np.sum(np.abs(w_mat_s[:, t])))
            is_parent = True

            def candidate_batches():
                seen = set()
                batch = []
                for a, b in product(
                    range(-max_w_t, max_w_t + 1),
                    range(1, max_w_j + 1),
                ):
                    w_new = a * w_mat_s[:, t] + b * w_mat_s[:, j]
                    canonical = _canonicalize_weight(w_new)
                    if canonical is None or canonical in seen:
                        continue
                    seen.add(canonical)
                    # Keep the first original-scale representative so the
                    # absolute eigenvalue tolerance behaves as before.
                    batch.append(w_new)
                    if len(batch) == batch_size:
                        yield np.asarray(batch, dtype=np.int64)
                        batch.clear()
                if batch:
                    yield np.asarray(batch, dtype=np.int64)

            h_mtj_b = np.linalg.qr(h_mat_s[mtj, :].T, "complete").Q
            h_mtj_nullb = h_mtj_b[:, len(mtj):]
            h_mtj_nullp = h_mtj_nullb @ h_mtj_nullb.T
            accepted_h = None

            def preserves_encoder_health(candidate):
                nonlocal accepted_h
                _, _, rw_colb = candidate
                u, _, _ = np.linalg.svd(
                    rw_colb @ rw_colb.T @ h_mtj_nullb
                )
                candidate_h = h_mat_s.copy()
                candidate_h[j, :] = u[:, 0]
                candidate_rank, candidate_cond, _ = _encoder_health(candidate_h)
                if (candidate_rank < n
                        or not np.isfinite(candidate_cond)
                        or candidate_cond > _MAX_ENCODER_CONDITION):
                    return False
                accepted_h = candidate_h
                return True

            result, _ = _search_weight_batches(
                candidate_batches(),
                rx_ij,
                h_mtj_nullp,
                atol_eigv,
                workers,
                candidate_validator=preserves_encoder_health,
            )
            if result is not None:
                w_new, _, _ = result
                h_mat_s = accepted_h
                w_mat_s[:, j] = w_new
                is_parent = False

            if is_parent:
                hat_g_s[t, j] = True
                hat_g_s[t, :] |= hat_g_s[j, :]

    return hat_g_s, h_mat_s, w_mat_s


def _unmixing_cov(
    X_list: list,
    hat_enc_s: npt.NDArray[np.floating],
    hat_g_s: npt.NDArray[np.bool_],
    w_mat_c: npt.NDArray[np.int_],
    atol_ci_test: float,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.floating], dict]:
    """Encoder refinement and graph pruning using hard-intervention covariance shifts.

    X_list[-1] is the reference domain; X_list[m] is intervention domain m.
    Uses w_mat_c (from _causal_order, not _ancestors) to locate relevant domains.
    """
    n = hat_enc_s.shape[0]
    hat_enc_h = hat_enc_s.copy()

    # Per-domain second moments; reindex so slot 0 = reference, slot m+1 = intervention m
    x_cov = np.stack([_sample_cov(X) for X in X_list])  # (T+1, n, n)
    x_cov = np.concatenate([x_cov[-1:], x_cov[:-1]], axis=0)

    ### ENCODER UPDATE
    for t in range(1, n):
        an_t = np.where(hat_g_s[:, t])[0]
        if len(an_t) == 0:
            continue
        hat_z_cov = hat_enc_h @ x_cov @ hat_enc_h.T  # (T+1, n, n)
        u_obs = _stable_solve(
            hat_z_cov[0][np.ix_(an_t, an_t)],
            -hat_z_cov[0][an_t, t],
        )
        for m in np.where(w_mat_c[:, t])[0]:
            u_m = _stable_solve(
                hat_z_cov[m + 1][np.ix_(an_t, an_t)],
                -hat_z_cov[m + 1][an_t, t],
            )
            if np.linalg.norm(u_m - u_obs) > 1e-1:
                hat_enc_h[t, :] += u_m @ hat_enc_h[an_t, :]
                break

    ### GRAPH UPDATE
    hat_z_obs = (hat_enc_h @ X_list[-1].T).T    # (N, n)
    suffstat = _partial_corr_suffstat(hat_z_obs)
    hat_g_h = np.empty((n, n), dtype=bool)
    for t in range(n):
        for j in range(n):
            if not hat_g_s[t, j]:
                hat_g_h[t, j] = False
            else:
                cond = _setminus(list(np.where(hat_g_s[:, j])[0]), [t])
                p_val = _partial_corr_test(suffstat, t, j, cond)["p_value"]
                hat_g_h[t, j] = p_val <= atol_ci_test

    return hat_g_h, hat_enc_h, suffstat


# ── public interface ──────────────────────────────────────────────────────────

def estimate(X_list, Z_list, T, n, xn, args):
    """UMNI-CRL estimate for linear hard-intervention data.

    Standard method interface: returns (A_hat_raw, aux_dict).

    X_list is expected from data.linear_hard_intervention.generate_data, so
    X_list[-1] is the reference domain and len(X_list) - 1 must be at least n.

    The algorithm requires sparse interventions: each domain should intervene on
    at most kappa nodes (default kappa=1). Use interv_dense=1/n when generating
    data to get single-node interventions per domain.
    """
    atol_eigv    = getattr(args, "atol_eigv",    0.05)
    atol_ci_test = getattr(args, "atol_ci_test", 0.05)
    kappa        = getattr(args, "kappa",         1)
    search_batch_size = getattr(args, "umni_batch_size", 2048)
    requested_workers = getattr(args, "umni_workers", 0)
    search_workers = (
        min(4, os.cpu_count() or 1)
        if requested_workers == 0
        else requested_workers
    )
    if kappa < 1:
        raise ValueError(f"UMNI --kappa must be >= 1, got {kappa}.")
    if search_batch_size < 1:
        raise ValueError(
            f"UMNI --umni-batch-size must be >= 1, got {search_batch_size}."
        )
    if search_workers < 1:
        raise ValueError(
            f"UMNI --umni-workers must be >= 1, got {search_workers}."
        )

    T_interv = len(X_list) - 1
    N = X_list[0].shape[0]

    if T_interv < n:
        raise ValueError(
            "UMNI-CRL requires #intervention_domains >= n_latent "
            f"({T_interv} vs {n}). Set T>=n when generating data."
        )

    # ── subspace projection (identity when xn == n) ───────────────────────
    x_cov_sum = sum(_sample_cov(X) for X in X_list)
    _, dec_svec = np.linalg.eigh(x_cov_sum)
    dec_colbt = dec_svec[:, -n:].T                         # (n, xn)
    X_proj = [(dec_colbt @ X.T).T for X in X_list]         # each (N, n)

    # ── score function differences (linear Gaussian: ds_t = Δ_t x_ref) ───
    # For X ~ N(0, Σ), score = -Σ^{-1} x; difference = (Σ_ref^{-1} - Σ_t^{-1}) x
    Sigma_ref = _sample_cov(X_proj[-1])
    Sigma_ref_inv = np.linalg.inv(Sigma_ref)

    dsx = []
    for t in range(T_interv):
        Sigma_t = _sample_cov(X_proj[t])
        Delta_t = np.linalg.inv(Sigma_t) - Sigma_ref_inv   # Σ_t^{-1} - Σ_ref^{-1}
        dsx.append((Delta_t @ X_proj[-1].T).T)              # (N, n)

    # rx_ij[j, i] = (1/N) dsx[i]^T dsx[j]  ←  E[dsx_i dsx_j^T]
    rx_ij = np.array([
        [dsx[i].T @ dsx[j] / N for i in range(T_interv)]
        for j in range(T_interv)
    ])  # (T, T, n, n)

    # ── causal structure recovery ─────────────────────────────────────────
    hat_enc_n_c, w_mat_c = _causal_order(
        rx_ij, atol_eigv, kappa, search_batch_size, search_workers
    )
    hat_g_s, hat_enc_n_s, _ = _ancestors(
        rx_ij,
        hat_enc_n_c,
        w_mat_c,
        atol_eigv,
        search_batch_size,
        search_workers,
    )
    hat_g_h, hat_enc_n_h, _ = _unmixing_cov(
        X_proj, hat_enc_n_s, hat_g_s, w_mat_c, atol_ci_test
    )

    # ── lift encoder back to xn dimensions; return mixing matrix ─────────
    hat_enc_h = hat_enc_n_h @ dec_colbt    # (n, xn)
    encoder_rank, encoder_cond, smallest_sv = _encoder_health(hat_enc_h)
    if (encoder_rank < n
            or not np.isfinite(encoder_cond)
            or encoder_cond > _MAX_ENCODER_CONDITION):
        raise RuntimeError(
            "UMNI-CRL produced an invalid final encoder: "
            f"shape={hat_enc_h.shape}, rank={encoder_rank} (expected {n}), "
            f"condition_number={encoder_cond:.3e}, "
            f"smallest_singular_value={smallest_sv:.3e}. "
            "The recovered latent directions are linearly dependent or too "
            "ill-conditioned to construct a reliable decoder. Try more "
            "samples or adjust --atol-eigv."
        )
    A_hat_raw = np.linalg.pinv(hat_enc_h)  # (xn, n)  ≈ (I-A)^{-1}

    return A_hat_raw, {
        "D_hats": None,
        "sigma_hat": None,
        "final_loss": None,
        "encoder_mcc": None,
        "encoder": None,
        "hat_g_h": hat_g_h,
        "hat_g_s": hat_g_s,
    }
