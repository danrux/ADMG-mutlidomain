import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from itertools import permutations


def get_aligned_z_hat(A_hat_raw, X_list):
    """Compute Z_hat for each domain, with rows permuted to align Z_hat_i -> X_i.

    Uses the same Hungarian row assignment as find_P_and_prune_fast so that
    Z_hat[sample, i] is the latent component that primarily drives X_i.
    """
    W = np.linalg.inv(A_hat_raw)
    n = W.shape[0]
    row_ind, col_ind = linear_sum_assignment(-np.abs(W))
    perm = np.empty(n, dtype=int)
    perm[col_ind] = row_ind
    W_perm = W[perm, :]
    return [X_t @ W_perm.T for X_t in X_list]


def estimate_bidirected_from_Z_hat_list(z_hat_all, threshold=0.05):
    """Estimate bidirected edges from the selected ``Z_hat`` sample matrix."""
    n = z_hat_all.shape[1]
    bidir = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            corr = np.corrcoef(z_hat_all[:, i], z_hat_all[:, j])[0, 1]
            if np.isnan(corr):
                continue
            if np.abs(corr) > threshold:
                bidir[i, j] = bidir[j, i] = 1

    return bidir


def select_bidirected_observations(X_list, domain_parameters, dgp):
    """Select the observations required for bidirected-edge estimation.

    Auxiliary and scaling DGPs define the relevant distribution over all
    domains, so their observations are pooled. Mask and hard-intervention
    DGPs instead use their appended observational reference domain.

    Returns ``(observations, domain_index, reason)``. ``domain_index`` is
    ``None`` for pooled data.
    """
    if len(X_list) == 0:
        raise ValueError("Bidirected estimation requires at least one domain.")

    if dgp in ("auxillary", "scaling"):
        reason = "auxiliary-pooled" if dgp == "auxillary" else "scaling-pooled"
        return np.vstack(X_list), None, reason

    if dgp in ("mask", "nonlinear_mask"):
        for index in range(len(domain_parameters) - 1, -1, -1):
            if np.all(np.asarray(domain_parameters[index]) == 1.0):
                return X_list[index], index, "all-ones-mask"
        raise ValueError(f"The {dgp} DGP contains no fully unmasked reference domain.")

    if dgp == "hard_intervention":
        for index in range(len(domain_parameters) - 1, -1, -1):
            if len(domain_parameters[index]) == 0:
                return X_list[index], index, "non-intervention"
        raise ValueError(
            "The hard-intervention DGP contains no non-intervened reference domain."
        )

    raise ValueError(f"No bidirected-observation policy is defined for DGP {dgp!r}.")


def estimate_bidirected_from_Z_hat(A_hat_raw, X_list, threshold=0.05):
    """Estimate bidirected edges after pooling the supplied domains.

    Callers for mask and hard-intervention DGPs must pass only the selected
    reference domain; :func:`select_bidirected_observations` implements that
    DGP-specific selection policy.
    """
    pooled = np.vstack(X_list)
    z_hat_all = get_aligned_z_hat(A_hat_raw, [pooled])[0]
    return estimate_bidirected_from_Z_hat_list(z_hat_all, threshold)


def compute_admg_metrics(directed_true, bidirected_true, directed_hat, bidirected_hat):
    """Evaluate ADMG estimation performance for directed and bidirected parts.

    bidirected matrices are symmetric; only the upper triangle is counted to
    avoid double-counting each edge.

    In addition to the legacy directed/bidirected metrics, ``endpoint_recovery``
    uses the DCD paper's TPR/FDR definitions on the supplied exact ADMGs. It
    reports recovery for skeletons, arrowheads, and tails. Skeletons are counted
    once per unordered node pair, while marks are counted at each endpoint.
    """
    dir_metrics = compute_graph_metrics_with_reverse(directed_true, directed_hat)

    n = bidirected_true.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    bt = bidirected_true[mask]
    bh = bidirected_hat[mask]
    tp = int(np.sum((bt == 1) & (bh == 1)))
    fp = int(np.sum((bt == 0) & (bh == 1)))
    fn = int(np.sum((bt == 1) & (bh == 0)))
    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)
    shd       = int(np.sum(bt != bh))
    bidir_metrics = {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "shd": shd,
    }

    endpoint_recovery = compute_admg_endpoint_recovery_metrics(
        directed_true,
        bidirected_true,
        directed_hat,
        bidirected_hat,
    )

    return {
        "directed": dir_metrics,
        "bidirected": bidir_metrics,
        "endpoint_recovery": endpoint_recovery,
    }


def _tpr_fdr(target, prediction):
    """Return DCD-style recovery rates for binary target/prediction vectors."""
    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if target.shape != prediction.shape:
        raise ValueError("Target and prediction must have the same shape.")

    tp = int(np.sum(target & prediction))
    fp = int(np.sum(~target & prediction))
    target_count = int(np.sum(target))
    predicted_count = int(np.sum(prediction))
    tpr = tp / target_count if target_count else 0.0
    fdr = fp / predicted_count if predicted_count else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "target": target_count,
        "predicted": predicted_count,
        "tpr": float(tpr),
        "fdr": float(fdr),
    }


def compute_admg_endpoint_recovery_metrics(
    directed_true,
    bidirected_true,
    directed_hat,
    bidirected_hat,
):
    """Compute DCD-style skeleton/arrowhead/tail TPR and FDR on exact ADMGs.

    Matrices follow this repository's convention: ``directed[child, parent]``
    denotes ``parent -> child``. Thus directed entries mark arrowheads and
    their transpose marks tails. A bidirected edge contributes an arrowhead at
    both endpoints. Separate masks also give well-defined counts for bows.
    """
    matrices = [
        np.asarray(directed_true),
        np.asarray(bidirected_true),
        np.asarray(directed_hat),
        np.asarray(bidirected_hat),
    ]
    shape = matrices[0].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"ADMG matrices must be square, got {shape}.")
    if any(matrix.shape != shape for matrix in matrices[1:]):
        raise ValueError("All true and estimated ADMG matrices must have equal shapes.")

    dt, bt, dh, bh = (matrix != 0 for matrix in matrices)
    off_diagonal = ~np.eye(shape[0], dtype=bool)
    upper = np.triu(np.ones(shape, dtype=bool), k=1)

    true_skeleton = (dt | dt.T | bt | bt.T)[upper]
    estimated_skeleton = (dh | dh.T | bh | bh.T)[upper]
    true_arrowheads = (dt | bt | bt.T)[off_diagonal]
    estimated_arrowheads = (dh | bh | bh.T)[off_diagonal]
    true_tails = dt.T[off_diagonal]
    estimated_tails = dh.T[off_diagonal]

    return {
        "skeleton": _tpr_fdr(true_skeleton, estimated_skeleton),
        "arrowhead": _tpr_fdr(true_arrowheads, estimated_arrowheads),
        "tail": _tpr_fdr(true_tails, estimated_tails),
    }


def corrcoef_torch(X, Y):
    Xm = X - X.mean(dim=0, keepdim=True)
    Ym = Y - Y.mean(dim=0, keepdim=True)
    cov = (Xm.T @ Ym) / (X.shape[0] - 1)
    std_X = Xm.std(dim=0, unbiased=True).unsqueeze(1)
    std_Y = Ym.std(dim=0, unbiased=True).unsqueeze(0)
    corr = cov / (std_X * std_Y + 1e-8)
    return corr


def compute_mcc(A_hat, X_list, Z_list, n):
    if isinstance(A_hat, np.ndarray):
        A_hat = torch.from_numpy(A_hat).float()
        device = torch.device("cpu")
    else:
        device = A_hat.device

    T = len(X_list)
    mcc_list = []
    cor_list = []

    for t in range(T):
        X_t = torch.tensor(X_list[t], dtype=torch.float32, device=device)
        Z_t = torch.tensor(Z_list[t], dtype=torch.float32, device=device)

        AtA = A_hat.T @ A_hat
        AtX = A_hat.T @ X_t.T
        hz_t = torch.linalg.solve(AtA, AtX).T

        cor_abs_t = torch.abs(corrcoef_torch(Z_t, hz_t))
        cor_abs_np = cor_abs_t.detach().cpu().numpy()

        row_ind, col_ind = linear_sum_assignment(-cor_abs_np)
        mcc_t = cor_abs_np[row_ind, col_ind].sum() / n

        mcc_list.append(mcc_t)
        cor_list.append(cor_abs_np)

    avg_mcc = np.mean(mcc_list)
    return avg_mcc, mcc_list, cor_list


# -----------------------------
# Permutation / post-processing
# -----------------------------

def find_P(W):
    # Step 1: find permutation that maximises sum of absolute diagonal entries
    best_perm = None
    best_score = -np.inf

    for perm in permutations(range(W.shape[0])):
        Wp = W[list(perm)]
        score = np.sum(np.abs(np.diag(Wp)))
        if score > best_score:
            best_score = score
            best_perm = perm

    W = W[list(best_perm)]

    # Step 2: normalize rows
    for i in range(W.shape[0]):
        W[i] /= W[i, i]

    # Step 3: compute B
    B = np.eye(W.shape[0]) - W
    return B


def is_dag(B):
    """Return True if the weighted adjacency matrix B represents a DAG."""
    B_bin = (np.abs(B) > 0).astype(int)
    np.fill_diagonal(B_bin, 0)

    n = B.shape[0]
    indeg = B_bin.sum(axis=0)
    stack = [i for i in range(n) if indeg[i] == 0]
    visited = 0

    while stack:
        u = stack.pop()
        visited += 1
        for v in range(n):
            if B_bin[u, v]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    stack.append(v)

    return visited == n


def find_cycle(B):
    """Return one directed cycle as [a, b, ..., a], or None if B is a DAG."""
    B_bin = (np.abs(B) > 0).astype(int)
    np.fill_diagonal(B_bin, 0)
    n = B.shape[0]

    state = [0] * n   # 0=unvisited, 1=visiting, 2=done
    parent = [-1] * n

    def dfs(u):
        state[u] = 1
        for v in range(n):
            if not B_bin[u, v]:
                continue
            if state[v] == 0:
                parent[v] = u
                cyc = dfs(v)
                if cyc is not None:
                    return cyc
            elif state[v] == 1:
                cycle = [v]
                cur = u
                while cur != v:
                    cycle.append(cur)
                    cur = parent[cur]
                cycle.append(v)
                cycle.reverse()
                return cycle
        state[u] = 2
        return None

    for i in range(n):
        if state[i] == 0:
            cyc = dfs(i)
            if cyc is not None:
                return cyc
    return None


def adaptive_threshold(B):
    """
    Find a threshold at the largest absolute gap in the sorted off-diagonal |B| values.
    This detects the natural separation between real edges and numerical noise.
    Falls back to 0.0 if fewer than 2 nonzero entries exist.
    """
    vals = np.abs(B.copy())
    np.fill_diagonal(vals, 0.0)
    vals = np.sort(vals.ravel())
    vals = vals[vals > 1e-10]
    if len(vals) < 2:
        return 0.0
    idx = int(np.argmax(np.diff(vals)))
    return float((vals[idx] + vals[idx + 1]) / 2.0)


def _cycle_edges_reachability(B_bin):
    """
    Return the set of edges (i, j) that lie on at least one directed cycle.
    Uses transitive closure (O(n^3)) so the full set is found in one pass,
    avoiding repeated DFS calls.
    """
    # reach[i, j] = True iff there is a directed path i -> j of length >= 1
    reach = B_bin.astype(bool).copy()
    n = B_bin.shape[0]
    for k in range(n):
        reach |= reach[:, k:k+1] & reach[k:k+1, :]
    return {(i, j) for i in range(n) for j in range(n)
            if B_bin[i, j] and reach[j, i]}


def prune_weighted_adjacency_to_dag(B, threshold=1e-3, relative_threshold=False,
                                    strategy="global", verbose=False):
    """
    Prune small edges, then break remaining cycles.

    threshold : numeric value, or "auto" to detect from the edge-weight distribution
    strategy  :
      "global"  — at each step remove the globally weakest edge that participates in
                  any cycle (better approximation to minimum feedback arc set)
      "greedy"  — original behaviour: remove weakest edge in the first found cycle
    """
    B = B.copy()
    np.fill_diagonal(B, 0.0)

    if threshold == "auto":
        thr = adaptive_threshold(B)
        if verbose:
            print(f"Adaptive threshold: {thr:.6e}")
    elif relative_threshold and np.max(np.abs(B)) > 0:
        thr = threshold * np.max(np.abs(B))
    else:
        thr = threshold

    B[np.abs(B) < thr] = 0.0
    np.fill_diagonal(B, 0.0)

    while True:
        B_bin = (np.abs(B) > 0).astype(int)

        if strategy == "global":
            cycle_edge_set = _cycle_edges_reachability(B_bin)
            if not cycle_edge_set:
                break
            i, j = min(cycle_edge_set, key=lambda e: abs(B[e[0], e[1]]))
        else:  # greedy
            cycle = find_cycle(B)
            if cycle is None:
                break
            edges = [(cycle[k], cycle[k + 1]) for k in range(len(cycle) - 1)]
            i, j = min(edges, key=lambda e: abs(B[e[0], e[1]]))

        if verbose:
            print(f"Removing cycle edge {i}->{j} with weight {B[i, j]:.6e}")
        B[i, j] = 0.0

    return B


def _permute_and_normalize(W):
    """Shared helper: Hungarian row permutation + row normalization."""
    n = W.shape[0]
    row_ind, col_ind = linear_sum_assignment(-np.abs(W))
    perm = np.empty(n, dtype=int)
    perm[col_ind] = row_ind
    W = W[perm, :]
    diag = np.diag(W)
    if np.any(np.isclose(diag, 0.0)):
        bad = np.where(np.isclose(diag, 0.0))[0]
        raise ValueError(f"Diagonal entries too close to zero after assignment at indices {bad.tolist()}")
    return W / diag[:, None], perm, diag


def align_z_hat(z_hat, A_hat_raw):
    """Apply the same permutation and normalization as find_P_and_prune_fast to z_hat columns.

    z_hat : (N, n) array — encoder output or W @ X.T
    Returns z_hat with columns permuted and scaled to match the B_hat variable ordering.
    """
    W = np.linalg.inv(A_hat_raw)
    _, perm, diag = _permute_and_normalize(W.copy())
    return z_hat[:, perm] / diag[np.newaxis, :]


def find_P_and_prune(W, threshold=1e-3, relative_threshold=False,
                     strategy="global", verbose=False):
    """
    Brute-force permutation version. Prefer find_P_and_prune_fast for n > ~8.
    """
    W = W.copy()
    n = W.shape[0]

    best_perm, best_score = None, -np.inf
    for perm in permutations(range(n)):
        Wp = W[list(perm), :]
        score = np.sum(np.abs(np.diag(Wp)))
        if score > best_score:
            best_score = score
            best_perm = perm

    W = W[list(best_perm), :]
    for i in range(n):
        if np.isclose(W[i, i], 0.0):
            raise ValueError(f"Diagonal entry W[{i},{i}] is too close to zero after permutation.")
        W[i, :] /= W[i, i]

    B = np.eye(n) - W
    np.fill_diagonal(B, 0.0)
    return prune_weighted_adjacency_to_dag(B, threshold=threshold,
                                           relative_threshold=relative_threshold,
                                           strategy=strategy, verbose=verbose)


def find_P_and_prune_fast(W, threshold=1e-3, relative_threshold=False,
                          strategy="global", verbose=False):
    """
    Hungarian-matching version. O(n^3) vs O(n!) for brute force.
    Pass threshold="auto" to detect the threshold from the edge-weight distribution.
    """
    W, _, _ = _permute_and_normalize(W.copy())
    B = np.eye(W.shape[0]) - W
    np.fill_diagonal(B, 0.0)
    return prune_weighted_adjacency_to_dag(B, threshold=threshold,
                                           relative_threshold=relative_threshold,
                                           strategy=strategy, verbose=verbose)


# -----------------------------
# Graph metrics
# -----------------------------

def collect_true_est_edges(A_true, B_hat):
    """Return (true_vals, est_vals) for all non-diagonal edges present in A_true."""
    mask = (A_true != 0)
    np.fill_diagonal(mask, False)
    return A_true[mask], B_hat[mask]




def compute_mse(B_true, B_hat):
    """
    MSE between B_hat and A_true (the true structural matrix).

    mse_all   : mean((B_hat - B_true)^2) over all off-diagonal entries.
    mse_edges : same but restricted to entries where B_true != 0
                (edge weight recovery quality).
                NaN if B_true has no nonzero off-diagonal entries.
    """
    mask_all = np.ones(B_true.shape, dtype=bool)
    np.fill_diagonal(mask_all, False)
    mse_all = float(np.mean((B_hat[mask_all] - B_true[mask_all]) ** 2))

    mask_edges = (np.abs(B_true) > 0)
    np.fill_diagonal(mask_edges, False)
    if mask_edges.any():
        mse_edges = float(np.mean((B_hat[mask_edges] - B_true[mask_edges]) ** 2))
    else:
        mse_edges = float("nan")

    return {"mse_all": mse_all, "mse_edges": mse_edges}


def compute_graph_metrics(B_true, B_hat):
    """TP/FP/FN/precision/recall/F1/SHD on directed edges."""
    B_true_bin = (np.abs(B_true) > 0).astype(int)
    B_hat_bin = (np.abs(B_hat) > 0).astype(int)
    np.fill_diagonal(B_true_bin, 0)
    np.fill_diagonal(B_hat_bin, 0)

    tp = np.sum((B_true_bin == 1) & (B_hat_bin == 1))
    fp = np.sum((B_true_bin == 0) & (B_hat_bin == 1))
    fn = np.sum((B_true_bin == 1) & (B_hat_bin == 0))
    tn = np.sum((B_true_bin == 0) & (B_hat_bin == 0))

    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)
    shd       = int(np.sum(B_true_bin != B_hat_bin))

    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "shd": shd,
        "n_true_edges": int(np.sum(B_true_bin)),
        "n_pred_edges": int(np.sum(B_hat_bin)),
    }


def compute_graph_metrics_with_reverse(B_true, B_hat):
    """Like compute_graph_metrics but also counts reversed edges."""
    B_true_bin = (np.abs(B_true) > 0).astype(int)
    B_hat_bin  = (np.abs(B_hat)  > 0).astype(int)
    np.fill_diagonal(B_true_bin, 0)
    np.fill_diagonal(B_hat_bin,  0)

    tp = np.sum((B_true_bin == 1) & (B_hat_bin == 1))
    fp = np.sum((B_true_bin == 0) & (B_hat_bin == 1))
    fn = np.sum((B_true_bin == 1) & (B_hat_bin == 0))

    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)
    shd       = int(np.sum(B_true_bin != B_hat_bin))

    n = B_true_bin.shape[0]
    reversed_edges = sum(
        1
        for i in range(n) for j in range(n)
        if i != j
        and B_hat_bin[i, j] == 1
        and B_true_bin[i, j] == 0
        and B_true_bin[j, i] == 1
    )

    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "shd": shd, "reversed": reversed_edges,
    }
