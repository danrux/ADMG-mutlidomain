import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F_fn
from torch import nn
try:
    import cooper
except ImportError:  # Optional dependency; only SP needs it.
    cooper = None


from scipy.optimize import linear_sum_assignment


def _compute_mcc_direct(Z_list, Z_hat_list, n):
    """MCC computed directly from g(X) vs true Z, pooled across all domains."""
    Z_all = np.vstack(Z_list)
    Z_hat_all = np.vstack(Z_hat_list)
    cor = np.abs(np.corrcoef(Z_all.T, Z_hat_all.T)[:n, n:])  # (n, n)
    cor = np.nan_to_num(cor, nan=0.0)
    row_ind, col_ind = linear_sum_assignment(-cor)
    mcc = float(cor[row_ind, col_ind].sum() / n)
    return mcc, [mcc], None


def _make_linear_mlp(n_in, n_out, hidden_sizes, output_bn=False, device='cpu'):
    """Deep linear network (no nonlinear activations). bias=False so the effective
    weight matrix can be recovered exactly by passing the identity matrix."""
    layers = []
    prev = n_in
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h, bias=False))
        prev = h
    layers.append(nn.Linear(prev, n_out, bias=False))
    if output_bn:
        layers.append(nn.BatchNorm1d(n_out))
    return nn.Sequential(*layers).to(device)


def _run_optimization(X_list, Z_list, n, xn,
                      num_steps=20000, lr=5e-4, batch_size=6144,
                      sparse_level=0.01, aug_lag_coef=0.0, log_every=250,
                      num_inits=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_all = torch.tensor(np.vstack(X_list), dtype=torch.float32, device=device)

    hidden = [n * 10, n * 10, n * 10, n * 10, n * 10, n * 10]
    #hidden=[]

    best_A_hat_raw = None
    best_loss = float("inf")

    for init_idx in range(num_inits):
        if num_inits > 1:
            print(f"--- Init {init_idx + 1}/{num_inits} ---")

        # g: encoder xn -> n with BN at output (output_normalization="bn" in original)
        g = _make_linear_mlp(xn, n, hidden, output_bn=True, device=device)
        # f_hat: decoder n -> xn, no output BN
        f_hat = _make_linear_mlp(n, xn, hidden, output_bn=False, device=device)

        class _SparsityProblem(cooper.ConstrainedMinimizationProblem):
            def __init__(self):
                super().__init__(is_constrained=True)

            def closure(self, x_batch):
                z_hat = g(x_batch)
                x_hat = f_hat(z_hat)
                loss = F_fn.mse_loss(x_hat, x_batch)
                ineq_defect = z_hat.abs().sum() / x_batch.shape[0] / n - sparse_level
                return cooper.CMPState(loss=loss, ineq_defect=ineq_defect, eq_defect=None)

        cmp = _SparsityProblem()
        formulation = cooper.LagrangianFormulation(cmp, aug_lag_coef)
        params = list(g.parameters()) + list(f_hat.parameters())
        primal_opt = cooper.optim.ExtraAdam(params, lr=lr)
        dual_opt = cooper.optim.partial_optimizer(cooper.optim.ExtraAdam, lr=lr / 2)
        optimizer = cooper.ConstrainedOptimizer(
            formulation=formulation,
            primal_optimizer=primal_opt,
            dual_optimizer=dual_opt,
        )

        N_total = X_all.shape[0]
        final_loss = float("inf")

        for step in range(1, num_steps + 1):
            g.train()
            f_hat.train()
            idx = torch.randint(0, N_total, (min(batch_size, N_total),), device=device)
            x_batch = X_all[idx]
            #x_batch = X_all

            optimizer.zero_grad()
            lagrangian = formulation.composite_objective(cmp.closure, x_batch)
            formulation.custom_backward(lagrangian)
            optimizer.step(cmp.closure, x_batch)

            if step % log_every == 0 or step == num_steps:
                g.eval()
                f_hat.eval()
                with torch.no_grad():
                    z_hat_all = g(X_all)
                    x_hat_all = f_hat(z_hat_all)
                    recon = F_fn.mse_loss(x_hat_all, X_all).item()
                    sparsity = z_hat_all.abs().mean().item() / n
                final_loss = recon
                if Z_list is not None:
                    # MCC from g(X) directly, not via f_hat's matrix
                    Z_hat_list = [g(torch.tensor(X_list[t], dtype=torch.float32, device=device)).detach().cpu().numpy()
                                  for t in range(len(X_list))]
                    maxcor, _, _ = _compute_mcc_direct(Z_list, Z_hat_list, n)
                    print(f"Step {step} — Recon: {recon:.6f}, Sparsity: {sparsity:.4f}, MCC: {maxcor:.4f}")
                else:
                    print(f"Step {step} — Recon: {recon:.6f}, Sparsity: {sparsity:.4f}")

        g.eval()
        f_hat.eval()
        with torch.no_grad():
            A_hat_raw = f_hat(torch.eye(n, device=device)).T.cpu().numpy()  # (xn, n)

        if final_loss < best_loss:
            best_loss = final_loss
            best_A_hat_raw = A_hat_raw
            best_g = g

    encoder_mcc = None
    if Z_list is not None:
        Z_hat_list = [best_g(torch.tensor(X_list[t], dtype=torch.float32, device=device)).detach().cpu().numpy()
                      for t in range(len(X_list))]
        encoder_mcc, _, _ = _compute_mcc_direct(Z_list, Z_hat_list, n)

    return best_A_hat_raw, best_loss, encoder_mcc, best_g


def estimate(X_list, Z_list, _T, n, xn, args):
    """Standard method interface: returns (A_hat_raw, aux_dict)."""
    if cooper is None:
        raise ImportError("SP requires the optional 'cooper' package.")
    A_hat_raw, final_loss, encoder_mcc, best_g = _run_optimization(
        X_list, Z_list, n, xn,
        num_steps=args.num_steps,
        lr=args.lr,
        batch_size=getattr(args, 'batch_size', 6144),
        sparse_level=getattr(args, 'sparse_level', 0.01),
        aug_lag_coef=getattr(args, 'aug_lag_coef', 0.0),
        log_every=getattr(args, 'log_every', 250),
        num_inits=getattr(args, 'num_initializations', 1),
    )
    best_g.eval()
    return A_hat_raw, {"D_hats": None, "sigma_hat": None, "final_loss": final_loss,
                       "encoder_mcc": encoder_mcc, "encoder": best_g}
