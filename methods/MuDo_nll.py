import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from evaluate import _permute_and_normalize, compute_mcc


def _compute_M(B, sigma_vec, n, device):
    """Shared helper: M = (I-B)^{-1} @ Sigma^2 @ (I-B)^{-T}"""
    I = torch.eye(n, device=device)
    I_B_inv = torch.inverse(I - B)
    Sigma = torch.diag(sigma_vec)
    return I_B_inv @ (Sigma @ Sigma) @ I_B_inv.T


def cov_loss_torch(A_hat, D_flat, sigma_vec, B, Cov_list, T, n):
    """
    Covariance-matching loss: sum_t ||S_t - A D_t M D_t A^T||_F^2
    """
    device = A_hat.device
    M = _compute_M(B, sigma_vec, n, device)

    Cov_tensor = torch.stack([torch.tensor(c, dtype=torch.float32, device=device) for c in Cov_list])
    D_tensor = D_flat.view(T, n)

    dM_batch = D_tensor[:, :, None] * M[None, :, :] * D_tensor[:, None, :]
    ADAT_batch = A_hat @ dM_batch @ A_hat.T

    diff = Cov_tensor - ADAT_batch
    return torch.sum(diff ** 2)


def nll_loss_torch(A_hat, D_flat, sigma_vec, B, Cov_list, T, n, eps=1e-6):
    """
    Gaussian negative log-likelihood loss (up to a constant):
        sum_t [ log|Sigma_t| + tr(Sigma_t^{-1} S_t) ]
    where Sigma_t = A D_t M D_t A^T and S_t is the sample covariance.
    """
    device = A_hat.device
    M = _compute_M(B, sigma_vec, n, device)
    D_tensor = D_flat.view(T, n)

    xn = A_hat.shape[0]
    reg = eps * torch.eye(xn, device=device)

    loss = torch.zeros(1, device=device).squeeze()
    for t in range(T):
        D_t = torch.diag(D_tensor[t])
        Sigma_t = A_hat @ D_t @ M @ D_t @ A_hat.T + reg

        _, logdet = torch.linalg.slogdet(Sigma_t)
        S_t = torch.tensor(Cov_list[t], dtype=torch.float32, device=device)
        trace_term = torch.trace(torch.linalg.solve(Sigma_t, S_t))

        loss = loss + logdet + trace_term

    return loss


def run_torch_optimization(Cov_list, X_list, Z_list, T, n, xn,
                           num_steps=2000, lr=1e-3, loss_type="cov",
                           scheduler="cosine", grad_clip=1.0,
                           patience=2000, min_delta=1e-6,
                           checkpoint_policy="best"):
    """
    scheduler : "cosine" | "plateau" | "none"
    grad_clip : max gradient norm (0 = disabled)
    patience  : early-stopping patience in steps (0 = disabled)
    min_delta : minimum loss improvement to reset patience counter
    checkpoint_policy : "best" restores the lowest-loss evaluated state;
                        "final" reproduces the original post-update return

    Note: Z_list contains ground-truth latent variables and is used only for
    diagnostic MCC logging during training. It does not affect gradients or
    the loss. Pass Z_list=None to disable MCC logging (e.g. in deployment).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    A_hat = nn.Parameter(torch.rand(xn, n, dtype=torch.float32, device=device))
    # Raw params for D and sigma; softplus is applied before the loss so they stay positive
    D_raw = nn.Parameter(torch.rand(n * T, dtype=torch.float32, device=device) * 0.8 + 0.2)
    sigma_raw = nn.Parameter(torch.rand(n, dtype=torch.float32, device=device) * 0.8 + 0.2)
    B_free = nn.Parameter(torch.zeros((n, n), dtype=torch.float32, device=device))

    optimizer = Adam([A_hat, D_raw, sigma_raw, B_free], lr=lr)

    if scheduler == "cosine":
        sched = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr * 1e-2)
    elif scheduler == "plateau":
        sched = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience // 4)
    else:
        sched = None

    loss_fn = nll_loss_torch if loss_type == "nll" else cov_loss_torch

    best_loss = float("inf")
    best_step = None
    best_state = None
    last_loss = None
    steps_without_improvement = 0

    for step in range(num_steps):
        optimizer.zero_grad()

        B = B_free.clone()
        B.fill_diagonal_(0.0)

        # Enforce positivity for scale parameters
        D_flat = F.softplus(D_raw)
        sigma_hat_vec = F.softplus(sigma_raw)

        loss = loss_fn(A_hat, D_flat, sigma_hat_vec, B, Cov_list, T, n)
        loss_value = float(loss.item())
        if not np.isfinite(loss_value):
            raise FloatingPointError(f"MuDo-nll loss became non-finite at step {step}")

        # The loss describes the parameters before optimizer.step().  Snapshot
        # that exact state so the returned checkpoint and reported objective
        # cannot get out of sync.
        improved = loss_value < best_loss - min_delta
        if improved:
            best_loss = loss_value
            best_step = step
            best_state = {
                "A_hat": A_hat.detach().cpu().clone(),
                "D_flat": F.softplus(D_raw).detach().cpu().clone(),
                "sigma_vec": F.softplus(sigma_raw).detach().cpu().clone(),
                "B": B.detach().cpu().clone(),
            }
        last_loss = loss_value

        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_([A_hat, D_raw, sigma_raw, B_free], grad_clip)

        optimizer.step()

        if sched is not None:
            if scheduler == "plateau":
                sched.step(loss.item())
            else:
                sched.step()

        if step % 1000 == 0 or step == num_steps - 1:
            if Z_list is not None:
                maxcor, _, _ = compute_mcc(A_hat, X_list, Z_list, n)
                print(f"Step {step} — Loss: {loss.item():.6f}, MCC: {maxcor:.4f}")
            else:
                print(f"Step {step} — Loss: {loss.item():.6f}")

        # Early stopping
        if patience > 0:
            if improved:
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
                if steps_without_improvement >= patience:
                    print(f"Early stopping at step {step} (no improvement for {patience} steps)")
                    break

    if checkpoint_policy not in {"best", "final"}:
        raise ValueError(f"Unknown checkpoint policy: {checkpoint_policy!r}")
    if best_state is None:
        raise RuntimeError("MuDo-nll optimization did not produce a finite checkpoint")

    if checkpoint_policy == "best":
        returned_state = best_state
        returned_loss = best_loss
        returned_step = best_step
    else:
        # This matches the original implementation: parameters are read after
        # the final optimizer update, while the reported loss was evaluated
        # immediately before that update.
        returned_state = {
            "A_hat": A_hat.detach().cpu(),
            "D_flat": F.softplus(D_raw).detach().cpu(),
            "sigma_vec": F.softplus(sigma_raw).detach().cpu(),
            "B": B_free.detach().cpu(),
        }
        returned_loss = last_loss
        returned_step = step

    A_hat_opt = returned_state["A_hat"].numpy()
    D_flat_np = returned_state["D_flat"].numpy()
    D_hats = [D_flat_np[i * n:(i + 1) * n] for i in range(T)]
    sigma_hat = np.diag(returned_state["sigma_vec"].numpy())
    B_hat = returned_state["B"].numpy()
    np.fill_diagonal(B_hat, 0.0)

    optimization = {
        "best_loss": float(best_loss),
        "last_loss": float(last_loss),
        "best_step": int(best_step),
        "returned_loss": float(returned_loss),
        "returned_step": int(returned_step),
        "checkpoint_policy": checkpoint_policy,
        "steps_run": int(step + 1),
    }
    return A_hat_opt, D_hats, sigma_hat, B_hat, optimization


def _run_multiple_initializations(Cov_list, X_list, Z_list, T, n, xn,
                                  num_steps=2000, lr=1e-3, num_initializations=5,
                                  loss_type="cov", scheduler="cosine",
                                  grad_clip=1.0, patience=2000,
                                  min_normalizing_diagonal=0.0,
                                  validate_normalization=True,
                                  checkpoint_policy="best"):
    """
    Z_list: ground-truth latents, forwarded to run_torch_optimization for diagnostic
    MCC logging only. Pass None to suppress MCC logging.
    """
    best_loss = np.inf
    best_result = None
    best_initialization = None
    initialization_summaries = []

    for i in range(num_initializations):
        print(f"Initialization {i + 1}/{num_initializations}")
        A_hat, D_hats, sigma_hat, B_hat, optimization = run_torch_optimization(
            Cov_list, X_list, Z_list, T, n, xn,
            num_steps=num_steps, lr=lr, loss_type=loss_type,
            scheduler=scheduler, grad_clip=grad_clip, patience=patience,
            checkpoint_policy=checkpoint_policy,
        )
        summary = {"initialization": i, **optimization}
        if not validate_normalization:
            valid = True
            summary.update({
                "normalization_validation_enabled": False,
                "decoder_condition_number": None,
                "minimum_normalizing_diagonal": None,
                "normalization_valid": True,
            })
        else:
            summary["normalization_validation_enabled"] = True
            try:
                decoder_condition = float(np.linalg.cond(A_hat))
                _, _, diagonal = _permute_and_normalize(np.linalg.inv(A_hat))
                minimum_diagonal = float(np.min(np.abs(diagonal)))
                valid = (
                    np.isfinite(decoder_condition)
                    and minimum_diagonal >= min_normalizing_diagonal
                )
                summary.update({
                    "decoder_condition_number": decoder_condition,
                    "minimum_normalizing_diagonal": minimum_diagonal,
                    "normalization_valid": bool(valid),
                })
                if not valid:
                    summary["rejection_reason"] = (
                        f"minimum normalizing diagonal {minimum_diagonal:.6g} "
                        f"is below the required {min_normalizing_diagonal:.6g}"
                    )
            except (np.linalg.LinAlgError, ValueError) as exc:
                valid = False
                summary.update({
                    "decoder_condition_number": None,
                    "minimum_normalizing_diagonal": None,
                    "normalization_valid": False,
                    "rejection_reason": f"{type(exc).__name__}: {exc}",
                })
        initialization_summaries.append(summary)

        selection_loss = optimization.get("returned_loss", optimization["best_loss"])
        if valid and selection_loss < best_loss:
            best_loss = selection_loss
            best_initialization = i
            best_result = (A_hat, D_hats, sigma_hat, B_hat, optimization)

    if best_result is None:
        reasons = "; ".join(
            f"restart {row['initialization']}: {row.get('rejection_reason', 'invalid')}"
            for row in initialization_summaries
        )
        raise np.linalg.LinAlgError(
            "No MuDo-nll initialization passed decoder normalization validation. "
            + reasons
        )

    return (*best_result, {
        "selected_initialization": int(best_initialization),
        "num_initializations_attempted": int(num_initializations),
        "num_initializations_valid": int(sum(
            row["normalization_valid"] for row in initialization_summaries
        )),
        "initialization_summaries": initialization_summaries,
    })


def estimate(X_list, Z_list, T, n, xn, args):
    """Standard method interface: returns (A_hat_raw, aux_dict)."""
    Cov_list = [np.cov(X_list[t], rowvar=False) for t in range(T)]
    A_hat_raw, D_hats, sigma_hat, B_hat, optimization, restart_diagnostics = _run_multiple_initializations(
        Cov_list, X_list, Z_list, T, n, xn,
        num_steps=args.num_steps, lr=args.lr,
        num_initializations=args.num_initializations,
        loss_type=args.loss_type, scheduler=args.scheduler,
        grad_clip=args.grad_clip, patience=args.patience,
        min_normalizing_diagonal=getattr(args, "min_normalizing_diagonal", 0.0),
        validate_normalization=getattr(args, "validate_normalization", True),
        checkpoint_policy=getattr(args, "checkpoint_policy", "best"),
    )
    return A_hat_raw, {
        "D_hats": D_hats,
        "sigma_hat": sigma_hat,
        "B_hat_internal": B_hat,
        # Keep final_loss for consumers of the legacy estimator interface. It
        # now deliberately names the selected restart's best checkpoint loss.
        "final_loss": optimization.get("returned_loss", optimization["best_loss"]),
        **optimization,
        **restart_diagnostics,
    }
