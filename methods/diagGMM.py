"""Exact marginal-likelihood estimator for a linear decoder and diagonal GMM.

The generative model is

    C ~ Categorical(pi)
    Z | C=c ~ N(mu_c, diag(var_c))
    X = W Z

so the discrete and continuous latent variables can be marginalized exactly:

    p(x) = sum_c pi_c N(x; W mu_c, W diag(var_c) W.T).

A small covariance jitter is added only for numerical Cholesky stability.
"""

import numpy as np
import torch
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
from torch import nn


def _matched_mcc(z_true, z_hat):
    correlation = np.abs(
        np.corrcoef(z_true.T, z_hat.T)[: z_true.shape[1], z_true.shape[1] :]
    )
    correlation = np.nan_to_num(correlation, nan=0.0)
    rows, columns = linear_sum_assignment(-correlation)
    return float(correlation[rows, columns].mean())


class DiagGMM(nn.Module):
    """Linear diagonal-latent GMM evaluated by exact marginal likelihood."""

    def __init__(self, observed, n_components):
        super().__init__()
        if n_components < 1:
            raise ValueError("n_components must be positive")

        observed = np.asarray(observed, dtype=np.float32)
        dimension = observed.shape[1]
        centroids, assignments = kmeans2(
            observed,
            n_components,
            iter=100,
            minit="++",
            missing="raise",
        )

        global_variance = np.var(observed, axis=0) + 1e-3
        variances = []
        weights = []
        for component in range(n_components):
            members = observed[assignments == component]
            if len(members) < 2:
                variances.append(global_variance)
                weights.append(1.0 / len(observed))
            else:
                variances.append(np.var(members, axis=0) + 1e-3)
                weights.append(len(members) / len(observed))

        weights = np.asarray(weights)
        weights /= weights.sum()
        self.decoder = nn.Parameter(torch.eye(dimension))
        self.means = nn.Parameter(torch.as_tensor(centroids, dtype=torch.float32))
        self.log_variances = nn.Parameter(
            torch.log(torch.as_tensor(np.asarray(variances), dtype=torch.float32))
        )
        self.pi_logits = nn.Parameter(
            torch.log(torch.as_tensor(weights, dtype=torch.float32))
        )

    @property
    def pi_prior(self):
        return torch.softmax(self.pi_logits, dim=0)

    @property
    def variances(self):
        return torch.exp(self.log_variances.clamp(-10.0, 10.0))

    def component_parameters(self, covariance_jitter):
        observed_means = self.means @ self.decoder.T
        observed_covariances = torch.einsum(
            "ik,ck,jk->cij",
            self.decoder,
            self.variances,
            self.decoder,
        )
        identity = torch.eye(
            self.decoder.shape[0],
            dtype=self.decoder.dtype,
            device=self.decoder.device,
        )
        observed_covariances = (
            observed_covariances + covariance_jitter * identity[None]
        )
        return observed_means, observed_covariances

    def negative_log_likelihood(self, observed, covariance_jitter=1e-6):
        if covariance_jitter <= 0:
            raise ValueError("covariance_jitter must be positive")

        means, covariances = self.component_parameters(covariance_jitter)
        cholesky = torch.linalg.cholesky(covariances)
        differences = observed[:, None, :] - means[None]
        whitened = torch.linalg.solve_triangular(
            cholesky,
            differences.permute(1, 2, 0),
            upper=False,
        )
        quadratic = whitened.square().sum(dim=1).T
        log_determinant = 2.0 * torch.log(
            torch.diagonal(cholesky, dim1=-2, dim2=-1)
        ).sum(dim=1)
        log_probabilities = (
            -0.5 * (quadratic + log_determinant[None])
            + torch.log_softmax(self.pi_logits, dim=0)[None]
        )
        return -torch.logsumexp(log_probabilities, dim=1).mean()

    def normalize_decoder_columns(self):
        """Fix decoder column norms without changing the represented mixture."""
        with torch.no_grad():
            norms = torch.linalg.vector_norm(self.decoder, dim=0).clamp_min(1e-8)
            self.decoder.div_(norms[None])
            self.means.mul_(norms[None])
            self.log_variances.add_(2.0 * torch.log(norms)[None])


class LinearUnmixing(nn.Module):
    """Deterministic inverse map exposed through the common encoder interface."""

    def __init__(self, decoder):
        super().__init__()
        inverse = np.linalg.pinv(decoder)
        self.linear = nn.Linear(
            inverse.shape[1], inverse.shape[0], bias=False
        )
        with torch.no_grad():
            self.linear.weight.copy_(torch.as_tensor(inverse, dtype=torch.float32))

    def forward(self, observed):
        return self.linear(observed)


def _dataset_nll(model, observed, covariance_jitter, chunk_size=32768):
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(observed), chunk_size):
            chunk = observed[start : start + chunk_size]
            total += (
                model.negative_log_likelihood(chunk, covariance_jitter).item()
                * len(chunk)
            )
    return total / len(observed)


def _fit_once(
    observed,
    n_components,
    num_steps,
    learning_rate,
    batch_size,
    covariance_jitter,
    log_every,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observed_tensor = torch.as_tensor(
        observed, dtype=torch.float32, device=device
    )
    model = DiagGMM(observed, n_components).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    batch_size = min(max(1, batch_size), len(observed_tensor))

    for step in range(1, num_steps + 1):
        indices = torch.randint(
            len(observed_tensor), (batch_size,), device=device
        )
        loss = model.negative_log_likelihood(
            observed_tensor[indices], covariance_jitter
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        if step % log_every == 0 or step == num_steps:
            full_loss = _dataset_nll(
                model, observed_tensor, covariance_jitter
            )
            weights = model.pi_prior.detach().cpu().numpy()
            print(
                f"Step {step} - Exact NLL: {full_loss:.6f}, "
                f"min component weight: {weights.min():.4f}"
            )

    model.normalize_decoder_columns()
    final_loss = _dataset_nll(model, observed_tensor, covariance_jitter)
    model.eval()
    return model, final_loss


def estimate(X_list, Z_list, T, n, xn, args):
    """Fit the exact linear GMM and return its decoder in the common interface."""
    if xn != n:
        raise ValueError(
            "diagGMM currently requires observed and latent "
            f"dimensions to match, but got xn={xn} and n={n}"
        )

    observed = np.vstack(X_list).astype(np.float32)
    n_components = getattr(args, "exact_components", None)
    if n_components is None:
        n_components = T
    covariance_jitter = getattr(args, "exact_jitter", 1e-6)
    batch_size = getattr(args, "batch_size", 6144)
    log_every = getattr(args, "log_every", 250)
    num_initializations = max(1, getattr(args, "num_initializations", 1))

    best_model = None
    best_loss = float("inf")
    for initialization in range(num_initializations):
        if num_initializations > 1:
            print(
                f"Exact GMM initialization "
                f"{initialization + 1}/{num_initializations}"
            )
        model, final_loss = _fit_once(
            observed=observed,
            n_components=n_components,
            num_steps=args.num_steps,
            learning_rate=args.lr,
            batch_size=batch_size,
            covariance_jitter=covariance_jitter,
            log_every=log_every,
        )
        if final_loss < best_loss:
            best_model = model
            best_loss = final_loss

    decoder = best_model.decoder.detach().cpu().numpy()
    encoder = LinearUnmixing(decoder)
    device = next(best_model.parameters()).device
    encoder = encoder.to(device).eval()

    encoder_mcc = None
    if Z_list is not None:
        with torch.no_grad():
            latent = encoder(
                torch.as_tensor(observed, dtype=torch.float32, device=device)
            ).cpu().numpy()
        encoder_mcc = _matched_mcc(np.vstack(Z_list), latent)
        print(f"diagGMM MCC: {encoder_mcc:.4f}")

    return decoder, {
        "D_hats": None,
        "sigma_hat": None,
        "final_loss": best_loss,
        "encoder_mcc": encoder_mcc,
        "encoder": encoder,
        "model": best_model,
        "mixture_weights": best_model.pi_prior.detach().cpu().numpy(),
        "mixture_means": best_model.means.detach().cpu().numpy(),
        "mixture_variances": best_model.variances.detach().cpu().numpy(),
    }
