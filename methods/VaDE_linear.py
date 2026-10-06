"""Variational Deep Embedding (VaDE) with linear encoder and decoder.

This adapts the model layout from:
https://github.com/mperezcarrasco/Pytorch-VaDE/blob/master/models.py

The original multilayer ReLU networks and Bernoulli decoder are replaced by
linear maps and a Gaussian reconstruction model for the continuous synthetic
data used in this repository.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
from torch import nn


class LinearVaDE(nn.Module):
    """VaDE model whose encoder and decoder are entirely linear."""

    def __init__(self, in_dim, latent_dim, n_classes, full_cov_posterior=False):
        super().__init__()
        if n_classes < 1:
            raise ValueError("n_classes must be positive")

        # Logits are used instead of unconstrained probabilities so the
        # categorical prior always lies on the probability simplex.
        self.pi_logits = nn.Parameter(torch.zeros(n_classes))
        # Small differences break the otherwise exact symmetry between mixture
        # components at initialization.
        self.mu_prior = nn.Parameter(0.05 * torch.randn(n_classes, latent_dim))
        self.log_var_prior = nn.Parameter(0.05 * torch.randn(n_classes, latent_dim))

        self.mu = nn.Linear(in_dim, latent_dim)
        self.full_cov_posterior = full_cov_posterior
        if full_cov_posterior:
            self.covariance_factor = nn.Linear(
                in_dim, latent_dim * (latent_dim + 1) // 2
            )
            rows, columns = torch.tril_indices(latent_dim, latent_dim)
            self.register_buffer("_factor_rows", rows)
            self.register_buffer("_factor_columns", columns)
        else:
            self.log_var = nn.Linear(in_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, in_dim, bias=False)

        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        if full_cov_posterior:
            nn.init.zeros_(self.covariance_factor.weight)
            nn.init.zeros_(self.covariance_factor.bias)
            diagonal = self._factor_rows == self._factor_columns
            with torch.no_grad():
                self.covariance_factor.bias[diagonal] = -1.0
        else:
            nn.init.zeros_(self.log_var.weight)
            nn.init.constant_(self.log_var.bias, -2.0)
        nn.init.xavier_uniform_(self.decoder.weight)

    @property
    def pi_prior(self):
        return torch.softmax(self.pi_logits, dim=0)

    def encode(self, x):
        mu = self.mu(x)
        if not self.full_cov_posterior:
            return mu, self.log_var(x).clamp(-10.0, 10.0)

        entries = self.covariance_factor(x)
        factor = torch.zeros(
            x.shape[0],
            self.mu.out_features,
            self.mu.out_features,
            dtype=x.dtype,
            device=x.device,
        )
        factor[:, self._factor_rows, self._factor_columns] = entries
        diagonal = torch.arange(self.mu.out_features, device=x.device)
        factor[:, diagonal, diagonal] = torch.exp(
            factor[:, diagonal, diagonal].clamp(-5.0, 5.0)
        )
        return mu, factor

    def decode(self, z):
        return self.decoder(z)

    @staticmethod
    def reparameterize(mu, posterior_scale):
        if not torch.is_grad_enabled():
            return mu
        if posterior_scale.ndim == 2:
            std = torch.exp(0.5 * posterior_scale)
            return mu + torch.randn_like(std) * std
        noise = torch.randn_like(mu)
        return mu + torch.bmm(posterior_scale, noise.unsqueeze(2)).squeeze(2)

    def forward(self, x):
        mu, posterior_scale = self.encode(x)
        z = self.reparameterize(mu, posterior_scale)
        x_hat = self.decode(z)
        return x_hat, mu, posterior_scale, z

    def responsibilities(self, mu, posterior_scale):
        """Compute q(c|x) from q(z|x) and the Gaussian-mixture prior."""
        prior_log_var = self.log_var_prior.clamp(-10.0, 10.0)
        prior_var = torch.exp(prior_log_var)
        if posterior_scale.ndim == 2:
            q_var = torch.exp(posterior_scale)
        else:
            q_var = posterior_scale.square().sum(dim=2)

        squared_distance = (mu[:, None, :] - self.mu_prior[None, :, :]).square()
        expected_log_p_z_c = -0.5 * torch.sum(
            prior_log_var[None, :, :]
            + (q_var[:, None, :] + squared_distance) / prior_var[None, :, :],
            dim=2,
        )
        log_pi = torch.log_softmax(self.pi_logits, dim=0)
        return torch.softmax(expected_log_p_z_c + log_pi[None, :], dim=1)


# Keep the class name used by the referenced implementation available.
VaDE = LinearVaDE


def vade_loss(model, x, recon_var=1.0, kl_weight=1.0):
    """Negative VaDE ELBO for a continuous Gaussian observation model."""
    if recon_var <= 0:
        raise ValueError("recon_var must be positive")

    x_hat, mu, posterior_scale, _ = model(x)
    gamma = model.responsibilities(mu, posterior_scale)

    # Constant likelihood terms are omitted because they do not affect
    # optimization. Sum over dimensions, then average over observations.
    reconstruction = 0.5 * (x_hat - x).square().sum(dim=1) / recon_var

    prior_log_var = model.log_var_prior.clamp(-10.0, 10.0)
    prior_var = torch.exp(prior_log_var)
    if posterior_scale.ndim == 2:
        q_var = torch.exp(posterior_scale)
        encoder_entropy_term = -0.5 * torch.sum(
            1.0 + posterior_scale, dim=1
        )
    else:
        q_var = posterior_scale.square().sum(dim=2)
        diagonal = torch.diagonal(posterior_scale, dim1=1, dim2=2)
        log_determinant = 2.0 * torch.log(diagonal).sum(dim=1)
        encoder_entropy_term = -0.5 * (
            posterior_scale.shape[1] + log_determinant
        )
    squared_distance = (mu[:, None, :] - model.mu_prior[None, :, :]).square()

    expected_gaussian_kl = 0.5 * torch.sum(
        gamma[:, :, None]
        * (
            prior_log_var[None, :, :]
            + (q_var[:, None, :] + squared_distance) / prior_var[None, :, :]
        ),
        dim=(1, 2),
    )
    categorical_kl = torch.sum(
        gamma
        * (
            torch.log(gamma.clamp_min(1e-10))
            - torch.log_softmax(model.pi_logits, dim=0)[None, :]
        ),
        dim=1,
    )
    kl = expected_gaussian_kl + encoder_entropy_term + categorical_kl

    loss = (reconstruction + kl_weight * kl).mean()
    return loss, reconstruction.mean(), kl.mean()


def _compute_mcc(Z_list, Z_hat_list, n):
    Z = np.vstack(Z_list)
    Z_hat = np.vstack(Z_hat_list)
    correlations = np.abs(np.corrcoef(Z.T, Z_hat.T)[:n, n:])
    correlations = np.nan_to_num(correlations, nan=0.0)
    rows, cols = linear_sum_assignment(-correlations)
    return float(correlations[rows, cols].mean())


def _pretrain_autoencoder(model, X, num_steps, lr, batch_size):
    """Pretrain the deterministic encoder mean and decoder by reconstruction."""
    if num_steps <= 0:
        return

    parameters = list(model.mu.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    n_samples = X.shape[0]

    for _ in range(num_steps):
        indices = torch.randint(n_samples, (batch_size,), device=X.device)
        x_batch = X[indices]
        optimizer.zero_grad()
        reconstruction = F.mse_loss(model.decode(model.encode(x_batch)[0]), x_batch)
        reconstruction.backward()
        nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
        optimizer.step()


def _initialize_gmm_from_encoder(model, X):
    """Initialize the diagonal mixture prior from deterministic encoder means."""
    with torch.no_grad():
        encoded = model.encode(X)[0].cpu().numpy()

    _, assignments = kmeans2(
        encoded,
        model.mu_prior.shape[0],
        iter=100,
        minit="++",
        missing="raise",
    )

    weights = []
    means = []
    variances = []
    global_variance = np.var(encoded, axis=0) + 1e-4
    for component in range(model.mu_prior.shape[0]):
        members = encoded[assignments == component]
        if len(members) == 0:
            means.append(encoded[np.random.randint(len(encoded))])
            variances.append(global_variance)
            weights.append(1.0 / len(encoded))
        else:
            means.append(np.mean(members, axis=0))
            variances.append(np.var(members, axis=0) + 1e-4)
            weights.append(len(members) / len(encoded))

    weights = np.asarray(weights)
    weights /= weights.sum()
    with torch.no_grad():
        model.pi_logits.copy_(
            torch.as_tensor(np.log(weights), dtype=X.dtype, device=X.device)
        )
        model.mu_prior.copy_(
            torch.as_tensor(np.asarray(means), dtype=X.dtype, device=X.device)
        )
        model.log_var_prior.copy_(
            torch.as_tensor(
                np.log(np.asarray(variances)), dtype=X.dtype, device=X.device
            )
        )


def _fit_once(
    X_list,
    Z_list,
    n,
    xn,
    n_classes,
    num_steps,
    lr,
    batch_size,
    recon_var,
    kl_weight,
    log_every,
    pretrain_steps=0,
    pretrain_lr=1e-3,
    gmm_init=False,
    kl_warmup_steps=0,
    full_cov_posterior=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(np.vstack(X_list), dtype=torch.float32, device=device)
    model = LinearVaDE(
        xn, n, n_classes, full_cov_posterior=full_cov_posterior
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_samples = X.shape[0]
    batch_size = min(max(1, batch_size), n_samples)
    final_loss = float("inf")

    _pretrain_autoencoder(model, X, pretrain_steps, pretrain_lr, batch_size)
    if gmm_init:
        _initialize_gmm_from_encoder(model, X)

    for step in range(1, num_steps + 1):
        model.train()
        indices = torch.randint(n_samples, (batch_size,), device=device)
        x_batch = X[indices]

        optimizer.zero_grad()
        if kl_warmup_steps > 0:
            step_kl_weight = kl_weight * min(1.0, step / kl_warmup_steps)
        else:
            step_kl_weight = kl_weight
        loss, _, _ = vade_loss(model, x_batch, recon_var, step_kl_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        if step % log_every == 0 or step == num_steps:
            model.eval()
            with torch.no_grad():
                full_loss, reconstruction, kl = vade_loss(
                    model, X, recon_var, kl_weight
                )
                final_loss = float(full_loss.item())
                message = (
                    f"Step {step} - Loss: {final_loss:.6f}, "
                    f"Recon: {reconstruction.item():.6f}, KL: {kl.item():.6f}"
                )
                if Z_list is not None:
                    Z_hat_list = [
                        model.encode(
                            torch.as_tensor(x, dtype=torch.float32, device=device)
                        )[0].cpu().numpy()
                        for x in X_list
                    ]
                    message += f", MCC: {_compute_mcc(Z_list, Z_hat_list, n):.4f}"
                print(message)

    model.eval()
    return model, final_loss


def estimate(X_list, Z_list, T, n, xn, args):
    """Fit linear VaDE and return its decoder matrix in the standard interface."""
    n_classes = getattr(args, "vade_components", None)
    n_classes = T if n_classes is None else n_classes
    num_inits = max(1, getattr(args, "num_initializations", 1))

    best_model = None
    best_loss = float("inf")
    for init_idx in range(num_inits):
        if num_inits > 1:
            print(f"Initialization {init_idx + 1}/{num_inits}")
        model, final_loss = _fit_once(
            X_list=X_list,
            Z_list=Z_list,
            n=n,
            xn=xn,
            n_classes=n_classes,
            num_steps=args.num_steps,
            lr=args.lr,
            batch_size=getattr(args, "batch_size", 6144),
            recon_var=getattr(args, "vade_recon_var", 1.0),
            kl_weight=getattr(args, "vade_kl_weight", 1.0),
            log_every=getattr(args, "log_every", 250),
            pretrain_steps=getattr(args, "vade_pretrain_steps", 0),
            pretrain_lr=getattr(args, "vade_pretrain_lr", 1e-3),
            gmm_init=getattr(args, "vade_gmm_init", False),
            kl_warmup_steps=getattr(args, "vade_kl_warmup_steps", 0),
            full_cov_posterior=getattr(
                args, "vade_full_cov_posterior", False
            ),
        )
        if final_loss < best_loss:
            best_model = model
            best_loss = final_loss

    device = next(best_model.parameters()).device
    A_hat_raw = best_model.decoder.weight.detach().cpu().numpy()

    encoder_mcc = None
    if Z_list is not None:
        with torch.no_grad():
            Z_hat_list = [
                best_model.encode(
                    torch.as_tensor(x, dtype=torch.float32, device=device)
                )[0].cpu().numpy()
                for x in X_list
            ]
        encoder_mcc = _compute_mcc(Z_list, Z_hat_list, n)

    # main.py expects an encoder module accepting x and returning z directly.
    best_model.mu.eval()
    return A_hat_raw, {
        "D_hats": None,
        "sigma_hat": None,
        "final_loss": best_loss,
        "encoder_mcc": encoder_mcc,
        "encoder": best_model.mu,
        "model": best_model,
    }
