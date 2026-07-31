"""
Day 2 of Gaussian process regression from scratch.

Yesterday built the prior: a kernel, a Gram matrix, and function draws that had
never seen data. Today the data arrives and the prior gets conditioned into a
posterior, which for a GP is a closed-form operation rather than an optimization.

The whole thing rests on one Gaussian identity. Stack the training values f(X)
and the test values f(X*) into a single joint Gaussian - which the GP definition
guarantees they are - and the conditional f(X*) | y is Gaussian too, with

    mean = K*^T (K + sigma^2 I)^-1 y
    cov  = K** - K*^T (K + sigma^2 I)^-1 K*

That is the entire model. No iterations, no gradient descent (that lands
tomorrow, for the hyperparameters). Note the covariance does not depend on y at
all: where the GP is uncertain is decided purely by where the inputs are, so the
error bars can be drawn before a single observation is read.

The implementation question is how to apply that inverse, and the answer is
never to form it. Cholesky-factor K + sigma^2 I once into L L^T and reuse L for
both the mean solve and the variance solve. That is ~n^3/3 flops instead of n^3,
it is numerically far better behaved, and it makes the O(n^3) training / O(n^2)
per-test-point cost of exact GP inference explicit - the reason exact GPs stop
at a few thousand points.

The noise variance also earns its keep numerically. It occupies the same
diagonal slot as yesterday's jitter, so a noisy problem is *better* conditioned
than a noiseless one: sigma^2 lifts every eigenvalue away from zero. Noise-free
interpolation is the hard case, not the easy one.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from day1_kernels import (
    rbf_kernel,
    matern32_kernel,
    gram_matrix,
    condition_number,
)


def gp_posterior(x_train, y_train, x_test, kernel, noise_var=1e-2, jitter=1e-8,
                 **kernel_kwargs):
    """Exact GP posterior mean and covariance at x_test.

    Returns (mean, cov) for the latent function f, excluding observation noise.
    The covariance is the full test-by-test matrix, which is what allows joint
    posterior sample paths later - the per-point variances on its diagonal are
    enough for error bars but not for drawing coherent functions.
    """
    x_train = np.atleast_2d(x_train)
    x_test = np.atleast_2d(x_test)
    y_train = np.asarray(y_train, dtype=float).ravel()

    # K + sigma^2 I. the jitter and the noise land in the same place; the noise
    # is a modelling statement and the jitter is an admission about floats, but
    # the arithmetic cannot tell them apart.
    k_train = gram_matrix(kernel, x_train, jitter=jitter, **kernel_kwargs)
    k_train[np.diag_indices_from(k_train)] += noise_var

    k_cross = kernel(x_train, x_test, **kernel_kwargs)   # K*  (n_train, n_test)
    k_test = kernel(x_test, x_test, **kernel_kwargs)     # K** (n_test, n_test)

    # one factorization, reused twice. cho_factor returns (L, lower) and
    # cho_solve applies both triangular solves for the mean.
    chol = cho_factor(k_train, lower=True)
    alpha = cho_solve(chol, y_train)                     # (K + s^2 I)^-1 y
    mean = k_cross.T @ alpha

    # for the covariance only the FORWARD solve is needed: with v = L^-1 K*,
    # K*^T (LL^T)^-1 K* is just v^T v. doing it this way keeps the subtracted
    # term manifestly positive semi-definite instead of relying on cancellation.
    lower = chol[0]
    v = solve_triangular(lower, k_cross, lower=True)
    cov = k_test - v.T @ v

    return mean, cov


def posterior_std(cov, noise_var=0.0):
    """Marginal predictive standard deviations from a posterior covariance.

    Passing noise_var gives the predictive interval for a new *observation*
    rather than for the latent function - the difference between "where is the
    curve" and "where will the next measurement land", which is the wider band.
    Tiny negatives can appear on the diagonal from round-off near training
    points where the true variance is ~0, so clip before the sqrt.
    """
    var = np.diag(cov).copy()
    var = np.maximum(var, 0.0) + noise_var
    return np.sqrt(var)


def sample_posterior(mean, cov, n_samples=5, jitter=1e-8, rng=None):
    """Draw joint function samples from the posterior.

    Same Cholesky trick as the prior draws, but the posterior covariance is
    close to singular by construction - it collapses to zero at every training
    point - so it needs its own jitter to factor at all.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n = mean.shape[0]
    chol = np.linalg.cholesky(cov + jitter * np.eye(n))
    u = rng.standard_normal(size=(n, n_samples))
    return mean[:, None] + chol @ u


def naive_posterior_mean(x_train, y_train, x_test, kernel, noise_var=1e-2,
                         **kernel_kwargs):
    """The same mean via an explicit matrix inverse - for contrast only.

    Kept to make the point that np.linalg.inv agrees with the Cholesky path on
    well-conditioned problems and drifts away on badly conditioned ones, which
    is the practical argument for never writing the inverse in the first place.
    """
    k_train = kernel(x_train, x_train, **kernel_kwargs)
    k_train = k_train + noise_var * np.eye(k_train.shape[0])
    k_cross = kernel(x_train, x_test, **kernel_kwargs)
    return k_cross.T @ np.linalg.inv(k_train) @ np.asarray(y_train).ravel()


def true_function(x):
    """The generating function used for all the demos below."""
    return np.sin(3.0 * x) + 0.3 * x


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    n_train, noise_sd = 12, 0.15
    x_train = rng.uniform(-3.0, 3.0, size=(n_train, 1))
    y_train = true_function(x_train).ravel() + noise_sd * rng.standard_normal(n_train)
    x_test = np.linspace(-4.0, 4.0, 200)[:, None]

    mean, cov = gp_posterior(x_train, y_train, x_test, rbf_kernel,
                             noise_var=noise_sd ** 2, lengthscale=0.7, variance=1.0)
    sd_latent = posterior_std(cov)
    sd_observed = posterior_std(cov, noise_var=noise_sd ** 2)

    print(f"posterior mean {mean.shape}, cov {cov.shape}")

    # score only inside the training range - the [-4, -3] and [3, 4] tails are
    # extrapolation and would dominate the error for reasons that say nothing
    # about the fit.
    truth = true_function(x_test).ravel()
    inside = np.abs(x_test.ravel()) <= 3.0
    print(f"rmse vs truth on [-3, 3]: "
          f"{np.sqrt(np.mean((mean[inside] - truth[inside]) ** 2)):.4f}")
    print(f"rmse including extrapolation: "
          f"{np.sqrt(np.mean((mean - truth) ** 2)):.4f}")

    # the posterior should nearly interpolate the data and revert to the prior
    # far from it. these two numbers are the entire behaviour of a GP.
    near = np.argmin(np.abs(x_test - x_train[0, 0]))
    print(f"\nsd at a training point : {sd_latent[near]:.4f}")
    print(f"sd at the far edge x=4 : {sd_latent[-1]:.4f}  (prior sd = 1.0)")
    print(f"observation band is wider by sqrt(sd^2 + s^2): "
          f"{sd_observed[-1]:.4f} vs {sd_latent[-1]:.4f}")

    # extrapolation is the honest failure mode - the mean decays to the prior
    # mean of zero rather than continuing the trend, and says so via the sd.
    print(f"\nmean at x=4 (outside the data): {mean[-1]:.4f}, "
          f"truth {true_function(4.0):.4f}")

    # rougher kernel, same data: less smoothing, wider bands between points.
    mean_m, cov_m = gp_posterior(x_train, y_train, x_test, matern32_kernel,
                                 noise_var=noise_sd ** 2, lengthscale=0.7)
    print(f"\nmatern32 mean sd across test set: {posterior_std(cov_m).mean():.4f}")
    print(f"rbf      mean sd across test set: {sd_latent.mean():.4f}")

    draws = sample_posterior(mean, cov, n_samples=5, rng=rng)
    print(f"\nposterior draws {draws.shape}, spread at x=4: {draws[-1].std():.4f}")

    # the fit above is mediocre and that is worth being explicit about: sin(3x)
    # has period ~2.1 and 12 points over [-3, 3] leaves gaps of ~0.5, so the
    # posterior smooths across wiggles it never saw. more data fixes it, and at
    # n=12 the hand-picked lengthscale is also wrong - which is the argument for
    # learning it instead of guessing, i.e. tomorrow.
    print("\nn_train  rmse(l=0.7)  rmse(l=0.3)")
    for n in (12, 30, 80):
        row = []
        for ls in (0.7, 0.3):
            xs = rng.uniform(-3.0, 3.0, size=(n, 1))
            ys = true_function(xs).ravel() + noise_sd * rng.standard_normal(n)
            m, _ = gp_posterior(xs, ys, x_test, rbf_kernel,
                                noise_var=noise_sd ** 2, lengthscale=ls)
            row.append(np.sqrt(np.mean((m[inside] - truth[inside]) ** 2)))
        print(f"  {n:<6d} {row[0]:<12.4f} {row[1]:.4f}")

    # why the factorization is not optional: shrink the noise and the system
    # degrades fast. cholesky still tracks the truth long after inv() drifts.
    print("\nnoise_var   cond(K)     max|cholesky - inv|")
    for nv in (1e-1, 1e-4, 1e-8):
        k = gram_matrix(rbf_kernel, x_train, jitter=0.0, lengthscale=0.7)
        k[np.diag_indices_from(k)] += nv
        m_chol, _ = gp_posterior(x_train, y_train, x_test, rbf_kernel,
                                 noise_var=nv, lengthscale=0.7)
        m_inv = naive_posterior_mean(x_train, y_train, x_test, rbf_kernel,
                                     noise_var=nv, lengthscale=0.7)
        print(f"  {nv:<9.0e} {condition_number(k):<11.2e} "
              f"{np.max(np.abs(m_chol - m_inv)):.3e}")
