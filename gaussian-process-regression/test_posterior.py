"""
Unit tests for the GP posterior. Light set, aimed at the properties that are
easy to break silently - a transposed cross-covariance still returns the right
shape, and a sign error in the variance still plots as a curve.
"""

from __future__ import annotations

import numpy as np
import pytest

from day1_kernels import rbf_kernel
from day2_posterior import (
    gp_posterior,
    naive_posterior_mean,
    posterior_std,
    sample_posterior,
    true_function,
)


def _toy_data(n: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(n, 1))
    return x, true_function(x).ravel()


def test_posterior_shapes() -> None:
    x, y = _toy_data()
    x_test = np.linspace(-3.0, 3.0, 25)[:, None]
    mean, cov = gp_posterior(x, y, x_test, rbf_kernel, lengthscale=1.0)
    assert mean.shape == (25,)
    assert cov.shape == (25, 25)


def test_posterior_interpolates_noise_free_data() -> None:
    # with essentially no noise the mean has to pass through the observations;
    # this is the check that catches a transposed K* immediately.
    x, y = _toy_data()
    mean, _ = gp_posterior(x, y, x, rbf_kernel, noise_var=1e-10, lengthscale=1.0)
    assert np.allclose(mean, y, atol=1e-4)


def test_variance_collapses_at_training_points() -> None:
    # the residual variance does not go to zero, it goes to noise_var + jitter -
    # the jitter is indistinguishable from noise to the arithmetic, so that sum
    # is the real floor. asserting against a round number instead of the floor
    # just encodes whatever jitter happened to be the default.
    noise_var, jitter = 1e-10, 1e-8
    x, y = _toy_data()
    _, cov = gp_posterior(x, y, x, rbf_kernel, noise_var=noise_var, jitter=jitter,
                          lengthscale=1.0)
    floor = np.sqrt(noise_var + jitter)
    assert np.all(posterior_std(cov) < 1.5 * floor)


def test_posterior_reverts_to_prior_far_from_data() -> None:
    # far outside the data the GP should forget it ever saw anything: mean back
    # to the prior mean of zero, sd back to the prior sd of sqrt(variance).
    x, y = _toy_data()
    x_far = np.array([[60.0]])
    mean, cov = gp_posterior(x, y, x_far, rbf_kernel, lengthscale=1.0, variance=2.0)
    assert abs(mean[0]) < 1e-6
    assert posterior_std(cov)[0] == pytest.approx(np.sqrt(2.0), rel=1e-6)


def test_covariance_is_symmetric_and_psd() -> None:
    x, y = _toy_data()
    x_test = np.linspace(-3.0, 3.0, 20)[:, None]
    _, cov = gp_posterior(x, y, x_test, rbf_kernel, lengthscale=1.0)
    assert np.allclose(cov, cov.T, atol=1e-10)
    # allow a small negative tolerance - the smallest eigenvalues sit at the
    # round-off floor near training points where the true variance is zero.
    assert np.min(np.linalg.eigvalsh(cov)) > -1e-8


def test_observation_band_is_wider_than_latent_band() -> None:
    x, y = _toy_data()
    x_test = np.linspace(-3.0, 3.0, 20)[:, None]
    _, cov = gp_posterior(x, y, x_test, rbf_kernel, noise_var=0.04, lengthscale=1.0)
    latent = posterior_std(cov)
    observed = posterior_std(cov, noise_var=0.04)
    assert np.all(observed > latent)
    assert np.allclose(observed ** 2, latent ** 2 + 0.04)


def test_cholesky_matches_explicit_inverse_when_well_conditioned() -> None:
    x, y = _toy_data()
    x_test = np.linspace(-2.0, 2.0, 15)[:, None]
    mean, _ = gp_posterior(x, y, x_test, rbf_kernel, noise_var=0.1, lengthscale=1.0)
    naive = naive_posterior_mean(x, y, x_test, rbf_kernel, noise_var=0.1,
                                 lengthscale=1.0)
    assert np.allclose(mean, naive, atol=1e-6)


def test_posterior_samples_have_the_right_mean_and_spread() -> None:
    x, y = _toy_data()
    x_test = np.linspace(-3.0, 3.0, 12)[:, None]
    mean, cov = gp_posterior(x, y, x_test, rbf_kernel, lengthscale=1.0)
    draws = sample_posterior(mean, cov, n_samples=4000,
                             rng=np.random.default_rng(0))
    assert draws.shape == (12, 4000)
    assert np.allclose(draws.mean(axis=1), mean, atol=0.05)
    assert np.allclose(draws.std(axis=1), posterior_std(cov), atol=0.05)


def test_more_data_reduces_uncertainty() -> None:
    x_test = np.linspace(-2.0, 2.0, 20)[:, None]
    sds = []
    for n in (4, 40):
        x, y = _toy_data(n=n, seed=1)
        _, cov = gp_posterior(x, y, x_test, rbf_kernel, lengthscale=1.0)
        sds.append(posterior_std(cov).mean())
    assert sds[1] < sds[0]
