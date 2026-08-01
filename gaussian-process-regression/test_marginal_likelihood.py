"""
Tests for the log marginal likelihood and its analytic gradient.

The gradient is the part worth pinning down. A sign or chain-rule slip in it
does not raise - the optimizer just walks somewhere slightly wrong and the fit
comes out mediocre, which looks exactly like a hard dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from day3_marginal_likelihood import (
    check_gradients,
    fit_hyperparameters,
    lml_terms,
    neg_log_marginal_likelihood,
)
from day2_posterior import true_function


def _toy_data(n: int = 15, noise_sd: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(n, 1))
    y = true_function(x).ravel() + noise_sd * rng.standard_normal(n)
    return x, y


@pytest.mark.parametrize(
    "log_params",
    [
        np.log([0.7, 1.0, 0.01]),
        np.log([0.2, 2.5, 0.1]),   # short lengthscale, near-identity K
        np.log([3.0, 0.5, 0.001]), # long lengthscale, badly conditioned K
    ],
)
def test_analytic_gradient_matches_finite_differences(log_params) -> None:
    x, y = _toy_data()
    analytic, numeric = check_gradients(log_params, x, y)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_lml_terms_sum_to_the_objective() -> None:
    """The fit/complexity/constant split must reconstruct the whole LML.

    Guards the Occam's-razor breakdown against drifting out of sync with the
    objective the optimizer actually sees - they compute log|K_y| by different
    routes and only agree if both are right.
    """
    x, y = _toy_data()
    log_params = np.log([0.6, 1.2, 0.02])

    neg_lml, _ = neg_log_marginal_likelihood(log_params, x, y)
    assert sum(lml_terms(log_params, x, y)) == pytest.approx(-neg_lml, rel=1e-9)


def test_fit_recovers_a_sane_noise_level() -> None:
    """Learned hyperparameters should be in the right ballpark, not exact.

    Deliberately loose: the noise estimate is biased low at these sample sizes
    because some noise is cheaper to explain as a wigglier function. The test
    is that it stays within an order of magnitude, which catches a broken
    objective while tolerating the real statistical bias.
    """
    noise_sd = 0.1
    x, y = _toy_data(n=60, noise_sd=noise_sd)
    fitted = fit_hyperparameters(x, y, n_restarts=3)

    assert 0.1 * noise_sd ** 2 < fitted["noise_var"] < 10.0 * noise_sd ** 2
    # sin(3x) turns over about every ~1.0 in x, so anything outside this is a
    # model that has either given up or started chasing the noise.
    assert 0.1 < fitted["lengthscale"] < 3.0
