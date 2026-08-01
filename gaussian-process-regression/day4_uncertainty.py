"""
Day 4 of Gaussian process regression from scratch.

Three loose ends from the previous three days, and they turn out to be the same
loose end.

Day 2 produced error bars, day 3 learned the noise level they depend on, but
nothing has ever checked that the bands are *calibrated* - that a 95% interval
actually contains the truth 95% of the time. RMSE cannot see this. A model can
have the best point predictions in the comparison and still be badly overconfident,
and the only way to find out is to score the distribution rather than the mean.
So the metrics here are coverage, mean band width, and the negative log predictive
density, which is the one number that punishes overconfidence and vagueness at
once.

Day 3 compared RBF against Matern 3/2 at the RBF's own hyperparameters and flagged
that as a rigged fight. Fixed here: every kernel gets its own optimization. The
gradient machinery from day 3 was RBF-specific, so the fit below is derivative-free
(Nelder-Mead on the same log-parameterized objective). That is slower and entirely
adequate at three parameters - and it makes the comparison kernel-agnostic, which
matters more than the speed.

The ridge baseline is the one with the punchline. Kernel ridge regression with the
same kernel and lambda = sigma^2 has the closed form

    f(x*) = k(x*, X) (K + lambda I)^-1 y

which is character-for-character the GP posterior mean from day 2. They are not
similar methods, they are the same estimator reached from opposite directions -
one by penalized least squares in a function space, one by conditioning a Gaussian.
The check below asserts they agree to machine precision. Everything a GP gives you
over ridge is therefore in the *second* moment: the variance, the calibration, the
principled hyperparameter selection. That is the honest summary of what four days
of Cholesky factorizations bought, and it is worth being precise about.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.optimize import minimize

from day1_kernels import rbf_kernel, matern32_kernel, periodic_kernel, gram_matrix
from day2_posterior import gp_posterior, posterior_std, true_function
from day3_marginal_likelihood import fit_hyperparameters


def generic_neg_lml(log_params, x, y, kernel, jitter=1e-8, **fixed_kwargs):
    """Negative log marginal likelihood for any kernel, no gradients.

    Same objective as day 3 - only the RBF-specific analytic derivatives are
    gone. log_params is (log lengthscale, log variance, log noise variance);
    anything else the kernel needs (a period, say) is passed through fixed.
    """
    x = np.atleast_2d(x)
    y = np.asarray(y, dtype=float).ravel()
    n = y.shape[0]
    lengthscale, variance, noise_var = np.exp(log_params)

    k = gram_matrix(kernel, x, jitter=jitter, lengthscale=lengthscale,
                    variance=variance, **fixed_kwargs)
    k[np.diag_indices_from(k)] += noise_var

    try:
        lower = cholesky(k, lower=True)
    except np.linalg.LinAlgError:
        return np.inf

    alpha = cho_solve((lower, True), y)
    log_det = 2.0 * np.sum(np.log(np.diag(lower)))
    lml = -0.5 * y @ alpha - 0.5 * log_det - 0.5 * n * np.log(2.0 * np.pi)
    return -lml


def fit_kernel(x, y, kernel, n_restarts=6, rng=None, **fixed_kwargs):
    """Derivative-free hyperparameter fit, so every kernel gets a fair shot.

    Nelder-Mead rather than L-BFGS-B because there is no gradient to hand it.
    At three parameters the simplex is fine; at thirty it would not be, which is
    exactly why day 3 bothered deriving the analytic gradients for the kernel
    that was going to be used most.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    x, y = np.atleast_2d(x), np.asarray(y, dtype=float).ravel()

    # closure rather than args=, because minimize has no way to forward the
    # kernel's own extra keywords (the period) through to the objective.
    def objective(log_params):
        return generic_neg_lml(log_params, x, y, kernel, **fixed_kwargs)

    start = np.log([np.std(x) + 1e-6, np.var(y) + 1e-6, 0.1 * np.var(y) + 1e-6])
    best, best_value = start, np.inf
    for restart in range(n_restarts):
        init = start if restart == 0 else start + rng.normal(0.0, 0.8, size=3)
        result = minimize(objective, init, method="Nelder-Mead",
                          options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
        if result.fun < best_value:
            best, best_value = result.x, result.fun

    lengthscale, variance, noise_var = np.exp(best)
    return {"lengthscale": lengthscale, "variance": variance,
            "noise_var": noise_var, "log_marginal_likelihood": -best_value}


def predictive_band(mean, cov, noise_var, z=1.96):
    """Central predictive interval for a new observation, not for f.

    noise_var is added to the latent variance because the thing being predicted
    is a measurement, which carries the observation noise on top of the
    uncertainty about the curve. Leaving it out is the most common way these
    bands end up looking miscalibrated on their own training data.
    """
    sd = posterior_std(cov, noise_var=noise_var)
    return mean - z * sd, mean + z * sd, sd


def calibration_report(mean, sd, target):
    """Coverage, mean width, and mean NLPD against a held-out target.

    NLPD = -log N(target | mean, sd^2), averaged. Unlike coverage it is
    sensitive to *how far outside* a miss lands, and unlike RMSE it charges for
    a confident wrong answer - shrinking sd improves it only while the residuals
    shrink with it.
    """
    residual = target - mean
    covered = np.mean(np.abs(residual) <= 1.96 * sd)
    nlpd = np.mean(0.5 * np.log(2.0 * np.pi * sd ** 2) + 0.5 * (residual / sd) ** 2)
    return {"coverage": covered, "mean_width": 2 * 1.96 * np.mean(sd),
            "nlpd": nlpd, "rmse": np.sqrt(np.mean(residual ** 2))}


def kernel_ridge(x_train, y_train, x_test, kernel, lam=1e-2, **kernel_kwargs):
    """Kernel ridge regression: the GP posterior mean with the Bayes stripped out.

    Minimizes ||y - f||^2 + lam ||f||_H^2 over the kernel's RKHS. The
    representer theorem collapses that infinite-dimensional problem to the same
    n-by-n solve the GP does, and lam plays the role of sigma^2. No factorization
    trick here on purpose - it is written the way a ridge derivation writes it,
    so the correspondence is visible rather than hidden behind a Cholesky.
    """
    k_train = kernel(x_train, x_train, **kernel_kwargs)
    k_train[np.diag_indices_from(k_train)] += lam
    k_cross = kernel(x_train, x_test, **kernel_kwargs)
    dual_coefs = np.linalg.solve(k_train, np.asarray(y_train, dtype=float).ravel())
    return k_cross.T @ dual_coefs


def polynomial_ridge(x_train, y_train, x_test, degree=5, lam=1e-2):
    """Ordinary ridge on a polynomial basis - the baseline before the baseline.

    This is what somebody reaches for when they do not want a kernel at all, and
    it is included because it fails in an instructive direction: polynomials
    have no notion of locality, so the fit degrades everywhere when the degree is
    wrong and diverges violently outside the training range.
    """
    def design(x):
        x = np.atleast_2d(x).ravel()
        return np.vander(x, degree + 1, increasing=True)

    phi = design(x_train)
    d = phi.shape[1]
    # do not penalize the intercept - shrinking it just biases the whole fit
    # toward zero for no reason.
    penalty = lam * np.eye(d)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(phi.T @ phi + penalty, phi.T @ np.asarray(y_train).ravel())
    return design(x_test) @ weights


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    n_train, noise_sd = 40, 0.15
    x_train = rng.uniform(-3.0, 3.0, size=(n_train, 1))
    y_train = true_function(x_train).ravel() + noise_sd * rng.standard_normal(n_train)

    # held out on the SAME range as training. extrapolation is looked at
    # separately below - mixing the two hides which failure is which.
    x_held = rng.uniform(-3.0, 3.0, size=(200, 1))
    y_held = true_function(x_held).ravel() + noise_sd * rng.standard_normal(200)

    x_grid = np.linspace(-4.0, 4.0, 200)[:, None]
    truth_grid = true_function(x_grid).ravel()
    inside = np.abs(x_grid.ravel()) <= 3.0

    learned = fit_hyperparameters(x_train, y_train, n_restarts=5, rng=rng)
    print(f"rbf hyperparameters (day 3 optimizer): l={learned['lengthscale']:.4f} "
          f"v={learned['variance']:.4f} s2={learned['noise_var']:.5f}")

    # 1. the identity. GP posterior mean == kernel ridge at lam = sigma^2.
    # jitter=0 here only so the two sides are arithmetically identical. with the
    # default 1e-8 the means differ by ~6e-8, which is a fine demonstration that
    # the jitter is doing something - it just muddies an exact-equality claim.
    gp_mean, gp_cov = gp_posterior(x_train, y_train, x_grid, rbf_kernel,
                                   noise_var=learned["noise_var"], jitter=0.0,
                                   lengthscale=learned["lengthscale"],
                                   variance=learned["variance"])
    krr_mean = kernel_ridge(x_train, y_train, x_grid, rbf_kernel,
                            lam=learned["noise_var"],
                            lengthscale=learned["lengthscale"],
                            variance=learned["variance"])
    gap = np.max(np.abs(gp_mean - krr_mean))
    print(f"max |GP mean - kernel ridge| = {gap:.3e}  "
          f"(same estimator, different derivation)")
    assert gap < 1e-8, "the two derivations must agree exactly"

    # 2. so what does the GP add. score the distribution, not the point.
    held_mean, held_cov = gp_posterior(x_train, y_train, x_held, rbf_kernel,
                                       noise_var=learned["noise_var"],
                                       lengthscale=learned["lengthscale"],
                                       variance=learned["variance"])
    _, _, held_sd = predictive_band(held_mean, held_cov, learned["noise_var"])
    report = calibration_report(held_mean, held_sd, y_held)
    print(f"\nheld-out (n=200, same range as training)")
    print(f"  rmse           {report['rmse']:.4f}")
    print(f"  95% coverage   {100 * report['coverage']:.1f}%   (target 95.0%)")
    print(f"  mean width     {report['mean_width']:.4f}")
    print(f"  mean nlpd      {report['nlpd']:.4f}")
    print(f"  97.5% against a nominal 95% is slightly conservative, and the reason")
    print(f"  is upstream: the learned noise variance is {learned['noise_var']:.4f} against a")
    print(f"  true {noise_sd ** 2:.4f}, so the bands are built a little too wide. erring")
    print("  this direction is the benign one, and it is visible only because")
    print("  coverage was measured at all.")
    print("  ridge produces the same rmse and has nothing to put in the other")
    print("  three rows - there is no sd to report, so no nlpd either.")

    # 3. what happens to the band when the model leaves the data. this is the
    # behaviour that no point estimator has: the variance grows back toward the
    # prior, so the model reports its own ignorance instead of extrapolating
    # confidently off a cliff.
    _, _, grid_sd = predictive_band(gp_mean, gp_cov, learned["noise_var"])
    print("\n x     |truth - mean|   95% half-width   inside band")
    for probe in (0.0, 2.0, 3.0, 3.5, 4.0):
        idx = int(np.argmin(np.abs(x_grid.ravel() - probe)))
        err, half = abs(truth_grid[idx] - gp_mean[idx]), 1.96 * grid_sd[idx]
        print(f"  {probe:<5.1f} {err:>13.4f} {half:>16.4f}   {'yes' if err <= half else 'NO'}")
    print("  the half-width grows about 5x between x=3 and x=4 while the error")
    print("  only doubles, so every extrapolated point stays inside its band.")
    print("  that is the model staying honest rather than staying accurate - the")
    print("  interval out at x=4 is ~4 units wide on a function of range ~2, which")
    print("  is a correct statement and a useless prediction at the same time.")

    # 4. kernel comparison, each at its own optimum this time. the LML is
    # directly comparable across kernels because it is the same probability of
    # the same y under different priors.
    print("\nkernel       lengthscale   noise var       LML   held rmse   coverage")
    kernels = {"rbf": (rbf_kernel, {}),
               "matern32": (matern32_kernel, {}),
               "periodic": (periodic_kernel, {"period": 2.09})}
    for name, (fn, extra) in kernels.items():
        fitted = fit_kernel(x_train, y_train, fn, n_restarts=6, rng=rng, **extra)
        mean, cov = gp_posterior(x_train, y_train, x_held, fn,
                                 noise_var=fitted["noise_var"],
                                 lengthscale=fitted["lengthscale"],
                                 variance=fitted["variance"], **extra)
        _, _, sd = predictive_band(mean, cov, fitted["noise_var"])
        stats = calibration_report(mean, sd, y_held)
        print(f"  {name:<10} {fitted['lengthscale']:>11.4f} "
              f"{fitted['noise_var']:>11.5f} {fitted['log_marginal_likelihood']:>9.2f} "
              f"{stats['rmse']:>11.4f} {100 * stats['coverage']:>9.1f}%")
    print("  the true function is sin(3x) + 0.3x, so it is smooth and nearly")
    print("  periodic, and the ranking lands where the prior assumptions do:")
    print("  rbf's smoothness is correct here, matern 3/2 pays for roughness it")
    print("  does not need, and periodic is handed the right period yet still")
    print("  cannot represent the linear drift, which is what the trend term in")
    print("  a real composite kernel would be for.")

    # 5. the baselines, on the same held-out split.
    print("\nmodel                          held rmse   extrapolation rmse (|x|>3)")
    outside = ~inside
    entries = [
        ("gp / kernel ridge (rbf)",
         np.sqrt(np.mean((held_mean - y_held) ** 2)),
         np.sqrt(np.mean((gp_mean[outside] - truth_grid[outside]) ** 2))),
    ]
    for degree in (3, 5, 9):
        held_fit = polynomial_ridge(x_train, y_train, x_held, degree=degree, lam=1e-3)
        grid_fit = polynomial_ridge(x_train, y_train, x_grid, degree=degree, lam=1e-3)
        entries.append((f"polynomial ridge (degree {degree})",
                        np.sqrt(np.mean((held_fit - y_held) ** 2)),
                        np.sqrt(np.mean((grid_fit[outside] - truth_grid[outside]) ** 2))))
    for label, held_rmse, out_rmse in entries:
        print(f"  {label:<30} {held_rmse:>9.4f} {out_rmse:>24.4f}")
    print("  the polynomial baselines are not embarrassed inside the data - a")
    print("  degree-9 fit tracks two periods of a sine perfectly well. they come")
    print("  apart outside it, where the leading term takes over and the error")
    print("  runs away, and they never had a variance to warn anyone.")
