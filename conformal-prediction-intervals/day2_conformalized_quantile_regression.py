"""
Day 2 of conformal prediction from scratch.

Day 1 ended on a specific failure. Split conformal with absolute-residual scores
produced a band of one fixed width, its marginal coverage landed on nominal
exactly as promised, and its coverage broken down by |x| was badly wrong in
every bin - far above nominal where the noise is small, far below it where the
noise is large. Nothing was broken. Marginal validity is all that was ever
claimed, and a procedure can satisfy it while being wrong everywhere in a way
that averages out.

Conformalized quantile regression (Romano, Patterson, Candes 2019) is the fix
that keeps the guarantee. The idea is to stop asking the model for a point
prediction and start asking it for an interval, then conformalize *that*:

  1. fit two quantile regressions on the training split, at alpha/2 and
     1 - alpha/2, giving a band that already varies in width with x;
  2. score each calibration point by how far outside that band it fell,
     E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i));
  3. take the same (n+1)-corrected quantile of those scores, Q;
  4. predict [q_lo(x) - Q, q_hi(x) + Q].

Two things are worth noticing before any of it runs.

The first is that the conformal step is unchanged. `conformal_quantile` from day
1 is imported here untouched, and the finite-sample guarantee it provides does
not care that the scores now come from a pair of quantile models rather than a
residual. That is the actual content of "any nonconformity score works": the
base method supplies the shape of the interval and conformal supplies the
validity, and the two do not negotiate.

The second is that the CQR score can be *negative*, which the day-1 score never
was. When y lands strictly inside the band both terms of the max are negative,
and if the quantile models are over-wide then Q comes out negative too and step
4 shrinks the band rather than inflating it. So the correction is two-sided and
repairs a miscalibrated base method in either direction. An absolute residual
has no way to express "this was too conservative".

What CQR does not do is achieve conditional coverage. That is impossible in
finite samples without assumptions. It stays marginal; the width simply adapts,
so the marginal guarantee is spread far more evenly over x. The bin table at
the bottom is the measurement of exactly that, against day 1 as the baseline.
"""

import numpy as np

from day1_split_conformal import (
    average_width,
    conformal_quantile,
    empirical_coverage,
    make_heteroscedastic_data,
    polynomial_features,
    split_conformal,
    three_way_split,
)


def pinball_loss(residual, tau):
    """Check loss for quantile `tau`, where residual = y - prediction.

    Asymmetric on purpose: overshooting costs (1 - tau) per unit, undershooting
    costs tau. Minimizing the expectation puts the prediction at the tau-th
    conditional quantile, which is the whole reason this loss exists - tau=0.5
    recovers the median and absolute error, and tau=0.95 buys one unit of
    under-prediction for the price of nineteen units of over-prediction.
    """
    return np.maximum(tau * residual, (tau - 1.0) * residual)


def _standardize(design):
    """Center and scale every non-intercept column, returning (matrix, mu, sd).

    Not cosmetic here. The design is a degree-6 Vandermonde on x in [-3, 3], so
    the raw columns span three orders of magnitude and a single learning rate
    cannot serve all of them - the high-degree column diverges at whatever rate
    lets the linear column move at all. Day 1 avoided this entirely by solving
    ridge in closed form; the pinball loss has no closed form, so the
    conditioning becomes the caller's problem.
    """
    mu = design.mean(axis=0)
    sd = design.std(axis=0)
    mu[0], sd[0] = 0.0, 1.0          # leave the intercept column alone
    sd[sd < 1e-12] = 1.0
    return (design - mu) / sd, mu, sd


def fit_quantile_regression(x, y, tau, degree=6, iterations=60, epsilon=1e-4):
    """Fit a linear-in-features quantile regression by MM / IRLS.

    The pinball loss is convex but piecewise linear, so there is no normal
    equation and no curvature to exploit. Subgradient descent is the obvious
    first thing to reach for and it is a trap here: the objective's kinks make
    a constant step oscillate forever, a 1/sqrt(t) decay fixes that but crawls,
    and at a few thousand steps the fit is still visibly short of the optimum -
    close enough to look converged on a plot and wrong enough to ruin the point
    of the day, since an under-fit quantile band does not widen where the noise
    widens and CQR then has nothing to adapt with.

    Hunter and Lange's majorize-minimize algorithm is the better tool. The
    check function is majorized at the current residuals r_k by a quadratic

        rho_tau(r) <= (1/4) [ r^2 / |r_k| + (4*tau - 2) * r + const ]

    which touches rho_tau at r = r_k and lies above it everywhere, so minimizing
    the majorizer can only decrease the true loss - descent is guaranteed with
    no step size to choose. The minimizer is a weighted least-squares solve
    with weights 1/|r_k|:

        (X' W X) b = X' W y + (2*tau - 1) * X' 1

    Each iteration is a solve of a (degree+1) square system, so sixty of them
    cost less than a few hundred gradient steps and land on the optimum instead
    of near it. `epsilon` floors the weights, since a residual of exactly zero
    is an interpolated point and its weight would otherwise be infinite - this
    is the standard perturbation and the reason the method is IRLS in practice
    rather than the exact linear program.

    Returns a dict rather than a bare coefficient vector because the
    standardization has to travel with the weights to be undone at predict
    time.
    """
    design = polynomial_features(x, degree)
    scaled, mu, sd = _standardize(design)
    y = np.asarray(y, dtype=float).reshape(-1)

    weights = np.zeros(scaled.shape[1])
    weights[0] = np.quantile(y, tau)      # start at the unconditional quantile
    linear_term = (2.0 * tau - 1.0) * scaled.sum(axis=0)
    ridge = 1e-9 * np.eye(scaled.shape[1])

    for _ in range(iterations):
        residual = y - scaled @ weights
        reweight = 1.0 / np.maximum(np.abs(residual), epsilon)
        gram = scaled.T @ (scaled * reweight[:, None]) + ridge
        weights = np.linalg.solve(gram, scaled.T @ (reweight * y) + linear_term)

    return {"weights": weights, "mu": mu, "sd": sd, "degree": degree, "tau": tau}


def predict_quantile(model, x):
    """Apply a `fit_quantile_regression` model to new inputs."""
    design = polynomial_features(x, model["degree"])
    scaled = (design - model["mu"]) / model["sd"]
    return scaled @ model["weights"]


def cqr_scores(lower, upper, y):
    """Signed distance outside the band: max(lower - y, y - upper).

    Positive when y falls outside and equal to how far outside, negative when y
    falls inside and equal to the distance to the nearer edge. The negative
    branch is the part that matters and is easy to discard by wrapping this in
    an abs(): it is what lets the calibration step tighten a band that was too
    wide, not merely widen one that was too narrow.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    return np.maximum(lower - y, y - upper)


def fit_cqr(x_train, y_train, alpha=0.1, degree=6, **fit_kwargs):
    """Fit the lower and upper quantile models for a target level `alpha`.

    Splitting alpha evenly between the tails is a convention, not a
    requirement - any pair of levels whose difference is 1 - alpha would do,
    and an asymmetric split is the right call when the loss of missing high
    differs from missing low.
    """
    low = fit_quantile_regression(x_train, y_train, alpha / 2.0,
                                  degree=degree, **fit_kwargs)
    high = fit_quantile_regression(x_train, y_train, 1.0 - alpha / 2.0,
                                   degree=degree, **fit_kwargs)
    return low, high


def conformalize_cqr(low_model, high_model, x_calib, y_calib, x_test, alpha=0.1):
    """Calibrate a fitted quantile band and return (lower, upper, Q).

    The two quantile models are fitted independently and nothing constrains
    q_lo <= q_hi, so they can cross where the data is thin - the classic
    quantile-crossing problem. Sorting the two predictions pointwise is the
    cheap repair and it is applied here. It is worth being clear that this is
    not required for validity: the conformal step would still deliver 1 - alpha
    coverage over a crossed band, because it only ever sees the scores. It is
    required for the output to be an interval rather than an empty set.
    """
    calib_low = predict_quantile(low_model, x_calib)
    calib_high = predict_quantile(high_model, x_calib)
    calib_low, calib_high = (np.minimum(calib_low, calib_high),
                             np.maximum(calib_low, calib_high))

    scores = cqr_scores(calib_low, calib_high, y_calib)
    correction = conformal_quantile(scores, alpha)

    test_low = predict_quantile(low_model, x_test)
    test_high = predict_quantile(high_model, x_test)
    test_low, test_high = (np.minimum(test_low, test_high),
                           np.maximum(test_low, test_high))

    return test_low - correction, test_high + correction, correction


def coverage_by_bin(x, lower, upper, y_true, edges):
    """Empirical coverage within each |x| bin, as a list of (left, right, n, cov).

    The diagnostic day 1 ended on. Marginal coverage is one number and hides
    everything; this is the cheapest way to see whether the width is tracking
    the noise or just sitting at its average.
    """
    magnitude = np.abs(np.asarray(x).reshape(-1))
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (magnitude >= left) & (magnitude < right)
        if mask.sum() == 0:
            continue
        rows.append((left, right, int(mask.sum()),
                     empirical_coverage(lower[mask], upper[mask], y_true[mask])))
    return rows


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x, y = make_heteroscedastic_data(2000, rng)
    (x_tr, y_tr), (x_ca, y_ca), (x_te, y_te) = three_way_split(x, y, rng)
    print(f"split sizes: train={x_tr.shape[0]} calib={x_ca.shape[0]} test={x_te.shape[0]}")

    # the quantile fits alone, before any conformal correction. these have no
    # guarantee at all - they are just a model, and whether they cover 90% of
    # the data depends entirely on whether degree-6 polynomials happen to
    # capture this noise structure.
    alpha = 0.1
    low_model, high_model = fit_cqr(x_tr, y_tr, alpha=alpha)
    raw_low = predict_quantile(low_model, x_te)
    raw_high = predict_quantile(high_model, x_te)
    print(f"\nuncalibrated quantile band (no guarantee): "
          f"coverage={empirical_coverage(raw_low, raw_high, y_te):.3f} "
          f"width={average_width(raw_low, raw_high):.3f}")

    cqr_low, cqr_high, q = conformalize_cqr(low_model, high_model,
                                            x_ca, y_ca, x_te, alpha=alpha)
    print(f"conformalized (Q={q:+.4f}): "
          f"coverage={empirical_coverage(cqr_low, cqr_high, y_te):.3f} "
          f"width={average_width(cqr_low, cqr_high):.3f}")
    print("  Q < 0 means the raw band was too wide and calibration tightened it;"
          "\n  Q > 0 means it was too narrow. an absolute-residual score cannot"
          "\n  express the first case at all.")

    # day 1's constant-width band on the identical split, as the baseline
    sc_low, sc_high, half = split_conformal(x_tr, y_tr, x_ca, y_ca, x_te, alpha=alpha)
    print(f"\nday 1 split conformal: "
          f"coverage={empirical_coverage(sc_low, sc_high, y_te):.3f} "
          f"width={average_width(sc_low, sc_high):.3f} (constant {2 * half:.3f})")

    # the point of the whole day. both are marginally valid; only one of them
    # is honest about where the uncertainty actually is.
    edges = [0.0, 0.75, 1.5, 2.25, 3.0]
    print(f"\nconditional coverage by |x| bin (nominal {1 - alpha:.2f}):")
    print(f"  {'bin':<16}{'n':>6}{'split':>10}{'cqr':>10}{'cqr width':>12}")
    sc_rows = coverage_by_bin(x_te, sc_low, sc_high, y_te, edges)
    cq_rows = coverage_by_bin(x_te, cqr_low, cqr_high, y_te, edges)
    for (left, right, n, sc_cov), (_, _, _, cq_cov) in zip(sc_rows, cq_rows):
        magnitude = np.abs(x_te.reshape(-1))
        mask = (magnitude >= left) & (magnitude < right)
        width = average_width(cqr_low[mask], cqr_high[mask])
        print(f"  [{left:.2f}, {right:.2f})  {n:>6}{sc_cov:>10.3f}"
              f"{cq_cov:>10.3f}{width:>12.3f}")

    # a single number for "how unevenly is the coverage spread", so the
    # comparison does not rest on reading the table
    sc_spread = max(r[3] for r in sc_rows) - min(r[3] for r in sc_rows)
    cq_spread = max(r[3] for r in cq_rows) - min(r[3] for r in cq_rows)
    print(f"\ncoverage spread across bins: split={sc_spread:.3f} cqr={cq_spread:.3f}"
          " (smaller is better)")

    # and the marginal guarantee survives the change of score, which is the
    # claim that matters most. averaged over independent calibration draws,
    # not over one lucky split.
    print(f"\n100 independent splits at alpha={alpha}:")
    cqr_cov, split_cov = [], []
    for seed in range(100):
        trial_rng = np.random.default_rng(2000 + seed)
        xs, ys = make_heteroscedastic_data(600, trial_rng)
        (a, b), (c, d), (e, f) = three_way_split(xs, ys, trial_rng)
        lo_m, hi_m = fit_cqr(a, b, alpha=alpha)
        lo, hi, _ = conformalize_cqr(lo_m, hi_m, c, d, e, alpha=alpha)
        cqr_cov.append(empirical_coverage(lo, hi, f))
        slo, shi, _ = split_conformal(a, b, c, d, e, alpha=alpha)
        split_cov.append(empirical_coverage(slo, shi, f))
    cqr_cov, split_cov = np.array(cqr_cov), np.array(split_cov)
    print(f"  cqr   mean={cqr_cov.mean():.4f} sd={cqr_cov.std():.4f}")
    print(f"  split mean={split_cov.mean():.4f} sd={split_cov.std():.4f}")
    print("  both sit at or just above nominal - adapting the width costs"
          "\n  nothing in validity, which is the entire selling point.")
