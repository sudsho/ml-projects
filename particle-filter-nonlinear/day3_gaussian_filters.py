"""
Day 3 of particle filtering.

The extended and unscented Kalman filters on the same model, and where each
one's Gaussian assumption breaks against the particle posterior. Day 1 said this
would be about the observation being even. It is not, and finding that out is
most of what today was.

The setup I expected to write up. `h(x) = x^2 / 20` is even, so `dh/dx = x / 10`
vanishes at the origin, so a filter whose update is `K = P H / S` has `K = 0`
whenever its predicted mean sits at zero - it throws the observation away exactly
where the posterior is most ambiguous. The UKF does the same thing for a
different-looking reason: the sigma set is symmetric about the mean, `h` is even,
so the cross-covariance `sum w_i (X_i - m)(Y_i - y_hat)` pairs an odd deviation
with an even function and cancels to zero identically. Two filters, two
mechanisms, one hole. That was going to be the day.

Three things killed it, in order.

**The gain collapse is not an approximation error, it is correct.** `h` is a
quadratic, so every moment a Gaussian filter needs has a closed form, and the run
checks all six against it. `Cov(x, h(x)) = m P / 10` exactly, and the EKF's `P H`
is `m P / 10`, and the UKF's weighted sum is `m P / 10`. Both to 1e-15. The
numerator of the gain contains no approximation at all. For a centred Gaussian
the covariance between the state and *any* even function of it is exactly zero
because the integrand is odd, so the optimal linear estimator has gain zero
there, and both filters are computing that correctly. The linear update cannot
use the observation at a centred prediction, and that is a fact about the model
rather than a defect in either method. A better linearization would not help,
because there is nothing to linearize better.

**And the errors are not where that story needs them to be.** RMSE against the
truth, split by whether the exact grid posterior is bimodal at that step:

    grid mean (MMSE optimum)   overall= 3.859  bimodal= 6.274  unimodal= 1.687
    particle filter, N=2000    overall= 3.959  bimodal= 6.456  unimodal= 1.691
    EKF                        overall=20.534  bimodal=19.618  unimodal=20.971
    UKF, beta = 2              overall=10.674  bimodal=10.635  unimodal=10.693

The Gaussian filters are 3x and 5x worse than the exact filter, which was
expected. What was not is the last two columns: they are *flat*. The EKF is
slightly worse on the unimodal steps than on the bimodal ones. Bimodality is what
day 1 built the whole model to produce and it is not the axis the failure lies
along. Same for the gain - the 19 steps with `|m_pre| < 2` carry 15.7% of the
EKF's squared error, less than their 19% share.

**So it is the transition, and the control says so.** Flip `f` to
`0.5 x + 8 cos(1.2 t)`, linear in `x`, and leave `h` alone - still even, still a
symmetric likelihood. The EKF goes from 5.32x the exact filter's RMSE to 0.98x.

That control is confounded and it took a second pass to see it: linearizing `f`
also collapsed the bimodal steps from 33 to 3, because the odd rational term is
what was keeping the prior nearly sign-symmetric. So the headline number tests
two changes at once. The comparison that survives is within each model's
*unimodal* steps, where the posterior is one bump in both cases and only the
dynamics differ:

    nonlinear EKF   12.43x the grid filter   (67 steps)
    nonlinear UKF    6.34x                   (67 steps)
    linear    EKF     1.01x                  (97 steps)
    linear    UKF     1.03x                  (97 steps)

Twelve-fold to nothing, at fixed posterior shape. The mechanism is the Jensen gap
in the prediction: `Q = 10` so the belief is never narrower than `sd ~ 3.2`, and
`f`'s rational term turns over at `|x| ~ 1`, so `f` is strongly curved across the
width of the belief at every single step and `f(E[x]) != E[f(x)]` by a lot. Started
from the exact posterior's own moments, the EKF's one-step predicted mean is off
by RMSE 5.19 against a between-step spread of the exact means of 7.55. The UKF
halves that to 2.87, which is what a third-order-accurate transform buys, and it
is still not close.

The last thing, and it is the one I would not have predicted. The UT's `beta` is
added to the zeroth covariance weight; the standard value is 2. On this `h` the
`P^2` coefficient of the transformed variance works out to `(c - 1 + beta) / 400`
against a true `2 / 400`, so with `c = 3` the transform is **exact at beta = 0**
and doubles the `P^2` term at the standard `beta = 2`. The exact one is worse:

    UKF, beta = 0    RMSE 13.539   mean NIS 10.207   95% coverage 0.55
    UKF, beta = 2    RMSE 10.674   mean NIS  1.739   95% coverage 0.70

Getting the one-step moment right makes the filter worse by a quarter, because
the inflated `S` was shrinking the gain and that was compensating for the
prediction bias the transform never modelled. Two errors partly cancelling, and
removing the one that is removable leaves the other one bare. Which also means
`beta = 2`'s good showing here is not a property of the tuning, and I should not
carry it anywhere.

None of the three is consistent, for the record - NEES 316, 4.6 and 42 against a
target of 1, and 95% intervals covering 0.39, 0.70 and 0.55 where the grid
posterior covers 0.95 and the particle filter covers 0.96.

What today is not: a claim that Gaussian filters cannot handle multimodality.
That claim is probably true and today did not test it, because the failure here
saturated before the bimodality could contribute anything measurable. Testing it
needs a model whose nonlinearity is gentle at the scale of `Q` and whose
observation is still even, and I do not have one. Day 4 is the smoother, so it
will have to wait.

Run: `python day3_gaussian_filters.py`
"""

import numpy as np

from day1_bootstrap_filter import (
    Q,
    R,
    X0_VAR,
    count_modes,
    grid_filter,
    grid_map,
    grid_mean,
    observation_mean,
    simulate,
    transition_mean,
)
from day2_resampling import run_filter


# --- the two Jacobians --------------------------------------------------------


def transition_jacobian(x, t):
    """`df/dx = 0.5 + 25 (1 - x^2) / (1 + x^2)^2`.

    The forcing term is constant in `x` and drops out, so the linearization is
    time-invariant even though `f` is not. Worth noting that this derivative is
    `25.5` at the origin and falls below zero for `|x| > 1`: the linearized
    dynamics are strongly expansive exactly where the state is small, which is
    also where the observation is least informative.
    """
    d = 1.0 + x * x
    return 0.5 + 25.0 * (1.0 - x * x) / (d * d)


def observation_jacobian(x):
    """`dh/dx = x / 10`, which is zero at `x = 0`.

    An even function has a stationary point at the origin, so its linearization
    there is the zero map, and a filter whose update is `K = P H / S` has `K = 0`
    when `H = 0` - the observation is discarded exactly where the posterior is
    most ambiguous. This looks like the day's headline and is not: `P H` is
    `Cov(x, h(x))` *exactly* here, so the zero is the optimal linear gain rather
    than a linearization artifact, and the run finds the damage is somewhere else
    entirely.
    """
    return x / 10.0


# --- exact moments of the observation under a Gaussian ------------------------
#
# h is a quadratic, so everything a Gaussian filter needs about `p(y | m, P)` has
# a closed form, and every approximation below can be scored against the closed
# form rather than against a fine-grained simulation. This is the same trick as
# day 1's grid filter one level down - build the exact thing first so the
# approximate thing has something to be wrong about.


def observation_moments(m, P):
    """Exact `E[h(x)]`, `Var(h(x))` and `Cov(x, h(x))` for `x ~ N(m, P)`.

    With `h(x) = x^2 / 20`:

        E[h]        = (m^2 + P) / 20
        Var(h)      = (4 m^2 P + 2 P^2) / 400 = m^2 P / 100 + P^2 / 200
        Cov(x, h)   = 2 m P / 20 = m P / 10

    from `E[x^2] = m^2 + P`, `Var(x^2) = 4 m^2 P + 2 P^2` and
    `E[x^3] = m^3 + 3 m P`.

    The third one is the important one and it does not involve any
    approximation: for a *centred* Gaussian the covariance between the state and
    any even function of it is exactly zero, because the integrand is odd.
    """
    mean = (m * m + P) / 20.0
    var = m * m * P / 100.0 + P * P / 200.0
    cov = m * P / 10.0
    return mean, var, cov


# --- extended Kalman filter ---------------------------------------------------


def extended_kalman_filter(observations, m0=0.0, P0=X0_VAR, var_floor=1e-9,
                           f=transition_mean, f_jac=transition_jacobian):
    """First-order EKF on the growth model, Joseph-form update.

    Records the pieces the comparison needs rather than only the state estimate:
    the predicted observation, the innovation covariance and the gain, so that
    each can be scored against `observation_moments` at the same `(m, P)`.

    `f` and `f_jac` are arguments only so that the control experiment at the end
    can swap in a linear transition while leaving the even observation alone.
    """
    n = len(observations)
    out = {key: np.empty(n) for key in
           ("mean", "var", "mean_pre", "var_pre", "y_hat", "S", "gain", "innovation")}

    m, P = m0, P0
    for i, y in enumerate(observations):
        t = i + 1

        # predict: linearize f about the current mean
        F = f_jac(m, t)
        m = f(m, t)
        P = F * P * F + Q
        out["mean_pre"][i] = m
        out["var_pre"][i] = P

        # update: linearize h about the predicted mean
        H = observation_jacobian(m)
        y_hat = observation_mean(m)
        S = H * P * H + R
        K = P * H / S

        m = m + K * (y - y_hat)
        # Joseph form. with a scalar state this is not about conditioning so much
        # as about staying positive when K is large; the kalman project's reason
        # for it carries over unchanged.
        A = 1.0 - K * H
        P = max(A * P * A + K * R * K, var_floor)

        out["y_hat"][i] = y_hat
        out["S"][i] = S
        out["gain"][i] = K
        out["innovation"][i] = y - y_hat
        out["mean"][i] = m
        out["var"][i] = P

    return out


# --- unscented Kalman filter --------------------------------------------------


def sigma_points(m, P, kappa=2.0, beta=2.0):
    """Three sigma points for a scalar state, with mean and covariance weights.

    `n = 1`, so with `alpha = 1` the scaled transform collapses to the original
    one: `c = n + kappa`, points at `m` and `m +/- sqrt(c P)`, mean weights
    `(c - 1)/c` and `1/(2c)` twice. `kappa = 3 - n = 2` is the usual choice and
    matches the fourth moment of a scalar Gaussian.

    `beta` is added to the zeroth covariance weight. The standard value is `2`,
    which is optimal for a Gaussian prior. It is a free parameter here and the
    run below is about what it costs on this particular `h`.
    """
    c = 1.0 + kappa
    spread = np.sqrt(max(c * P, 0.0))

    points = np.array([m, m + spread, m - spread])
    w_mean = np.array([(c - 1.0) / c, 0.5 / c, 0.5 / c])
    w_cov = w_mean.copy()
    w_cov[0] += beta

    return points, w_mean, w_cov


def unscented_kalman_filter(observations, m0=0.0, P0=X0_VAR, kappa=2.0, beta=2.0,
                            var_floor=1e-9, f=transition_mean):
    """UKF on the growth model. Same recorded diagnostics as the EKF.

    `f` is an argument for the same reason it is on the EKF - the control
    experiment needs a linear transition with the observation left alone.
    """
    n = len(observations)
    out = {key: np.empty(n) for key in
           ("mean", "var", "mean_pre", "var_pre", "y_hat", "S", "gain", "innovation")}

    m, P = m0, P0
    for i, y in enumerate(observations):
        t = i + 1

        # predict: push the sigma points through f and re-fit
        points, w_mean, w_cov = sigma_points(m, P, kappa, beta)
        propagated = f(points, t)
        m = float(w_mean @ propagated)
        P = float(w_cov @ (propagated - m) ** 2) + Q
        P = max(P, var_floor)
        out["mean_pre"][i] = m
        out["var_pre"][i] = P

        # update: re-draw sigma points at the predicted moments and push through h
        points, w_mean, w_cov = sigma_points(m, P, kappa, beta)
        observed = observation_mean(points)
        y_hat = float(w_mean @ observed)
        S = float(w_cov @ (observed - y_hat) ** 2) + R
        C = float(w_cov @ ((points - m) * (observed - y_hat)))
        K = C / S

        m = m + K * (y - y_hat)
        P = max(P - K * S * K, var_floor)

        out["y_hat"][i] = y_hat
        out["S"][i] = S
        out["gain"][i] = K
        out["innovation"][i] = y - y_hat
        out["mean"][i] = m
        out["var"][i] = P

    return out


# --- scoring helpers ----------------------------------------------------------


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def grid_var(grid, posteriors):
    dx = grid[1] - grid[0]
    mean = (posteriors * grid).sum(axis=1) * dx
    second = (posteriors * grid ** 2).sum(axis=1) * dx
    return second - mean ** 2


def central_interval(grid, posterior, level=0.95):
    """Equal-tailed `level` interval of a gridded density."""
    dx = grid[1] - grid[0]
    cdf = np.cumsum(posterior) * dx
    cdf /= cdf[-1]
    tail = (1.0 - level) / 2.0
    lo = grid[int(np.searchsorted(cdf, tail))]
    hi = grid[int(min(np.searchsorted(cdf, 1.0 - tail), len(grid) - 1))]
    return lo, hi


def coverage(states, means, variances, level=1.96):
    half = level * np.sqrt(variances)
    return float(np.mean(np.abs(states - means) <= half))


# --- run ----------------------------------------------------------------------

if __name__ == "__main__":
    n_steps = 100
    states, observations = simulate(n_steps, seed=7)

    print("gaussian filters against the particle posterior")
    print(f"  steps                     : {n_steps}")
    print(f"  Q = {Q}, R = {R}")

    grid, posteriors = grid_filter(observations)
    truth_mean = grid_mean(grid, posteriors)
    truth_map = grid_map(grid, posteriors)
    truth_var = grid_var(grid, posteriors)

    bimodal = np.array([
        len(count_modes(grid, posteriors[i])[0]) > 1 for i in range(n_steps)
    ])
    print(f"  bimodal steps (grid)      : {int(bimodal.sum())} of {n_steps}")

    # ---------------------------------------------------------------- part 1
    # the closed form first. every claim after this rests on these three being
    # right, so they are asserted rather than described.

    print("\n-- exactness of each approximation at a fixed (m, P) --")

    checks = [(0.0, 4.0), (3.0, 9.0), (-7.5, 12.0), (12.0, 2.5)]
    ekf_mean_err = []
    ukf_mean_err = []
    ekf_var_err = []
    ukf_var_err_b2 = []
    ukf_var_err_b0 = []
    ekf_cov_err = []
    ukf_cov_err = []

    for m, P in checks:
        true_mean, true_var, true_cov = observation_moments(m, P)

        ekf_mean_err.append(observation_mean(m) - true_mean)
        H = observation_jacobian(m)
        ekf_var_err.append(H * P * H - true_var)
        ekf_cov_err.append(P * H - true_cov)

        for beta, bucket in ((2.0, ukf_var_err_b2), (0.0, ukf_var_err_b0)):
            pts, w_mean, w_cov = sigma_points(m, P, kappa=2.0, beta=beta)
            obs = observation_mean(pts)
            y_hat = float(w_mean @ obs)
            bucket.append(float(w_cov @ (obs - y_hat) ** 2) - true_var)
            if beta == 2.0:
                ukf_mean_err.append(y_hat - true_mean)
                ukf_cov_err.append(
                    float(w_cov @ ((pts - m) * (obs - y_hat))) - true_cov
                )

    print("  predicted observation mean, error vs exact E[h(x)]:")
    print(f"    EKF  h(m)          : {np.array(ekf_mean_err)}")
    print(f"    UKF  sum w h(X_i)  : max |err| = {np.abs(ukf_mean_err).max():.2e}")
    print("  observation variance, error vs exact Var(h(x)):")
    print(f"    EKF  H P H         : {np.array(ekf_var_err)}")
    print(f"    UKF  beta = 2      : {np.array(ukf_var_err_b2)}")
    print(f"    UKF  beta = 0      : max |err| = {np.abs(ukf_var_err_b0).max():.2e}")
    print("  cross-covariance, error vs exact Cov(x, h(x)):")
    print(f"    EKF  P H           : max |err| = {np.abs(ekf_cov_err).max():.2e}")
    print(f"    UKF  sum w dx dy   : max |err| = {np.abs(ukf_cov_err).max():.2e}")

    # the UT reproduces E[h] for any kappa because h is quadratic and the sigma
    # set matches the first two moments; the weighted sum of a quadratic depends
    # on nothing else. the same argument gives the cross-covariance.
    assert np.abs(ukf_mean_err).max() < 1e-10
    assert np.abs(ukf_cov_err).max() < 1e-10
    # and the EKF's cross-covariance is exact too, which is the part that matters
    # below: P H = m P / 10 = Cov(x, h(x)) identically, no approximation in it.
    assert np.abs(ekf_cov_err).max() < 1e-10
    # the variance is where they separate. the P^2 coefficient of the UT is
    # (c - 1 + beta) / 400 against a true 2 / 400, so with c = 3 it is exact at
    # beta = 0 and doubled at the standard beta = 2; the EKF has no P^2 term at
    # all and is short by exactly P^2 / 200.
    for (m, P), err in zip(checks, ekf_var_err):
        assert abs(err + P * P / 200.0) < 1e-9
    assert np.abs(ukf_var_err_b0).max() < 1e-10
    for (m, P), err in zip(checks, ukf_var_err_b2):
        assert abs(err - 2.0 * P * P / 400.0) < 1e-9
    print("  [ok] all six closed-form claims hold")

    # ---------------------------------------------------------------- part 2
    # so neither filter is wrong about the gain's numerator. run them.

    ekf = extended_kalman_filter(observations)
    ukf = unscented_kalman_filter(observations, kappa=2.0, beta=2.0)
    ukf0 = unscented_kalman_filter(observations, kappa=2.0, beta=0.0)
    pf = run_filter(observations, n_particles=2000, seed=11, scheme="systematic")

    print("\n-- RMSE against the true trajectory --")
    rows = [
        ("grid mean (MMSE optimum)", truth_mean),
        ("grid MAP", truth_map),
        ("particle filter, N=2000", pf["mean_pre"]),
        ("EKF", ekf["mean"]),
        ("UKF, beta = 2", ukf["mean"]),
        ("UKF, beta = 0", ukf0["mean"]),
    ]
    for name, est in rows:
        overall = rmse(est, states)
        bi = rmse(np.asarray(est)[bimodal], states[bimodal])
        uni = rmse(np.asarray(est)[~bimodal], states[~bimodal])
        print(f"  {name:26s} overall={overall:7.3f}  bimodal={bi:7.3f}  unimodal={uni:7.3f}")

    print("\n-- distance from the best available Gaussian centre (the grid mean) --")
    for name, est in rows[2:]:
        print(f"  {name:26s} {rmse(est, truth_mean):7.3f}")

    # ---------------------------------------------------------------- part 3
    # the gain.

    print("\n-- the gain, and where it goes --")
    for name, filt in (("EKF", ekf), ("UKF, beta = 2", ukf)):
        gains = np.abs(filt["gain"])
        near_zero = np.abs(filt["mean_pre"]) < 1.0
        print(f"  {name:14s} |K| median={np.median(gains):.4f} "
              f"min={gains.min():.2e}  steps with |m_pre| < 1: {int(near_zero.sum())}"
              f"  |K| there: median={np.median(gains[near_zero]) if near_zero.any() else float('nan'):.4f}")

    # the exact optimal linear gain, computed from the closed form at the same
    # predicted moments, is Cov(x, h) / (Var(h) + R). if the EKF's gain matched it
    # the linearization would be costing nothing in the numerator, which is what
    # part 1 already proved pointwise; this is the same statement along the run.
    exact_cov = ekf["mean_pre"] * ekf["var_pre"] / 10.0
    print(f"  EKF gain numerator vs exact Cov(x, h): max |err| = "
          f"{np.abs(ekf['var_pre'] * observation_jacobian(ekf['mean_pre']) - exact_cov).max():.2e}")

    # so at a centred prediction the *optimal* linear update has gain zero, and
    # both filters compute that correctly. the question is how much of the error
    # lands on those steps.
    centred = np.abs(ekf["mean_pre"]) < 2.0
    if centred.any():
        share = np.sum((ekf["mean"][centred] - states[centred]) ** 2) / \
                np.sum((ekf["mean"] - states) ** 2)
        print(f"  steps with |m_pre| < 2: {int(centred.sum())}, "
              f"carrying {100 * share:.1f}% of the EKF's squared error")

    # ---------------------------------------------------------------- part 4
    # consistency. NIS should average 1 in one dimension if S is honest.

    print("\n-- consistency --")
    for name, filt in (("EKF", ekf), ("UKF, beta = 2", ukf), ("UKF, beta = 0", ukf0)):
        nis = filt["innovation"] ** 2 / filt["S"]
        nees = (states - filt["mean"]) ** 2 / filt["var"]
        cov95 = coverage(states, filt["mean"], filt["var"])
        print(f"  {name:14s} mean NIS={nis.mean():7.3f}  mean NEES={nees.mean():8.3f}"
              f"  95% coverage={cov95:.2f}")

    grid_cov = np.mean([
        central_interval(grid, posteriors[i])[0] <= states[i] <=
        central_interval(grid, posteriors[i])[1]
        for i in range(n_steps)
    ])
    pf_cov = coverage(states, pf["mean_pre"], pf["var_pre"])
    print(f"  {'grid posterior':14s} 95% coverage={grid_cov:.2f}")
    print(f"  {'PF, N=2000':14s} 95% coverage={pf_cov:.2f}")

    # ---------------------------------------------------------------- part 5
    # the variance each filter reports, against the variance the posterior has.

    print("\n-- reported variance vs the grid posterior's --")
    for name, filt in (("EKF", ekf), ("UKF, beta = 2", ukf), ("UKF, beta = 0", ukf0),
                       ("PF, N=2000", pf)):
        v = filt["var"] if "var" in filt else filt["var_pre"]
        ratio = v / truth_var
        print(f"  {name:14s} median ratio={np.median(ratio):6.3f}  "
              f"on bimodal steps={np.median(ratio[bimodal]):6.3f}  "
              f"on unimodal steps={np.median(ratio[~bimodal]):6.3f}")

    # ---------------------------------------------------------------- part 6
    # both candidate mechanisms are now dead - the failure is not concentrated on
    # the bimodal steps and it is not concentrated on the low-gain steps. so the
    # question is whether a single step of either filter is bad, or whether single
    # steps are fine and the recursion is what destroys them. that is answerable
    # by refusing to let the filter carry its own state: start every step from the
    # grid posterior's moments, take one step, and score against the grid
    # posterior's moments one step later.

    print("\n-- one step from an oracle-initialized belief --")

    dx = grid[1] - grid[0]

    exact_pred_mean = np.empty(n_steps - 1)
    exact_pred_var = np.empty(n_steps - 1)
    ekf_pred_mean = np.empty(n_steps - 1)
    ekf_pred_var = np.empty(n_steps - 1)
    ukf_pred_mean = np.empty(n_steps - 1)
    ukf_pred_var = np.empty(n_steps - 1)
    ekf_step_mean = np.empty(n_steps - 1)
    ukf_step_mean = np.empty(n_steps - 1)

    for i in range(n_steps - 1):
        t = i + 2
        p = posteriors[i]
        m, P = truth_mean[i], truth_var[i]

        # exact one-step prediction: push the whole density through f by quadrature
        f_grid = transition_mean(grid, t)
        e1 = float((p * f_grid).sum() * dx)
        e2 = float((p * f_grid ** 2).sum() * dx)
        exact_pred_mean[i] = e1
        exact_pred_var[i] = e2 - e1 * e1 + Q

        # EKF prediction from the same moments
        F = transition_jacobian(m, t)
        ekf_pred_mean[i] = transition_mean(m, t)
        ekf_pred_var[i] = F * P * F + Q

        # UKF prediction from the same moments
        pts, w_mean, w_cov = sigma_points(m, P, kappa=2.0, beta=2.0)
        prop = transition_mean(pts, t)
        um = float(w_mean @ prop)
        ukf_pred_mean[i] = um
        ukf_pred_var[i] = float(w_cov @ (prop - um) ** 2) + Q

        # and the full cycle: predict, then update on the real next observation
        y = observations[i + 1]
        H = observation_jacobian(ekf_pred_mean[i])
        S = H * ekf_pred_var[i] * H + R
        K = ekf_pred_var[i] * H / S
        ekf_step_mean[i] = ekf_pred_mean[i] + K * (y - observation_mean(ekf_pred_mean[i]))

        pts, w_mean, w_cov = sigma_points(ukf_pred_mean[i], ukf_pred_var[i],
                                          kappa=2.0, beta=2.0)
        obs = observation_mean(pts)
        y_hat = float(w_mean @ obs)
        S = float(w_cov @ (obs - y_hat) ** 2) + R
        C = float(w_cov @ ((pts - ukf_pred_mean[i]) * (obs - y_hat)))
        ukf_step_mean[i] = ukf_pred_mean[i] + (C / S) * (y - y_hat)

    print(f"  prediction mean, RMSE vs exact : EKF={rmse(ekf_pred_mean, exact_pred_mean):7.3f}"
          f"  UKF={rmse(ukf_pred_mean, exact_pred_mean):7.3f}"
          f"  (spread of the exact means: {exact_pred_mean.std():.3f})")
    print(f"  prediction var, median ratio   : EKF={np.median(ekf_pred_var / exact_pred_var):7.3f}"
          f"  UKF={np.median(ukf_pred_var / exact_pred_var):7.3f}")
    print(f"  one full cycle, RMSE vs the grid posterior mean one step later:")
    print(f"    EKF={rmse(ekf_step_mean, truth_mean[1:]):7.3f}"
          f"  UKF={rmse(ukf_step_mean, truth_mean[1:]):7.3f}"
          f"  against free-running EKF={rmse(ekf['mean'][1:], truth_mean[1:]):7.3f}"
          f"  UKF={rmse(ukf['mean'][1:], truth_mean[1:]):7.3f}")

    # if the oracle-initialized numbers are small and the free-running ones are
    # not, the transform is not the problem and the recursion is.

    print("\n-- does either filter recover once it is lost? --")
    for name, filt in (("EKF", ekf), ("UKF, beta = 2", ukf)):
        lost = np.abs(filt["mean"] - states) > 10.0
        first = int(np.argmax(lost)) if lost.any() else -1
        # longest consecutive run of lost steps
        best = run = 0
        for flag in lost:
            run = run + 1 if flag else 0
            best = max(best, run)
        print(f"  {name:14s} lost on {int(lost.sum())}/{n_steps} steps, "
              f"first at t={first}, longest unbroken run={best}")

    # ---------------------------------------------------------------- part 7
    # the control. everything above is correlational: the failure is spread
    # evenly over bimodal and unimodal steps, it does not sit on the low-gain
    # steps, the update's cross-covariance carries no approximation error at all,
    # and one oracle-initialized step is already most of the way to the
    # free-running error. all of that points at the prediction rather than at the
    # observation, and it can be tested directly by flipping one thing.
    #
    #     f(x, t) = 0.5 x + 8 cos(1.2 t)
    #
    # linear in x, same forcing, same Q. h is left exactly as it was, so the
    # observation is still even, the likelihood is still symmetric, and the
    # posterior is still bimodal. If the Gaussian filters are fine here, the
    # bimodality was never what was hurting them.

    print("\n-- control: linear transition, same even observation --")

    def linear_transition(x, t):
        return 0.5 * x + 8.0 * np.cos(1.2 * t)

    def linear_jacobian(x, t):
        return 0.5

    def simulate_linear(n, seed):
        rng = np.random.default_rng(seed)
        x = np.sqrt(X0_VAR) * rng.standard_normal()
        xs, ys = np.empty(n), np.empty(n)
        for i in range(n):
            x = linear_transition(x, i + 1) + np.sqrt(Q) * rng.standard_normal()
            xs[i] = x
            ys[i] = observation_mean(x) + np.sqrt(R) * rng.standard_normal()
        return xs, ys

    def grid_filter_linear(ys, lo=-35.0, hi=35.0, n_grid=2001):
        g = np.linspace(lo, hi, n_grid)
        step = g[1] - g[0]
        prior = np.exp(-0.5 * g ** 2 / X0_VAR) / np.sqrt(2 * np.pi * X0_VAR)
        out = np.empty((len(ys), n_grid))
        for i, y in enumerate(ys):
            means = linear_transition(g, i + 1)
            # transition kernel as a dense matrix: rows = from, cols = to
            kernel = np.exp(-0.5 * (g[None, :] - means[:, None]) ** 2 / Q)
            predicted = (prior[:, None] * kernel).sum(axis=0)
            predicted /= predicted.sum() * step
            like = np.exp(-0.5 * (y - observation_mean(g)) ** 2 / R)
            post = predicted * like
            post /= post.sum() * step
            out[i] = post
            prior = post
        return g, out

    lin_states, lin_obs = simulate_linear(n_steps, seed=7)
    lin_grid, lin_post = grid_filter_linear(lin_obs)
    lin_truth_mean = grid_mean(lin_grid, lin_post)
    lin_bimodal = np.array([
        len(count_modes(lin_grid, lin_post[i])[0]) > 1 for i in range(n_steps)
    ])

    lin_ekf = extended_kalman_filter(lin_obs, f=linear_transition, f_jac=linear_jacobian)
    lin_ukf = unscented_kalman_filter(lin_obs, f=linear_transition)
    lin_pf = run_filter(lin_obs, n_particles=2000, seed=11, scheme="systematic")

    print(f"  bimodal steps (grid)      : {int(lin_bimodal.sum())} of {n_steps}"
          f"   [nonlinear model had {int(bimodal.sum())}]")
    print("  RMSE against the true trajectory:")
    for name, est in (("grid mean", lin_truth_mean),
                      ("EKF", lin_ekf["mean"]),
                      ("UKF, beta = 2", lin_ukf["mean"])):
        print(f"    {name:16s} overall={rmse(est, lin_states):7.3f}"
              f"  bimodal={rmse(np.asarray(est)[lin_bimodal], lin_states[lin_bimodal]):7.3f}"
              f"  unimodal={rmse(np.asarray(est)[~lin_bimodal], lin_states[~lin_bimodal]):7.3f}")
    print(f"    {'ratio EKF/grid':16s} {rmse(lin_ekf['mean'], lin_states) / rmse(lin_truth_mean, lin_states):.2f}"
          f"   [nonlinear model: {rmse(ekf['mean'], states) / rmse(truth_mean, states):.2f}]")
    print(f"    {'ratio UKF/grid':16s} {rmse(lin_ukf['mean'], lin_states) / rmse(lin_truth_mean, lin_states):.2f}"
          f"   [nonlinear model: {rmse(ukf['mean'], states) / rmse(truth_mean, states):.2f}]")

    # the control is confounded and it is worth saying so rather than banking it:
    # linearizing f did not only remove the Jensen gap, it also collapsed the
    # bimodal steps from 33 to 3, because the odd rational term is what kept the
    # prior nearly sign-symmetric. so "EKF matches the grid filter here" is not by
    # itself evidence about the observation.
    #
    # the uncontaminated comparison is within the unimodal steps of each model.
    # there the posterior is a single bump in both cases and the only difference
    # left is the transition.
    print("  restricted to each model's unimodal steps - same posterior shape,"
          " only the dynamics differ:")
    for label, est, ref, mask, st in (
        ("nonlinear EKF", ekf["mean"], truth_mean, ~bimodal, states),
        ("nonlinear UKF", ukf["mean"], truth_mean, ~bimodal, states),
        ("linear    EKF", lin_ekf["mean"], lin_truth_mean, ~lin_bimodal, lin_states),
        ("linear    UKF", lin_ukf["mean"], lin_truth_mean, ~lin_bimodal, lin_states),
    ):
        est = np.asarray(est)
        print(f"    {label:16s} RMSE/grid = "
              f"{rmse(est[mask], st[mask]) / rmse(np.asarray(ref)[mask], st[mask]):6.2f}"
              f"   ({int(mask.sum())} steps)")

    print("\ndone")
