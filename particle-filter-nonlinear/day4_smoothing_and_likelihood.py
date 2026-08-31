"""
Day 4 of particle filtering.

Backward-simulation smoothing, the likelihood estimator and its unbiasedness, and
RMSE/coverage against an exact filter on a linear model.

The smoother is enormous here and I had the reason wrong.

    RMSE, exact filter     3.859
    RMSE, exact smoother   1.315      65.9% better
    mean posterior variance   21.652 -> 2.206

Two-thirds of the filter's error is not irreducible, it is just early. The whole
project has been about a model whose posterior goes genuinely bimodal because `h`
is even and an observation cannot see the sign; the backward pass sees the sign,
because the *next* few states are not sign-symmetric once the forcing term moves.
Split by whether the filter was bimodal at that step:

    filter's bimodal steps (33)   filter 6.274   smoother 1.560
    the rest (67)                 filter 1.687   smoother 1.175

so the gain is concentrated exactly where day 1 said the difficulty was, 4x
against 1.4x. That much I expected.

**The explanation I would have given for it is wrong, and the linear control says
so.** The obvious story is "the filter is very uncertain here (variance 21.7) and
smoothing has room to work". Weaken the linear model's observation gain and its
filter gets uncertain too, over a 12x range, with linearity and unimodality left
alone:

    linear, c=1.00   filter var  0.911   smoother  0.893   smoothing gain  0.3%
    linear, c=0.50   filter var  2.914   smoother  2.769   smoothing gain  0.7%
    linear, c=0.20   filter var  8.118   smoother  7.621   smoothing gain  0.7%
    linear, c=0.10   filter var 11.368   smoother 11.016   smoothing gain  1.3%
    nonlinear        filter var 21.652   smoother  2.206   smoothing gain 65.9%

Twelve-fold more filter variance buys one extra point of smoothing gain. So the
gain is not a function of how uncertain the filter is, and posterior width is the
wrong variable entirely. What the backward pass is worth depends on the *kind* of
uncertainty: a Gaussian filter's leftover is a continuous spread and the future
shrinks it by a gain that is small and stays small, while this model's leftover is
a discrete question - which branch - and one look at the future answers it
outright. Variance is the wrong summary for a two-atom ambiguity, which is a
thing I have been writing down as a variance since day 1.

**And the mode count is not measuring what I have been using it for.** The
smoothed posteriors are bimodal at *more* steps than the filtered ones, 36 against
33, while being 4x more accurate and 10x tighter. Bimodality has been standing in
for "ambiguous" in every previous day's tables and it does not track ambiguity at
all - it is a shape statistic and it survives the collapse of the thing it was
supposed to indicate. First measurement in the project that separates the two.

**The smoother is not better at its job than the filter, it just has an easier
job.** Backward simulation with `M = 800` trajectories lands at RMSE 0.123 from
the exact smoother mean, where the `N = 2000` filter is 0.323 from the exact
filter mean, and reading that as "the smoother is more accurate" is a scale error.
Divided by each posterior's own sd the order reverses, 0.083 against 0.069. The
smoothed target is a tighter density, so there is less to get wrong before any
method is involved.

`M` also buys much less than it looks like it should - 50, 200 and 800
trajectories give 0.192, 0.197, 0.123, which is not `M^-1/2` and is barely
monotone. It cannot be: the backward pass only ever re-weights the cloud the
forward pass already produced, so its error floor is set by `N`, not `M`. Spending
on trajectories when the filter is the bottleneck buys noise.

**The degeneracy check comes out backwards.** The reason to pay `O(M N T)` for
backward simulation rather than tracing resampling ancestry is that ancestral
paths collapse in the *past* - every trajectory shares one ancestor a few dozen
steps back. These do not: 160 distinct values at step 0, median 390 across steps.
The thin step is the *last* one, 56 of 800, and the cause is visible in the filter
- ESS at step `T-1` is 29.6 of 2000 against a median of 587, and `x_T` is drawn
straight from those weights with no backward reweighting to flatten them. So the
method fixes the degeneracy it is advertised against and inherits a different one
at the single step where it is doing nothing.

**The likelihood estimator is unbiased and that is nearly useless at small `N`.**
`Zhat = prod_t (1/N) sum_j p(y_t | x_j^t)` against the grid filter's exact
`log Z = -67.428` over 25 steps, 400 independent runs each:

    N=  100   mean Zhat/Z 0.996 +/- 0.100   sd(Zhat/Z) 2.010   mean log Zhat - log Z -3.501
    N=  400   mean Zhat/Z 1.012 +/- 0.038   sd(Zhat/Z) 0.768   mean log Zhat - log Z -0.250
    N= 1600   mean Zhat/Z 1.005 +/- 0.019   sd(Zhat/Z) 0.374   mean log Zhat - log Z -0.058

Unbiasedness holds at every `N`, exactly as advertised and with no `N`-dependence
to it. But at `N = 100` the estimator's standard deviation is twice the quantity
it is estimating, so a single run carries essentially no information and only the
average over 400 of them is worth anything. The property is about the ensemble and
says nothing about the run you have. Meanwhile `log Zhat` is biased *down* by 3.5
nats there - Jensen, since the estimator is unbiased on the raw scale and `log` is
concave - and that bias falls off roughly like `1/N` (3.501, 0.250, 0.058 across
16x). Anything that consumes a log-likelihood, which is everything, is reading a
downward-biased number whose bias depends on `N`.

**The linear control, where 'exact' needs no qualifier.** Kalman is the posterior
rather than an approximation to it, so the whole gap is Monte Carlo error:

    N=   50   RMSE vs Kalman mean 0.2474   95% coverage 0.922
    N=  200   RMSE vs Kalman mean 0.1031   95% coverage 0.969   ratio 2.40
    N=  800   RMSE vs Kalman mean 0.0504   95% coverage 0.972   ratio 2.05
    N= 3200   RMSE vs Kalman mean 0.0249   95% coverage 0.971   ratio 2.02
    Kalman                                 95% coverage 0.970

Textbook `N^-1/2` once `N` is past 200, and the coverage converges onto the exact
filter's own 0.970 from below - at `N = 50` the intervals undercover at 0.922
because the particle quantiles cannot resolve a 2.5% tail out of 50 points. The
nonlinear model's exact filter covers 0.950 on the nose, for the record, so
nothing above is a calibration problem in the model.

Run: `python day4_smoothing_and_likelihood.py`.
"""

import numpy as np

from day1_bootstrap_filter import (
    Q,
    R,
    X0_VAR,
    count_modes,
    grid_filter,
    grid_mean,
    observation_mean,
    simulate,
    transition_mean,
)
from day2_resampling import systematic_resample


# --- the instrument, one level fuller than day 1's ----------------------------
#
# day 1's grid filter returns the filtering posteriors and nothing else, which is
# all day 1 through 3 needed. Today needs two more things out of the same
# recursion and neither can be recovered afterwards:
#
#   - the one-step *predictive* densities p(x_t | y_{1:t-1}), because the
#     smoothing recursion divides by them;
#   - the normalising constants p(y_t | y_{1:t-1}), whose product is the
#     likelihood the particle estimate is scored against.
#
# So the recursion is written out again here rather than imported. The filtering
# output is checked against day 1's to make sure the two agree.


def grid_filter_full(observations, lo=-35.0, hi=35.0, n_grid=2001):
    """Filtering and predictive densities plus the exact log-likelihood.

    Same quadrature as day 1. The only additions are that `predicted` is kept
    and that the update divides by its own integral instead of by the sum, so the
    divisor is `p(y_t | y_{1:t-1})` and the running sum of its logs is
    `log p(y_{1:T})` exactly (up to the grid and the truncation window).

    The likelihood *must* carry the `1 / sqrt(2 pi R)` factor here. Day 1 dropped
    it because a constant divides out of a normalised posterior; it does not
    divide out of the evidence, and leaving it off would shift `log Z` by
    `T * 0.919` and quietly break every comparison below.

    Returns `(grid, filtered, predicted, log_evidence)` with `log_evidence[i]`
    the increment `log p(y_i | y_{1:i-1})`.
    """
    grid = np.linspace(lo, hi, n_grid)
    dx = grid[1] - grid[0]

    posterior = np.exp(-0.5 * grid ** 2 / X0_VAR)
    posterior /= posterior.sum() * dx

    n = len(observations)
    filtered = np.empty((n, n_grid))
    predicted = np.empty((n, n_grid))
    log_evidence = np.empty(n)

    for i, y in enumerate(observations):
        t = i + 1

        centres = transition_mean(grid, t)
        offsets = grid[:, None] - centres[None, :]
        kernel = np.exp(-0.5 * offsets ** 2 / Q) / np.sqrt(2.0 * np.pi * Q)
        pred = kernel @ posterior * dx
        predicted[i] = pred

        like = (np.exp(-0.5 * (y - observation_mean(grid)) ** 2 / R)
                / np.sqrt(2.0 * np.pi * R))
        unnormalised = pred * like
        evidence = unnormalised.sum() * dx
        log_evidence[i] = np.log(evidence)

        posterior = unnormalised / evidence
        filtered[i] = posterior

    return grid, filtered, predicted, log_evidence


def grid_smoother(grid, filtered, predicted):
    """Exact `p(x_t | y_{1:T})` by the forward-backward (Kitagawa) recursion.

        p(x_t | y_{1:T}) = p(x_t | y_{1:t}) * int f(x' | x_t) p(x' | y_{1:T})
                                                 / p(x' | y_{1:t}) dx'

    read as: the filtering density reweighted by how well each `x_t` explains the
    smoothed future, with the division correcting for the fact that the future
    already contains the information the filter used.

    Two implementation notes that are not cosmetic.

    The division by `predicted` is the reason day 1's filter could not be reused.
    It is also where this recursion is fragile: in the tails `predicted` is
    numerically zero while `smoothed` is zero too, and `0/0` has to become `0`
    rather than a `nan`, so the ratio is masked instead of floored. Flooring
    biases the tails upward by manufacturing a ratio where there is no density.

    The kernel is indexed `[k, j] = N(grid_k; f(grid_j, t), Q)`, matching day 1,
    so the integral over `x'` at fixed `x_t` contracts the *first* axis and the
    transpose in the matrix product is doing real work.
    """
    dx = grid[1] - grid[0]
    n = len(filtered)
    smoothed = np.empty_like(filtered)
    smoothed[-1] = filtered[-1]

    for i in range(n - 2, -1, -1):
        ratio = np.zeros_like(grid)
        live = predicted[i + 1] > 0.0
        ratio[live] = smoothed[i + 1][live] / predicted[i + 1][live]

        centres = transition_mean(grid, i + 2)
        offsets = grid[:, None] - centres[None, :]
        kernel = np.exp(-0.5 * offsets ** 2 / Q) / np.sqrt(2.0 * np.pi * Q)

        integral = kernel.T @ ratio * dx
        post = filtered[i] * integral
        smoothed[i] = post / (post.sum() * dx)

    return smoothed


# --- the particle filter, this time keeping its history -----------------------


def filter_with_history(observations, n_particles, seed):
    """Bootstrap filter that stores every step's particles and weights.

    Day 2's `run_filter` throws the cloud away as it goes, which is correct for
    filtering and useless for smoothing: the backward pass needs the *filtering*
    particles at every `t`, since it reweights them against a future state
    sampled later. So the storage is `O(T N)` instead of `O(N)`, and that is the
    honest cost of the smoother rather than an implementation detail.

    What is stored at step `i` is the cloud *before* resampling - the propagated
    particles and their normalised weights. That pair is the filtering
    approximation. Storing the post-resampling cloud instead would still be
    consistent but strictly noisier, for day 2's reason: resampling is a bootstrap
    sample of the weighted average and adds variance to any estimate taken from
    it.

    `log_increment[i]` is `log((1/N) sum_j p(y_i | x_j))` with the full Gaussian
    constant, computed in a log-sum-exp so it survives the tail steps where every
    particle is far from the observation. It is only the standard unbiased
    estimator because the particles going into it are equally weighted, which
    holds here since this filter resamples at every step.
    """
    rng = np.random.default_rng(seed)
    n = len(observations)

    particles = np.sqrt(X0_VAR) * rng.standard_normal(n_particles)

    hist_particles = np.empty((n, n_particles))
    hist_weights = np.empty((n, n_particles))
    log_increment = np.empty(n)
    ess = np.empty(n)

    log_norm = -0.5 * np.log(2.0 * np.pi * R)

    for i, y in enumerate(observations):
        t = i + 1

        particles = (transition_mean(particles, t)
                     + np.sqrt(Q) * rng.standard_normal(n_particles))

        log_like = log_norm - 0.5 * (y - observation_mean(particles)) ** 2 / R

        # log((1/N) sum exp(log_like)), stabilised. this is the incremental
        # evidence estimate and it is what has to be unbiased.
        peak = log_like.max()
        log_increment[i] = peak + np.log(np.exp(log_like - peak).mean())

        weights = np.exp(log_like - peak)
        weights /= weights.sum()

        hist_particles[i] = particles
        hist_weights[i] = weights
        ess[i] = 1.0 / float(weights @ weights)

        particles = particles[systematic_resample(weights, rng)]

    return {
        "particles": hist_particles,
        "weights": hist_weights,
        "log_increment": log_increment,
        "log_likelihood": float(log_increment.sum()),
        "ess": ess,
        "mean": (hist_particles * hist_weights).sum(axis=1),
    }


def weighted_quantile(values, weights, q):
    """Quantile of a weighted particle set, by the empirical CDF."""
    order = np.argsort(values)
    v = values[order]
    cw = np.cumsum(weights[order])
    cw /= cw[-1]
    return float(v[np.searchsorted(cw, q, side="left").clip(0, len(v) - 1)])


# --- backward-simulation smoothing --------------------------------------------


def backward_simulation(history, n_trajectories, seed):
    """FFBSi: draw whole smoothed trajectories backwards through the filter cloud.

    Sample `x_T ~ {w_j^T}`, then walk backwards drawing

        P(x_t = x_j^t | x_{t+1}) ∝ w_j^t * N(x_{t+1}; f(x_j^t, t+1), Q)

    Each draw is an independent sample from `p(x_{1:T} | y_{1:T})`, so the
    trajectories can be averaged into marginals or used whole - which is the
    thing the marginal (FFBSm) smoother cannot give you, and the reason to pay
    for this one.

    Why this is not the filter's own ancestry. Tracing resampling parents
    backwards also produces trajectories, and they are degenerate: every path
    collapses to a single ancestor a few dozen steps back, so the smoothed
    marginal at early `t` is supported on one or two distinct values no matter
    how large `N` is. Backward simulation resamples from the *whole* cloud at
    each step instead of following stored links, so the support does not thin -
    at the price of an `O(N)` weight vector per step per trajectory, i.e.
    `O(M N T)` overall against the filter's `O(N T)`.

    Vectorised over trajectories: the backward weights are an `(M, N)` array per
    step, built once and sampled by inverse-CDF along the particle axis.
    """
    rng = np.random.default_rng(seed)
    particles = history["particles"]
    weights = history["weights"]
    n_steps, n_particles = particles.shape

    out = np.empty((n_trajectories, n_steps))

    idx = rng.choice(n_particles, size=n_trajectories, p=weights[-1])
    out[:, -1] = particles[-1][idx]

    for i in range(n_steps - 2, -1, -1):
        # transition into step i+1 uses time index i+2 (day 1: state i is made
        # with t = i+1), so this is f(x_j^i, i+2).
        centres = transition_mean(particles[i], i + 2)
        resid = out[:, i + 1][:, None] - centres[None, :]

        log_w = np.log(weights[i] + 1e-300)[None, :] - 0.5 * resid ** 2 / Q
        log_w -= log_w.max(axis=1, keepdims=True)
        w = np.exp(log_w)
        w /= w.sum(axis=1, keepdims=True)

        cdf = np.cumsum(w, axis=1)
        u = rng.random(n_trajectories)[:, None]
        pick = (cdf < u).sum(axis=1).clip(0, n_particles - 1)
        out[:, i] = particles[i][pick]

    return out


# --- the linear-Gaussian control ----------------------------------------------
#
# Everything above is scored against a grid filter, which is exact only up to a
# quadrature and a truncation window. For the RMSE and coverage numbers it is
# worth having a model where "exact" needs no qualifier at all, so: a linear
# Gaussian model, where the Kalman filter and the RTS smoother are the posterior
# rather than an approximation to it, and any gap the particle filter shows is
# Monte Carlo error and nothing else.

LIN_A = 0.5
LIN_C = 1.0


def linear_transition(x, t):
    return LIN_A * x + 8.0 * np.cos(1.2 * t)


def simulate_linear(n_steps, seed, c=LIN_C):
    """`c` is the observation gain, and it is an argument because the size of the
    smoothing gain below turns out to depend on it far more than on linearity."""
    rng = np.random.default_rng(seed)
    x = np.sqrt(X0_VAR) * rng.standard_normal()
    states = np.empty(n_steps)
    obs = np.empty(n_steps)
    for i in range(n_steps):
        t = i + 1
        x = linear_transition(x, t) + np.sqrt(Q) * rng.standard_normal()
        states[i] = x
        obs[i] = c * x + np.sqrt(R) * rng.standard_normal()
    return states, obs


def kalman_filter(observations, m0=0.0, P0=X0_VAR, c=LIN_C):
    """Exact filter for the linear model, plus the exact log-likelihood.

    The evidence increment is the innovation's own density,
    `log N(y_t; C m_pre, C P_pre C + R)`, which is exact here for the same reason
    the mean is: every distribution in the recursion really is Gaussian. This is
    the reference the particle likelihood estimator is scored against.
    """
    n = len(observations)
    out = {k: np.empty(n) for k in ("mean", "var", "mean_pre", "var_pre")}
    log_evidence = np.empty(n)

    m, P = m0, P0
    for i, y in enumerate(observations):
        t = i + 1
        m = linear_transition(m, t)
        P = LIN_A * P * LIN_A + Q
        out["mean_pre"][i] = m
        out["var_pre"][i] = P

        S = c * P * c + R
        innovation = y - c * m
        log_evidence[i] = -0.5 * (np.log(2.0 * np.pi * S) + innovation ** 2 / S)

        K = P * c / S
        m = m + K * innovation
        P = (1.0 - K * c) * P
        out["mean"][i] = m
        out["var"][i] = P

    out["log_evidence"] = log_evidence
    out["log_likelihood"] = float(log_evidence.sum())
    return out


def rts_smoother(kf):
    """Exact `p(x_t | y_{1:T})` for the linear model, backwards."""
    n = len(kf["mean"])
    mean = kf["mean"].copy()
    var = kf["var"].copy()
    for i in range(n - 2, -1, -1):
        gain = kf["var"][i] * LIN_A / kf["var_pre"][i + 1]
        mean[i] = kf["mean"][i] + gain * (mean[i + 1] - kf["mean_pre"][i + 1])
        var[i] = kf["var"][i] + gain * (var[i + 1] - kf["var_pre"][i + 1]) * gain
    return {"mean": mean, "var": var}


def linear_particle_filter(observations, n_particles, seed, c=LIN_C):
    """Bootstrap filter on the linear model. Same algorithm, different `f`, `h`."""
    rng = np.random.default_rng(seed)
    n = len(observations)
    particles = np.sqrt(X0_VAR) * rng.standard_normal(n_particles)

    mean = np.empty(n)
    lo = np.empty(n)
    hi = np.empty(n)
    log_increment = np.empty(n)
    log_norm = -0.5 * np.log(2.0 * np.pi * R)

    for i, y in enumerate(observations):
        t = i + 1
        particles = (linear_transition(particles, t)
                     + np.sqrt(Q) * rng.standard_normal(n_particles))

        log_like = log_norm - 0.5 * (y - c * particles) ** 2 / R
        peak = log_like.max()
        log_increment[i] = peak + np.log(np.exp(log_like - peak).mean())

        w = np.exp(log_like - peak)
        w /= w.sum()

        mean[i] = float(w @ particles)
        lo[i] = weighted_quantile(particles, w, 0.025)
        hi[i] = weighted_quantile(particles, w, 0.975)

        particles = particles[systematic_resample(w, rng)]

    return {"mean": mean, "lo": lo, "hi": hi,
            "log_likelihood": float(log_increment.sum())}


# --- helpers ------------------------------------------------------------------


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def grid_var(grid, densities):
    dx = grid[1] - grid[0]
    m = (densities * grid).sum(axis=1) * dx
    second = (densities * grid ** 2).sum(axis=1) * dx
    return second - m ** 2


def grid_interval(grid, density, level=0.95):
    dx = grid[1] - grid[0]
    cdf = np.cumsum(density) * dx
    cdf /= cdf[-1]
    tail = (1.0 - level) / 2.0
    lo = grid[int(np.searchsorted(cdf, tail))]
    hi = grid[int(min(np.searchsorted(cdf, 1.0 - tail), len(grid) - 1))]
    return lo, hi


def interval_coverage(states, lo, hi):
    return float(np.mean((states >= np.asarray(lo)) & (states <= np.asarray(hi))))


# --- run ----------------------------------------------------------------------

if __name__ == "__main__":
    n_steps = 100
    states, observations = simulate(n_steps, seed=7)

    print("smoothing, likelihood and the linear control")
    print(f"  steps                     : {n_steps}")

    # ---- the instrument agrees with day 1's ---------------------------------
    grid, filtered, predicted, log_ev = grid_filter_full(observations)
    day1_grid, day1_post = grid_filter(observations)
    print("\n1. the fuller grid filter against day 1's")
    print(f"  max abs difference in the filtering densities : "
          f"{np.abs(filtered - day1_post).max():.3e}")
    print(f"  exact log p(y_1:T)                            : {log_ev.sum():.4f}")

    # ---- smoothing ----------------------------------------------------------
    smoothed = grid_smoother(grid, filtered, predicted)
    dx = grid[1] - grid[0]
    print(f"  smoothed densities integrate to               : "
          f"{(smoothed.sum(axis=1) * dx).min():.6f} .. "
          f"{(smoothed.sum(axis=1) * dx).max():.6f}")

    filt_mean = grid_mean(grid, filtered)
    smooth_mean = grid_mean(grid, smoothed)
    filt_var = grid_var(grid, filtered)
    smooth_var = grid_var(grid, smoothed)

    bimodal = np.array([len(count_modes(grid, filtered[i])[0]) > 1
                        for i in range(n_steps)])
    smooth_bimodal = np.array([len(count_modes(grid, smoothed[i])[0]) > 1
                               for i in range(n_steps)])

    print("\n2. what the backward pass buys, on the exact densities")
    print(f"  RMSE, exact filter   : {rmse(filt_mean, states):7.3f}")
    print(f"  RMSE, exact smoother : {rmse(smooth_mean, states):7.3f}"
          f"   ({100 * (1 - rmse(smooth_mean, states) / rmse(filt_mean, states)):.1f}% better)")
    print(f"  mean posterior variance  filter={filt_var.mean():7.3f}"
          f"  smoother={smooth_var.mean():7.3f}")
    print(f"  bimodal steps            filter={int(bimodal.sum()):3d}"
          f"       smoother={int(smooth_bimodal.sum()):3d}")
    print(f"  RMSE on the filter's bimodal steps   filter={rmse(filt_mean[bimodal], states[bimodal]):7.3f}"
          f"  smoother={rmse(smooth_mean[bimodal], states[bimodal]):7.3f}")
    print(f"  RMSE on the rest                     filter={rmse(filt_mean[~bimodal], states[~bimodal]):7.3f}"
          f"  smoother={rmse(smooth_mean[~bimodal], states[~bimodal]):7.3f}")

    # ---- the particle smoother against it -----------------------------------
    print("\n3. backward simulation against the exact smoother")
    history = filter_with_history(observations, n_particles=2000, seed=11)
    print(f"  particle filter mean, RMSE vs exact filter mean : "
          f"{rmse(history['mean'], filt_mean):.4f}")
    for n_traj in (50, 200, 800):
        traj = backward_simulation(history, n_trajectories=n_traj, seed=23)
        pm = traj.mean(axis=0)
        print(f"  M={n_traj:4d} trajectories"
              f"   RMSE vs exact smoother mean = {rmse(pm, smooth_mean):6.3f}"
              f"   RMSE vs truth = {rmse(pm, states):6.3f}")

    # raw RMSEs are not comparable between the two: the smoother's target is a
    # much tighter density, so there is less to get wrong before any method is
    # involved. in units of each posterior's own sd the ordering reverses.
    print(f"  in units of the posterior sd:"
          f"  filter {rmse(history['mean'], filt_mean) / np.sqrt(filt_var.mean()):.3f}"
          f"   smoother (M=800) "
          f"{rmse(backward_simulation(history, 800, seed=23).mean(axis=0), smooth_mean) / np.sqrt(smooth_var.mean()):.3f}")

    traj = backward_simulation(history, n_trajectories=800, seed=23)
    # the degeneracy check, and it comes out backwards from the ancestral one.
    # tracing resampling parents thins the *past*; this thins the last step and
    # nothing else, because step T-1 is drawn straight from the filter weights
    # and inherits their ESS, while every earlier step re-draws from the whole
    # cloud with backward weights that are much flatter.
    uniq = [len(np.unique(traj[:, i])) for i in range(n_steps)]
    print(f"  distinct values among the M=800 trajectories:"
          f"  step 0 = {uniq[0]}   median over steps = {int(np.median(uniq))}"
          f"   step {n_steps - 1} = {uniq[-1]}   min = {min(uniq)}")
    print(f"  filter ESS at the last step = {history['ess'][-1]:.1f} of 2000"
          f"   (median over steps {np.median(history['ess']):.1f})")

    # ---- likelihood ---------------------------------------------------------
    print("\n4. the likelihood estimator, and what is unbiased about it")
    short = 25
    _, _, _, short_log_ev = grid_filter_full(observations[:short])
    exact_short = float(short_log_ev.sum())
    print(f"  exact log p(y_1:{short})   : {exact_short:.4f}")

    for n_particles in (100, 400, 1600):
        logs = np.array([
            filter_with_history(observations[:short], n_particles, seed=1000 + r)
            ["log_likelihood"]
            for r in range(400)
        ])
        ratio = np.exp(logs - exact_short)
        se = ratio.std(ddof=1) / np.sqrt(len(ratio))
        print(f"  N={n_particles:5d}"
              f"   mean Zhat/Z = {ratio.mean():6.3f} +/- {se:.3f}"
              f"   sd(Zhat/Z) = {ratio.std(ddof=1):6.3f}"
              f"   mean log Zhat - log Z = {(logs - exact_short).mean():7.3f}")

    # ---- linear control -----------------------------------------------------
    print("\n5. linear-Gaussian control, where 'exact' needs no qualifier")
    lin_states, lin_obs = simulate_linear(n_steps, seed=7)
    kf = kalman_filter(lin_obs)
    rts = rts_smoother(kf)

    print(f"  exact log p(y_1:T), Kalman : {kf['log_likelihood']:.4f}")
    print(f"  RMSE  Kalman filter={rmse(kf['mean'], lin_states):6.3f}"
          f"   RTS smoother={rmse(rts['mean'], lin_states):6.3f}")

    print("  particle filter vs the exact posterior, by particle count:")
    prev = None
    for n_particles in (50, 200, 800, 3200):
        errs = []
        covs = []
        for r in range(20):
            pf = linear_particle_filter(lin_obs, n_particles, seed=500 + r)
            errs.append(rmse(pf["mean"], kf["mean"]))
            covs.append(interval_coverage(lin_states, pf["lo"], pf["hi"]))
        err = float(np.mean(errs))
        line = (f"    N={n_particles:5d}"
                f"   RMSE vs Kalman mean = {err:.4f}"
                f"   95% coverage = {np.mean(covs):.3f}")
        if prev is not None:
            line += f"   ratio to previous = {prev / err:.2f}  (sqrt(4) = 2.00)"
        prev = err
        print(line)

    kal_lo = kf["mean"] - 1.96 * np.sqrt(kf["var"])
    kal_hi = kf["mean"] + 1.96 * np.sqrt(kf["var"])
    print(f"    Kalman 95% coverage = "
          f"{interval_coverage(lin_states, kal_lo, kal_hi):.3f}")

    grid_lo, grid_hi = zip(*[grid_interval(grid, filtered[i]) for i in range(n_steps)])
    print(f"  for contrast, the nonlinear model's exact filter covers "
          f"{interval_coverage(states, grid_lo, grid_hi):.3f}")

    # ---- what the 66% vs 0.2% gap is actually about -------------------------
    #
    # the two models differ in two ways at once - linearity and how informative
    # the observation is - so "smoothing helps 66% here and 0.2% there" does not
    # isolate anything on its own. c is the observation gain; shrinking it makes
    # the linear observation weak while leaving the model linear and unimodal.
    print("\n6. is the smoothing gain about nonlinearity, or about the"
          " observation being weak?")
    print(f"  nonlinear model : mean filter variance {filt_var.mean():7.3f}"
          f"  smoother {smooth_var.mean():7.3f}"
          f"   smoothing gain {100 * (1 - rmse(smooth_mean, states) / rmse(filt_mean, states)):5.1f}%")
    for c in (1.0, 0.5, 0.2, 0.1):
        st, ob = simulate_linear(n_steps, seed=7, c=c)
        k = kalman_filter(ob, c=c)
        s = rts_smoother(k)
        gain = 100 * (1 - rmse(s["mean"], st) / rmse(k["mean"], st))
        print(f"  linear, c={c:4.2f}   mean filter variance {k['var'].mean():7.3f}"
              f"  smoother {s['var'].mean():7.3f}"
              f"   smoothing gain {gain:5.1f}%")

    print("\ndone")
