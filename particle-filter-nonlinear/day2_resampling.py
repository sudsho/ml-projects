"""
Day 2 of particle filtering.

Yesterday ended with a bootstrap filter using multinomial resampling, which I
described as "the worst of the standard three" and deferred to today. Today is
that comparison, and it turned into something narrower and more useful than a
ranking: three of the four things I expected to find are wrong, and each one is
wrong because I was measuring a single resampling step and reasoning about a
filter that runs a hundred of them.

The three schemes are all inverse-CDF sampling and differ only in how the `N`
uniforms are generated - `N` independent draws (multinomial), one per stratum
(stratified), or one draw and a regular grid (systematic). All three are
unbiased, `E[count_i] = N w_i`, which is the only property resampling is
required to have; everything else is variance.

On a single step, with `N = 200` and a deliberately degenerate weight vector
(`ESS = 4.2`), the sum of offspring-count variances is

    multinomial   152.7      stratified   18.2      systematic   13.7

so 8.4x and 11.2x reductions, and multinomial matches its `sum N w (1-w)` to
three digits. That much was expected. What follows was not.

**1. The ordering caveat I wrote into the systematic docstring is invisible to
the obvious measurement, and provably so.** Systematic's counts depend on where
a particle sits in the array, so I permuted the weight vector and re-measured
the summed variance. It does not move - 0.6% across twelve permutations, less
than multinomial's 3.0% Monte Carlo noise. That is a theorem, not a null result:
`count_i = floor(N C_i - U) - floor(N C_{i-1} - U)` and `N C_i - U` is uniform
mod 1 for any partial sum `C_i`, so `Var(count_i) = frac(N w_i)(1 - frac(N w_i))`
with no other weight in it. The predicted total is 13.683 against 13.684
measured. The ordering lives entirely in the covariances, and a sum of marginal
variances is exactly the statistic that cannot see covariances. Measure
`Var(sum_i count_i x_i / N)` instead and it appears at once: 104x between the
best and worst ordering for stratified, 5.4x for systematic, and 1.0x for
multinomial, whose counts are exchangeable given the weights.

**2. That 104x is worth almost nothing in the filter.** Sorting the particles by
state before resampling is one `argsort`, and it is the intervention section 2b
says should dominate the choice of scheme. Paired over 40 seeds it moves
stratified by `-0.032 +/- 0.028` RMSE and systematic by `+0.006 +/- 0.029`, i.e.
nothing measurable, against a between-scheme gap of `0.876 -> 0.782`. A hundred
steps of a mixing nonlinear transition destroy the correlation structure that
the sort creates, and the sort has to be redone every step to keep buying a
step's worth of it. The single-step variance ratio is a real number about a real
quantity and it is not the quantity the filter's error is made of.

**3. Adaptive resampling is worse here at every threshold, monotonically.** The
standard rule is to resample only when `ESS < N/2`, and the reasoning behind it
is sound and is measured directly below: resampling replaces a weighted average
by a bootstrap sample of it, so the estimate *at that step* gets strictly worse -
by 6.8% for multinomial and 0.6% for systematic. Deferring it should therefore
pay. It does not:

    always          RMSE 0.782      resampled 100.0 / 100 steps
    ESS < 0.50 N    RMSE 0.840      resampled  73.5 / 100
    ESS < 0.25 N    RMSE 1.034      resampled  54.4 / 100
    ESS < 0.10 N    RMSE 1.409      resampled  38.0 / 100

With `R = 1` against `Q = 10` the likelihood is much narrower than the one-step
prior, so the weights degenerate within a couple of steps and by the time ESS
has fallen to `N/2` the damage is already in the particle locations, which
resampling cannot undo - it can only redistribute what is there. The 0.6% paid
per step is small and the compounding is not.

**4. The low-variance schemes preserve more distinct particles, not fewer.**
87.3 of 200 for systematic against 70.7 for multinomial. I had this backwards on
the reasoning that low variance makes the counts track `N w_i` and so makes the
tail round to zero reliably. Survival needs `count_i >= 1` and `E[count_i]` is
fixed at `N w_i`, so lowering variance moves mass off `{2, 3, ...}` and onto
`{0, 1}`; it suppresses duplication, and duplication is what costs diversity.

And ESS, which day 1 called a description of the weight vector rather than a
measurement of accuracy. Across 4000 step-seed pairs the correlation with
absolute error is `-0.208` overall, which looks like ESS working. Split by
whether the exact posterior is bimodal at that step and it is `+0.067` on the 67
unimodal steps and `-0.410` on the 33 bimodal ones. The overall number is
entirely a between-group effect - hard steps have both lower ESS (0.336 N vs
0.416 N) and much larger error (0.907 vs 0.131) - and on the easy steps ESS
carries no information about accuracy at all. It is a diagnostic for the weights,
which is what it was defined to be, and reading it as a health check on the
estimate works only because degeneracy and difficulty happen to coincide here.

Day 3 takes the extended and unscented Kalman filters to the same track, where
the question stops being variance and goes back to whether a Gaussian can be in
the right place at all.
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


# --- the three schemes -------------------------------------------------------
#
# All three are inverse-CDF sampling. They differ only in how the N uniforms fed
# to the inverse CDF are generated, which is the entire subject of the day:
#
#   multinomial  N independent draws from U(0, 1)
#   stratified   one independent draw inside each of N equal strata
#   systematic   one independent draw, then a regular grid of spacing 1/N
#
# so they use N, N and 1 random numbers respectively, and the reduction in
# randomness is the reduction in variance. Every one of them is unbiased in the
# sense that E[count_i] = N w_i, which is the only property resampling is
# actually required to have, and the test block checks it for all three rather
# than deriving it for one and assuming the rest.


def multinomial_resample(weights, rng):
    """N independent draws from the categorical on `weights`.

    The scheme that follows directly from the identity being approximated - the
    weighted empirical measure and an unweighted sample from it represent the
    same distribution - and the worst of the three.

    `count_i ~ Binomial(N, w_i)` marginally, so `Var(count_i) = N w_i (1 - w_i)`,
    which is `O(N)` per particle. The two schemes below get that to `O(1)`.
    """
    n = len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, rng.random(n))


def stratified_resample(weights, rng):
    """One independent uniform inside each of `N` equal strata of `[0, 1)`.

    `u_k = (k + U_k) / N` with `U_k` iid uniform. The `k`-th draw is confined to
    `[k/N, (k+1)/N)`, so the draws cannot clump the way `N` independent uniforms
    can, and a particle with weight `w_i` is guaranteed at least `floor(N w_i)`
    offspring no matter what the uniforms do - the strata entirely inside its
    interval each contribute one.

    Still `N` random numbers. What it removes is not the randomness but the
    ability of that randomness to pile up in one place.
    """
    n = len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    positions = (np.arange(n) + rng.random(n)) / n
    return np.searchsorted(cumulative, positions)


def systematic_resample(weights, rng):
    """One independent uniform, then a regular grid of spacing `1/N`.

    `u_k = (k + U) / N` with a single `U`. The whole scheme has one degree of
    freedom, so an interval of length `w_i` catches either `floor(N w_i)` or
    `ceil(N w_i)` of the grid points and never anything else - the offspring
    count is deterministic up to one unit. That is as close to no resampling
    variance as a scheme can get while remaining unbiased.

    The price is that the guarantee is about the *interval*, and which interval a
    particle owns depends on where it sits in the array, so the joint law of the
    counts is not a function of the weight multiset alone. It is worth being
    careful about what that does and does not affect, because I got it wrong
    first: the **marginal** law is a function of `w_i` alone. Writing
    `C_i` for the partial sums, `count_i = floor(N C_i - U) - floor(N C_{i-1} - U)`,
    and `N C_i - U` is uniform mod 1 for any `C_i`, so

        count_i = ceil(N w_i)   with probability frac(N w_i)
                  floor(N w_i)  otherwise

    with no other weight appearing. The ordering shows up only in the
    covariances, and the run below measures both - the statistic that cannot see
    it, and the one that can.
    """
    n = len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    positions = (np.arange(n) + rng.random()) / n
    return np.searchsorted(cumulative, positions)


SCHEMES = {
    "multinomial": multinomial_resample,
    "stratified": stratified_resample,
    "systematic": systematic_resample,
}


def offspring_counts(scheme, weights, rng):
    """`counts[i]` = how many offspring particle `i` gets, as a vector."""
    index = SCHEMES[scheme](weights, rng)
    return np.bincount(index, minlength=len(weights))


# --- the filter, with the scheme as a parameter ------------------------------


def run_filter(observations, n_particles, seed, scheme="systematic",
               ess_fraction=1.0, sort_particles=False):
    """Bootstrap filter with a pluggable resampler and full diagnostics.

    `ess_fraction` is the adaptive-resampling threshold as a fraction of `N`:
    `1.0` resamples at every step (the standard bootstrap filter, since ESS is
    at most `N`), `0.5` is the common adaptive rule, `0.0` never resamples and
    degenerates to sequential importance sampling.

    Two means are recorded per step and the distinction is the point:

      `mean_pre`   the weighted mean of the propagated particles, computed
                   *before* any resampling. This is the estimator the importance
                   sampling identity actually justifies.
      `mean_post`  the equally-weighted mean after resampling.

    Resampling happens after the estimate is taken, so `mean_post` is a noisier
    version of the same quantity - same conditional expectation, strictly more
    variance, because it replaces a weighted average by a bootstrap sample of it.
    Anything that reports `mean_post` is paying for resampling twice: once in the
    step it happens and again nowhere, since the benefit is entirely in the
    steps that follow. The run measures the size of that.

    `sort_particles` sorts by state before resampling. It is a no-op for
    multinomial, whose counts are exchangeable given the weights, and for the
    other two it is an `argsort` that buys more than the choice between them -
    see the ordering measurements in the run block.

    Returns a dict.
    """
    rng = np.random.default_rng(seed)

    particles = np.sqrt(X0_VAR) * rng.standard_normal(n_particles)
    log_weights = np.zeros(n_particles)

    n_steps = len(observations)
    mean_pre = np.empty(n_steps)
    mean_post = np.empty(n_steps)
    var_pre = np.empty(n_steps)
    ess = np.empty(n_steps)
    resampled = np.zeros(n_steps, dtype=bool)
    unique_after = np.empty(n_steps, dtype=int)

    threshold = ess_fraction * n_particles

    for i, y in enumerate(observations):
        t = i + 1

        particles = (
            transition_mean(particles, t)
            + np.sqrt(Q) * rng.standard_normal(n_particles)
        )

        log_weights = log_weights - 0.5 * (y - observation_mean(particles)) ** 2 / R
        log_weights -= log_weights.max()

        weights = np.exp(log_weights)
        weights /= weights.sum()

        mean_pre[i] = float(weights @ particles)
        var_pre[i] = float(weights @ (particles - mean_pre[i]) ** 2)
        ess[i] = 1.0 / float(weights @ weights)

        if ess[i] < threshold or ess_fraction >= 1.0:
            if sort_particles:
                order = np.argsort(particles)
                particles = particles[order]
                weights = weights[order]

            index = SCHEMES[scheme](weights, rng)
            particles = particles[index]
            log_weights = np.zeros(n_particles)
            resampled[i] = True
            unique_after[i] = len(np.unique(index))
        else:
            log_weights = np.log(weights + 1e-300)
            unique_after[i] = n_particles

        mean_post[i] = float(particles.mean())

    return {
        "mean_pre": mean_pre,
        "mean_post": mean_post,
        "var_pre": var_pre,
        "ess": ess,
        "resampled": resampled,
        "unique_after": unique_after,
        "particles": particles,
    }


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# --- run --------------------------------------------------------------------

if __name__ == "__main__":
    n_steps = 100
    states, observations = simulate(n_steps, seed=7)

    grid, posteriors = grid_filter(observations)
    exact_mean = grid_mean(grid, posteriors)
    mode_counts = np.array([len(count_modes(grid, p)[0]) for p in posteriors])
    bimodal = mode_counts >= 2

    print("day 2 - degeneracy, ESS, and the three resamplers")
    print(f"  model                     : Q = {Q}, R = {R}, {n_steps} steps")
    print(f"  bimodal steps (grid)      : {int(bimodal.sum())} / {n_steps}")

    # --- 1. all three are unbiased, and that is the only thing they must be ---
    #
    # E[count_i] = N w_i. Checked by simulation on a deliberately awkward weight
    # vector - a few dominant particles and a long tail of near-zeros, which is
    # what an actual degenerate step looks like - rather than on uniform weights,
    # where all three schemes coincide and the check is vacuous.
    rng = np.random.default_rng(20260825)
    n = 200

    raw = rng.random(n) ** 8
    raw[0], raw[1], raw[2] = 40.0, 25.0, 15.0
    awkward = raw / raw.sum()

    trials = 20000
    print(f"\nunbiasedness and offspring-count variance"
          f"  (N = {n}, {trials} trials, weights: max {awkward.max():.3f}, "
          f"ESS {1.0 / (awkward @ awkward):.1f})")

    theory_multinomial = float((n * awkward * (1.0 - awkward)).sum())
    stats = {}

    for name in SCHEMES:
        counts = np.empty((trials, n))
        scheme_rng = np.random.default_rng(11)
        for k in range(trials):
            counts[k] = offspring_counts(name, awkward, scheme_rng)

        empirical_mean = counts.mean(axis=0)
        empirical_var = counts.var(axis=0)
        target = n * awkward

        bias = float(np.abs(empirical_mean - target).max())
        total_var = float(empirical_var.sum())
        stats[name] = (bias, total_var, empirical_var)

        print(f"  {name:<12} max |E[count] - N w| = {bias:6.3f}"
              f"   sum Var(count) = {total_var:9.3f}")

        # unbiasedness is the correctness condition, so it is asserted rather
        # than printed and moved past. the tolerance has to be per-particle: the
        # weights here span four orders of magnitude, so a single absolute
        # threshold is either vacuous for the dominant particles or impossible
        # for the tail. multinomial's binomial sd is the largest of the three, so
        # using it for all three is conservative in the right direction.
        standard_error = np.sqrt(n * awkward * (1.0 - awkward) / trials)
        z = float((np.abs(empirical_mean - target) / np.maximum(standard_error, 1e-9)).max())
        assert z < 5.0, (name, bias, z)

    print(f"  multinomial theory        : sum N w (1-w) = {theory_multinomial:.3f}")
    assert abs(stats["multinomial"][1] - theory_multinomial) < 0.05 * theory_multinomial

    # the ordering the three come in is the entire practical content of the day.
    assert stats["stratified"][1] < 0.5 * stats["multinomial"][1]
    assert stats["systematic"][1] < stats["stratified"][1]

    ratio_strat = stats["multinomial"][1] / stats["stratified"][1]
    ratio_syst = stats["multinomial"][1] / stats["systematic"][1]
    print(f"  variance reduction vs multinomial: "
          f"stratified {ratio_strat:.1f}x, systematic {ratio_syst:.1f}x")

    # --- 2a. the ordering effect the marginal variances cannot see ------------
    #
    # I wrote into the `systematic_resample` docstring that permuting the
    # particles changes the law of the counts, and then measured the obvious
    # thing - the sum of the per-particle variances - over random permutations of
    # one weight multiset. It does not move. That is not a measurement failure,
    # it is a theorem: `N C_i - U` is uniform mod 1 whatever `C_i` is, so
    # `Var(count_i) = frac(N w_i) (1 - frac(N w_i))`, in which no other weight
    # appears. The sum of marginal variances is exactly the statistic that is
    # blind to the effect I was looking for.
    print(f"\nordering dependence, seen through the sum of marginal variances")

    predicted_fraction = np.modf(n * awkward)[0]
    predicted = float((predicted_fraction * (1.0 - predicted_fraction)).sum())
    print(f"  systematic, predicted sum Var = frac(Nw)(1-frac(Nw)) : {predicted:.3f}")
    print(f"  systematic, measured                                 : "
          f"{stats['systematic'][1]:.3f}")
    assert abs(stats["systematic"][1] - predicted) < 0.05 * predicted

    for name in ("multinomial", "stratified", "systematic"):
        totals = []
        for p in range(12):
            perm_rng = np.random.default_rng(500 + p)
            permuted = awkward[perm_rng.permutation(n)]

            counts = np.empty((4000, n))
            scheme_rng = np.random.default_rng(97)
            for k in range(4000):
                counts[k] = offspring_counts(name, permuted, scheme_rng)
            totals.append(float(counts.var(axis=0).sum()))

        totals = np.array(totals)
        spread = float(np.ptp(totals)) / float(totals.mean())
        print(f"  {name:<12} over 12 permutations: "
              f"{totals.min():8.3f} .. {totals.max():8.3f}   spread {spread:.1%}")

    # --- 2b. the ordering effect, seen through an estimator -------------------
    #
    # The dependence is in the covariances, so it needs a statistic that has
    # covariances in it. The resampled estimate of `E[x]` is the obvious one and
    # also the one that matters:
    #
    #     xhat = sum_i count_i x_i / N
    #
    # and its variance is `sum_ij Cov(count_i, count_j) x_i x_j / N^2`. Same
    # weights, same values, three orderings of the pair `(w, x)`.
    print(f"\nordering dependence, seen through Var(sum count_i x_i / N)")

    values = np.random.default_rng(4).standard_normal(n) * 5.0
    orderings = {
        "by state x": np.argsort(values),
        "by weight w": np.argsort(awkward),
        "shuffled": np.random.default_rng(9).permutation(n),
    }

    estimator_var = {}
    for name in SCHEMES:
        row = {}
        for label, order in orderings.items():
            ordered_w = awkward[order]
            ordered_x = values[order]

            scheme_rng = np.random.default_rng(5)
            draws = np.empty(8000)
            for k in range(8000):
                draws[k] = offspring_counts(name, ordered_w, scheme_rng) @ ordered_x / n
            row[label] = float(draws.var())

        estimator_var[name] = row
        span = max(row.values()) / min(row.values())
        print(f"  {name:<12} " + "   ".join(
            f"{label} {value:.5f}" for label, value in row.items()
        ) + f"   max/min {span:.1f}x")

    # multinomial's counts are exchangeable given the weights, so its estimator
    # variance genuinely cannot depend on the order. the other two swing by an
    # order of magnitude or more on the same numbers.
    multinomial_span = max(estimator_var["multinomial"].values()) / min(
        estimator_var["multinomial"].values()
    )
    assert multinomial_span < 1.1, multinomial_span
    for name in ("stratified", "systematic"):
        span = max(estimator_var[name].values()) / min(estimator_var[name].values())
        assert span > 5.0, (name, span)

    # and the good ordering is not an arbitrary one. sorting by the state puts
    # neighbouring intervals on neighbouring values, so the +/-1 rounding errors
    # in adjacent counts cancel in the sum instead of accumulating.
    for name in ("stratified", "systematic"):
        row = estimator_var[name]
        assert row["by state x"] < row["shuffled"], (name, row)

    # --- 3. does any of it reach the filter? ---------------------------------
    #
    # Offspring-count variance is a property of a resampling step in isolation.
    # What matters is the error of the filter against the exact posterior, and
    # the two are separated by a hundred steps of dynamics that mix. Averaged
    # over seeds, since a single seed cannot distinguish a scheme from luck.
    print(f"\nfilter error against the grid posterior  (RMSE of the mean, 40 seeds)")

    n_particles = 200
    seeds = range(40)
    filter_rmse = {}

    for name in SCHEMES:
        errors = [
            rmse(run_filter(observations, n_particles, seed=s, scheme=name)["mean_pre"],
                 exact_mean)
            for s in seeds
        ]
        filter_rmse[name] = np.array(errors)
        print(f"  {name:<12} {np.mean(errors):.4f}  "
              f"+/- {np.std(errors) / np.sqrt(len(errors)):.4f} (sem)"
              f"   worst seed {np.max(errors):.4f}")

    # and the same three with the particles sorted by state first. section 2b
    # measured that as a 104x reduction in single-step estimator variance for
    # stratified, so the prediction going in was that it would dominate the
    # choice of scheme. paired over seeds, since the seed-to-seed spread is
    # larger than any plausible effect and an unpaired comparison at 40 seeds
    # would show nothing either way.
    print(f"  --- sorted by state before resampling ---")

    for name in SCHEMES:
        paired = [
            rmse(run_filter(observations, n_particles, seed=s, scheme=name,
                            sort_particles=True)["mean_pre"], exact_mean)
            for s in seeds
        ]
        paired = np.array(paired)
        delta = paired - filter_rmse[name]
        print(f"  {name:<12} {paired.mean():.4f}   paired delta "
              f"{delta.mean():+.4f} +/- {delta.std() / np.sqrt(len(delta)):.4f}"
              f"   better on {int((delta < 0).sum())} / {len(seeds)} seeds")

    # --- 4. resampling makes the current estimate worse ----------------------
    #
    # The weighted mean before resampling and the flat mean after have the same
    # conditional expectation, and the second is a bootstrap sample of the first,
    # so it is strictly noisier. Resampling is paid for at every step and earns
    # its keep only in the steps that follow. Reporting the post-resample mean is
    # a free loss and it is not a small one.
    print(f"\nreporting the mean before vs after the resampling step  (40 seeds)")

    for name in SCHEMES:
        before, after = [], []
        for s in seeds:
            out = run_filter(observations, n_particles, seed=s, scheme=name)
            before.append(rmse(out["mean_pre"], exact_mean))
            after.append(rmse(out["mean_post"], exact_mean))
        penalty = (np.mean(after) - np.mean(before)) / np.mean(before)
        print(f"  {name:<12} pre {np.mean(before):.4f}   post {np.mean(after):.4f}"
              f"   penalty {penalty:+.1%}")
        assert np.mean(after) > np.mean(before), name

    # --- 5. what ESS actually measures ---------------------------------------
    #
    # Day 1 asserted that ESS describes the weight vector and not the accuracy.
    # Here it gets measured. The correlation between per-step ESS and per-step
    # absolute error is computed across every step of every seed, and separately
    # on the steps the grid filter says are bimodal, where the failure mode is
    # not degeneracy at all - the particles agree with each other, ESS is high,
    # and they agree on one of two modes.
    print(f"\nESS as a predictor of error")

    all_ess, all_err, all_bimodal = [], [], []
    for s in seeds:
        out = run_filter(observations, n_particles, seed=s, scheme="systematic")
        all_ess.append(out["ess"])
        all_err.append(np.abs(out["mean_pre"] - exact_mean))
        all_bimodal.append(bimodal)

    all_ess = np.concatenate(all_ess)
    all_err = np.concatenate(all_err)
    all_bimodal = np.concatenate(all_bimodal)

    overall = float(np.corrcoef(all_ess, all_err)[0, 1])
    uni = float(np.corrcoef(all_ess[~all_bimodal], all_err[~all_bimodal])[0, 1])
    bi = float(np.corrcoef(all_ess[all_bimodal], all_err[all_bimodal])[0, 1])

    print(f"  corr(ESS, |error|) overall        : {overall:+.3f}")
    print(f"  ... on unimodal steps             : {uni:+.3f}")
    print(f"  ... on bimodal steps              : {bi:+.3f}")
    print(f"  mean ESS / N  unimodal            : {all_ess[~all_bimodal].mean() / n_particles:.3f}")
    print(f"  mean ESS / N  bimodal             : {all_ess[all_bimodal].mean() / n_particles:.3f}")
    print(f"  mean |error|  unimodal            : {all_err[~all_bimodal].mean():.3f}")
    print(f"  mean |error|  bimodal             : {all_err[all_bimodal].mean():.3f}")

    # the healthy-ESS-and-large-error quadrant, counted rather than described.
    healthy = all_ess > 0.5 * n_particles
    bad = all_err > np.quantile(all_err, 0.9)
    print(f"  steps with ESS > N/2 and error in the worst decile: "
          f"{int((healthy & bad).sum())} / {int(bad.sum())} bad steps "
          f"({(healthy & bad).sum() / max(bad.sum(), 1):.0%})")

    # --- 6. adaptive resampling ----------------------------------------------
    #
    # If resampling costs accuracy at the step it happens and buys accuracy
    # later, then doing it only when the weights have actually degenerated should
    # dominate doing it always. Whether it does is an empirical question about
    # this model and the answer is not the one the reasoning predicts.
    print(f"\nadaptive resampling  (systematic, 40 seeds)")

    for fraction in (1.0, 0.8, 0.5, 0.25, 0.1):
        errors, fires = [], []
        for s in seeds:
            out = run_filter(observations, n_particles, seed=s,
                             scheme="systematic", ess_fraction=fraction)
            errors.append(rmse(out["mean_pre"], exact_mean))
            fires.append(int(out["resampled"].sum()))
        label = "always" if fraction >= 1.0 else f"ESS < {fraction:.2f} N"
        print(f"  {label:<14} RMSE {np.mean(errors):.4f}   "
              f"resampled at {np.mean(fires):5.1f} / {n_steps} steps")

    # --- 7. degeneracy without resampling, for the record --------------------
    #
    # The thing all of the above is preventing. SIS on the same data, same
    # particles, one line removed.
    sis = run_filter(observations, n_particles, seed=0, scheme="systematic",
                     ess_fraction=0.0)
    print(f"\nno resampling at all  (SIS, one seed)")
    print(f"  ESS after 1 / 5 / 20 / 100 steps : "
          f"{sis['ess'][0]:.1f} / {sis['ess'][4]:.1f} / "
          f"{sis['ess'][19]:.2f} / {sis['ess'][99]:.2f}")
    print(f"  RMSE of the mean                 : {rmse(sis['mean_pre'], exact_mean):.3f}")
    assert sis["ess"][-1] < 2.0, sis["ess"][-1]

    # --- 8. particle diversity after resampling ------------------------------
    #
    # The cost side of the ledger. I expected the low-variance schemes to destroy
    # *more* distinct particles, on the reasoning that low variance means the
    # counts track `N w_i` and a particle with `N w_i << 1` reliably rounds to
    # zero instead of occasionally getting lucky. That is backwards, and the
    # arithmetic says so plainly once written down: a particle survives iff its
    # count is at least one, `E[count_i] = N w_i` is fixed, so pushing variance
    # down pushes mass from `{0, 2, 3, ...}` onto `{0, 1}` and, for the many
    # particles with `N w_i < 1`, onto `{0, 1}` with `P(1) = N w_i` exactly. Low
    # variance does not make survival less likely, it makes duplication less
    # likely, and duplication is the thing that costs diversity.
    print(f"\ndistinct particles surviving a resampling step  (mean over steps, 40 seeds)")

    for name in SCHEMES:
        survivors = [
            run_filter(observations, n_particles, seed=s, scheme=name)["unique_after"].mean()
            for s in seeds
        ]
        print(f"  {name:<12} {np.mean(survivors):6.1f} / {n_particles}")

    print("\nall good")
