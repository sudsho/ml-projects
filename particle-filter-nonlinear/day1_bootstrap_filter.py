"""
Day 1 of particle filtering.

The Kalman project ended with a filter that is exact. Not approximately exact -
exact, in the sense that for a linear-Gaussian state-space model the filtering
posterior `p(x_t | y_{1:t})` *is* a Gaussian, the recursions propagate its two
sufficient statistics without discarding anything, and the RTS smoother does the
same backwards. Everything that project measured was about numerical conditioning
or about estimating the noise covariances, never about the belief being the wrong
shape, because there was no shape to be wrong about.

Two things have to hold for that. The transition and observation maps have to be
linear, so that a Gaussian pushed through them stays Gaussian; and the noise has
to be Gaussian, so that the likelihood is a Gaussian in the state. Drop either and
the posterior stops being in any parametric family at all - it is a function, and
the only exact representation of it is the function.

The extended Kalman filter's answer is to linearize and keep the Gaussian anyway.
That is day 3's subject and it is not a bad method. But it answers a question
about *representation* with a change to the *model*, and when the true posterior
has two separated modes there is no linearization that fixes it, because the
failure is not that the mean is in the wrong place. The failure is that a mean is
the wrong object.

Sequential Monte Carlo takes the other branch. Represent the posterior by samples,
which can be any shape, and pay for it in variance instead of in bias. Today
builds the smallest working version of that - the bootstrap filter, three lines of
idea: propagate each sample through the dynamics, weight it by how well it
explains the observation, resample.

The model is the univariate nonlinear growth model, which is the standard
benchmark for this and is standard for a reason:

    x_t = 0.5 x_{t-1} + 25 x_{t-1} / (1 + x_{t-1}^2) + 8 cos(1.2 t) + v_t
    y_t = x_t^2 / 20 + w_t

with `v_t ~ N(0, 10)` and `w_t ~ N(0, 1)`. The observation is **even in the
state**, so an observation says something about `|x_t|` and nothing whatsoever
about its sign. The dynamics are the only thing that can break the tie, and when
the state is near zero they barely do. This is not a hard nonlinearity chosen to
embarrass a Gaussian; it is a structural ambiguity, and a unimodal belief cannot
represent it no matter how it is fitted.

Three things get built, in this order, because each one is the instrument for
judging the next:

  1. **A grid filter.** The state is one-dimensional, so the exact filtering
     recursion can be done by numerical integration on a fine grid, to a precision
     set by the mesh rather than by a sampling seed. This is not part of the
     method - it does not generalize past two or three dimensions, which is the
     entire reason particle filters exist - but it means every claim today about
     what the true posterior looks like is measured rather than asserted, and
     every claim about the particle filter's error is separable into
     approximation error and irreducible posterior spread. That separation is the
     thing I most want and could not have on any project so far.

  2. **Sequential importance sampling**, the base case: sample from the prior,
     accumulate likelihood weights, never resample. It is what the bootstrap
     filter is minus one line, and watching where it dies is the argument for the
     line.

  3. **The bootstrap filter**, with multinomial resampling. Not because
     multinomial is the right choice - it is the worst of the standard three and
     day 2 is about exactly that - but because it is the one that follows
     directly from the identity being approximated, so day 2's alternatives have
     something to be alternatives *to*.

What came out, with the numbers, since two of the four were not what I expected:

  - the posterior is bimodal at 33 of 100 steps, and the two modes are near
    reflections (median `|x_lo + x_hi| / |x|` of 0.029), so the bimodality really
    is the sign ambiguity and not some other structure wearing its clothes;
  - at those steps the reported posterior mean sits at 0.255 of the peak density
    on average and at 2.7e-20 of it at the worst step, against 0.968 on the
    unimodal steps - the summary lands where the posterior says nothing is;
  - **and it is still the better point estimate.** RMSE 3.859 against the MAP's
    4.520, and the margin is widest on exactly the bimodal steps (6.274 vs
    7.514). Where the MAP is ahead is the unimodal steps, 1.637 vs 1.687, where
    the objection to the mean does not apply. That reverses what I was going to
    write about the previous bullet and it is worked through below;
  - SIS reaches `ESS < 2` at `t = 3` with 200 particles and at `t = 3` with 2000,
    so a tenfold increase buys nothing, which is the sharpest form the argument
    for resampling takes.

What today is not: any claim about which resampling scheme to use, about the
effective sample size as a diagnostic rather than a description, or about the
particle filter beating a Gaussian filter. The third is day 3's and it needs the
EKF and UKF actually written to mean anything.

Run: `python day1_bootstrap_filter.py`
"""

import numpy as np


# --- model ------------------------------------------------------------------

Q = 10.0          # process noise variance
R = 1.0           # observation noise variance
X0_VAR = 5.0      # prior variance at t = 0, mean zero


def transition_mean(x, t):
    """`f(x, t) = 0.5 x + 25 x / (1 + x^2) + 8 cos(1.2 t)`.

    Three terms doing three different things. The linear part is a mild
    contraction. The rational part is the interesting one - it is odd, peaks near
    `|x| = 1` and decays like `25 / x` after that, so it pushes small states
    outward and leaves large ones alone. The cosine is a deterministic forcing
    with period `2 pi / 1.2 ~ 5.24` steps, which is what stops the model from
    settling into a stationary regime where the ambiguity below would average out.

    Note it is a function of `t` and not only of `x`, so the transition kernel is
    time-varying and nothing here can be precomputed once.
    """
    return 0.5 * x + 25.0 * x / (1.0 + x * x) + 8.0 * np.cos(1.2 * t)


def observation_mean(x):
    """`h(x) = x^2 / 20`.

    Even, and that is the whole point. `h(x) = h(-x)`, so the likelihood
    `p(y | x)` is symmetric about zero and an observation constrains `|x|` while
    saying nothing about the sign. Two states related by a sign flip are
    observationally identical *at that instant*, and the only thing that can
    separate them is the prior arriving from the dynamics.

    Since `f` is odd in `x` apart from the forcing term, even the dynamics are
    close to sign-symmetric when the forcing is small, which is why the ambiguity
    survives for several steps at a time rather than being resolved immediately.
    """
    return x * x / 20.0


def simulate(n_steps, seed):
    """Draw a state trajectory and its observations."""
    rng = np.random.default_rng(seed)

    x = np.sqrt(X0_VAR) * rng.standard_normal()
    states = np.empty(n_steps)
    observations = np.empty(n_steps)

    for i in range(n_steps):
        t = i + 1
        x = transition_mean(x, t) + np.sqrt(Q) * rng.standard_normal()
        states[i] = x
        observations[i] = observation_mean(x) + np.sqrt(R) * rng.standard_normal()

    return states, observations


# --- the instrument: an exact filter by numerical integration ---------------


def grid_filter(observations, lo=-35.0, hi=35.0, n_grid=2001):
    """Filtering posteriors `p(x_t | y_{1:t})` on a fixed grid, by quadrature.

    The two recursions written out:

        predict:  p(x_t | y_{1:t-1})  =  int N(x_t; f(x', t), Q) p(x' | y_{1:t-1}) dx'
        update:   p(x_t | y_{1:t})    ∝  N(y_t; h(x_t), R) p(x_t | y_{1:t-1})

    The predict step is a matrix-vector product once the integral is a Riemann
    sum: element `(i, j)` of the kernel is the density of landing on grid point
    `i` from grid point `j`, so a full step is `O(n_grid^2)` and the whole run is
    `O(n_steps n_grid^2)`. That is why this does not generalize - in `d`
    dimensions the grid is `n^d` and the kernel is `n^2d` - and it is exactly why
    the method the rest of the project is about exists.

    Two things worth being explicit about, since everything today is scored
    against this:

      - the truncation to `[lo, hi]` is not free. Any mass the true posterior puts
        outside the window is silently dropped and the renormalization redivides
        it among what is left. The window is checked below against the actual
        trajectory rather than assumed wide enough.
      - the normalization is by `sum * dx`, so `post` is a *density* and not a
        probability vector, and every expectation below carries the `dx`. Getting
        this wrong is invisible in the mean (the errors cancel in a ratio) and
        very visible in the densities, which is the reason the mode-height
        comparisons later are done on ratios of densities at the same `t`.

    Returns `(grid, posteriors)` with `posteriors[i]` the density at step `i + 1`.
    """
    grid = np.linspace(lo, hi, n_grid)
    dx = grid[1] - grid[0]

    posterior = np.exp(-0.5 * grid ** 2 / X0_VAR)
    posterior /= posterior.sum() * dx

    out = np.empty((len(observations), n_grid))

    for i, y in enumerate(observations):
        t = i + 1

        centres = transition_mean(grid, t)
        offsets = grid[:, None] - centres[None, :]
        kernel = np.exp(-0.5 * offsets ** 2 / Q) / np.sqrt(2.0 * np.pi * Q)
        predicted = kernel @ posterior * dx

        likelihood = np.exp(-0.5 * (y - observation_mean(grid)) ** 2 / R)
        posterior = predicted * likelihood
        posterior /= posterior.sum() * dx

        out[i] = posterior

    return grid, out


def grid_mean(grid, posteriors):
    dx = grid[1] - grid[0]
    return (posteriors * grid).sum(axis=1) * dx


def grid_map(grid, posteriors):
    return grid[posteriors.argmax(axis=1)]


def count_modes(grid, density, relative_floor=0.05):
    """Local maxima of `density` carrying at least `relative_floor` of the peak.

    The floor is doing real work and is not a cosmetic filter. On a fine grid the
    quadrature produces small numerical wiggles in the far tails, and every one of
    them is a local maximum; without a floor this counts noise. With the floor at
    5% of the peak it counts structure, and the number it returns is stable to the
    grid resolution, which is checked below rather than hoped for.

    Returns `(locations, heights)`.
    """
    interior = density[1:-1]
    left = density[:-2]
    right = density[2:]

    peaks = (interior > left) & (interior >= right) & (interior > relative_floor * density.max())
    where = np.flatnonzero(peaks) + 1

    return grid[where], density[where]


def density_at(grid, density, x):
    """Density at an arbitrary point, by nearest grid cell."""
    j = int(np.clip(np.searchsorted(grid, x), 0, len(grid) - 1))
    return density[j]


# --- sequential importance sampling, and the one line that fixes it ---------


def particle_filter(observations, n_particles, seed, resample=True):
    """Bootstrap filter (`resample=True`) or plain SIS (`resample=False`).

    The bootstrap proposal is the transition prior: propose `x_t^i ~ p(x_t |
    x_{t-1}^i)` and the incremental weight collapses to the likelihood
    `p(y_t | x_t^i)`, because the proposal and the prior cancel. That is the
    cheapest possible choice and it is the reason this filter is three lines - it
    is also the reason it degenerates when the likelihood is much narrower than
    the prior, since the proposal has not looked at `y_t` at all. Here
    `R = 1` against `Q = 10`, so the likelihood is roughly three times narrower
    than the one-step prior spread, and that is the regime where the choice
    starts to hurt.

    With `resample=False` this is sequential importance sampling: weights
    multiply across time, and the product of `t` likelihood ratios has a variance
    that grows geometrically in `t`. One particle ends up carrying essentially all
    the weight and the other `N - 1` cost compute and contribute nothing. The
    standard summary of that is the effective sample size

        ESS = 1 / sum_i w_i^2

    which is `N` for uniform weights and `1` when one weight is `1` and the rest
    are `0`. It is a description of the weight vector and not a measurement of
    accuracy - a filter can have `ESS = N` and be wrong, if every particle is in
    the wrong place - and day 2 is about how far that description can be pushed.

    Weights are carried in logs and normalized with a max-subtraction, not by
    `log` of the normalized weights. The difference matters: after a few SIS steps
    the smallest normalized weights underflow to exactly zero, `log(0)` is `-inf`,
    and while `-inf` happens to propagate correctly through the arithmetic here it
    raises a warning and stops being obviously correct. Carrying the unnormalized
    log-weights and subtracting the max keeps everything finite and is one
    operation shorter.

    Returns `(means, ess, particles_final)`.
    """
    rng = np.random.default_rng(seed)

    particles = np.sqrt(X0_VAR) * rng.standard_normal(n_particles)
    log_weights = np.zeros(n_particles)

    means = np.empty(len(observations))
    ess = np.empty(len(observations))

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

        means[i] = float(weights @ particles)
        ess[i] = 1.0 / float(weights @ weights)

        if resample:
            # multinomial: N independent draws from the categorical on weights.
            # the identity being approximated is that the weighted empirical
            # measure and an unweighted sample from it represent the same
            # distribution, so this is the resampling scheme that follows directly
            # from the definition. it is also the highest-variance one of the
            # standard three, and day 2 is about that gap.
            index = rng.choice(n_particles, size=n_particles, replace=True, p=weights)
            particles = particles[index]
            log_weights = np.zeros(n_particles)
        else:
            log_weights = np.log(weights + 1e-300)

    return means, ess, particles


def particle_density(particles, grid, bandwidth):
    """Gaussian kernel density estimate of an equally-weighted particle set.

    Only used to put the particle posterior and the grid posterior on the same
    axis for a shape comparison. The bandwidth is a free parameter and no
    conclusion below rests on its value - the comparisons that matter are of
    means and of mode counts, both of which are computed from the particles
    directly.
    """
    offsets = (grid[:, None] - particles[None, :]) / bandwidth
    return np.exp(-0.5 * offsets ** 2).sum(axis=1) / (
        len(particles) * bandwidth * np.sqrt(2.0 * np.pi)
    )


# --- run --------------------------------------------------------------------

if __name__ == "__main__":
    n_steps = 100
    states, observations = simulate(n_steps, seed=7)

    print("univariate nonlinear growth model")
    print(f"  steps                     : {n_steps}")
    print(f"  state range               : [{states.min():.1f}, {states.max():.1f}]")
    print(f"  Q = {Q}, R = {R}")

    grid, posteriors = grid_filter(observations)
    dx = grid[1] - grid[0]

    # the truncation window has to actually contain the posterior, or every number
    # below is computed on a redistributed remainder. checked two ways: the true
    # trajectory is inside it, and the mass in the outermost cells is negligible.
    assert states.min() > grid[0] + 5.0 and states.max() < grid[-1] - 5.0
    edge_mass = (posteriors[:, :20].sum(axis=1) + posteriors[:, -20:].sum(axis=1)) * dx
    print(f"  grid                      : {len(grid)} points on "
          f"[{grid[0]:.0f}, {grid[-1]:.0f}], dx = {dx:.4f}")
    print(f"  max edge mass over steps  : {edge_mass.max():.2e}")
    assert edge_mass.max() < 1e-6

    # --- how often is the posterior actually bimodal? ------------------------
    #
    # the reason for the whole project, stated as a number rather than as a
    # property of the model. "the observation is even so the posterior is
    # bimodal" is true of the *likelihood* at every step; whether it survives
    # into the posterior depends on whether the prior arriving from the dynamics
    # already prefers one sign, and that is not something to reason about from
    # the equations.
    mode_counts = np.array([len(count_modes(grid, p)[0]) for p in posteriors])
    bimodal = mode_counts >= 2

    print(f"\nposterior shape")
    print(f"  mode counts               : "
          f"{ {int(k): int(v) for k, v in zip(*np.unique(mode_counts, return_counts=True))} }")
    print(f"  bimodal steps             : {bimodal.sum()} / {n_steps} "
          f"({bimodal.mean():.0%})")

    assert bimodal.sum() > 0.15 * n_steps, bimodal.sum()

    # and the count has to be a fact about the posterior rather than about the
    # mesh. re-running on a coarser grid has to give the same answer at almost
    # every step, or the 5% floor is picking up quadrature noise.
    coarse_grid, coarse_posteriors = grid_filter(observations, n_grid=1001)
    coarse_counts = np.array([len(count_modes(coarse_grid, p)[0]) for p in coarse_posteriors])
    agreement = float((coarse_counts == mode_counts).mean())
    print(f"  mode count agreement at half resolution: {agreement:.0%}")
    assert agreement > 0.9, agreement

    # the two modes should be near-reflections of each other, since the
    # likelihood is symmetric and only the prior breaks the tie. if they are not,
    # the bimodality is coming from somewhere other than the sign ambiguity and
    # the story about `h` being even is decoration.
    reflection_error = []
    for p in posteriors[bimodal]:
        locations, _ = count_modes(grid, p)
        lo_mode, hi_mode = locations[0], locations[-1]
        reflection_error.append(abs(lo_mode + hi_mode) / max(abs(lo_mode), abs(hi_mode)))
    reflection_error = np.array(reflection_error)
    print(f"  |x_lo + x_hi| / |x| across bimodal steps: "
          f"median {np.median(reflection_error):.3f}, "
          f"90th pct {np.quantile(reflection_error, 0.9):.3f}")

    # --- the posterior mean sits where there is no mass ----------------------
    mean_estimate = grid_mean(grid, posteriors)
    map_estimate = grid_map(grid, posteriors)

    height_ratio = np.array([
        density_at(grid, p, m) / p.max() for p, m in zip(posteriors, mean_estimate)
    ])

    print(f"\ndensity at the reported mean, relative to the peak")
    print(f"  unimodal steps            : {height_ratio[~bimodal].mean():.3f}")
    print(f"  bimodal steps             : {height_ratio[bimodal].mean():.3f}")
    print(f"  worst bimodal step        : {height_ratio[bimodal].min():.2e}")

    # the summary statistic lands in a region the posterior assigns essentially
    # nothing to. that is the concrete version of "a Gaussian belief is the wrong
    # object here" and it is the reason the EKF cannot be fixed by a better
    # linearization on day 3.
    assert height_ratio[bimodal].mean() < 0.5 * height_ratio[~bimodal].mean()

    # --- and it is still the better point estimate ---------------------------
    #
    # this is the one i had backwards, and it was backwards in a way that would
    # have quietly shaped the next three days.
    #
    # i expected the measurement above to be an indictment of the posterior mean:
    # it reports a value the posterior calls impossible, so a mode should be the
    # honest summary and should score better. it does not. the mean wins on RMSE
    # overall, and - this is the part that actually reverses the argument - it
    # wins *by the largest margin on exactly the bimodal steps*, where the "it
    # reports a point with no mass" objection is at its strongest. where the two
    # are near-tied, and where the MAP is very slightly ahead, is the unimodal
    # steps, where the objection does not apply at all.
    #
    # in hindsight this is not a surprise so much as a definition: the posterior
    # mean is the minimizer of expected squared error, so measuring with RMSE and
    # then being surprised the mean wins is asking a question whose answer was
    # fixed by the choice of question. what the measurement genuinely establishes
    # is narrower and more useful - "sits in a region of no mass" and "is a bad
    # estimate" are different claims, and the first does not imply the second
    # under any loss that averages over the posterior.
    #
    # so the argument for representing the shape cannot be an argument about
    # point-estimate error, and today's honest position is that i have not yet
    # measured anything a Gaussian filter could not also report. the quantity
    # that separates them has to be one the posterior mean cannot express -
    # interval coverage, the probability of a sign, the likelihood of the data -
    # and all three are later days. worth writing down now, before day 3 is
    # tempted to declare victory on an RMSE table.
    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    print(f"\nRMSE against the true state")
    print(f"  posterior mean, all       : {rmse(mean_estimate, states):.3f}")
    print(f"  posterior MAP , all       : {rmse(map_estimate, states):.3f}")
    print(f"  posterior mean, bimodal   : {rmse(mean_estimate[bimodal], states[bimodal]):.3f}")
    print(f"  posterior MAP , bimodal   : {rmse(map_estimate[bimodal], states[bimodal]):.3f}")
    print(f"  posterior mean, unimodal  : {rmse(mean_estimate[~bimodal], states[~bimodal]):.3f}")
    print(f"  posterior MAP , unimodal  : {rmse(map_estimate[~bimodal], states[~bimodal]):.3f}")

    assert rmse(mean_estimate, states) < rmse(map_estimate, states)
    assert rmse(mean_estimate[bimodal], states[bimodal]) < rmse(map_estimate[bimodal], states[bimodal])

    # --- SIS, and where it dies ---------------------------------------------
    print(f"\nsequential importance sampling (no resampling)")
    for n_particles in (200, 2000):
        _, ess, _ = particle_filter(observations, n_particles, seed=1, resample=False)
        first_dead = int(np.argmax(ess < 2.0)) + 1
        print(f"  N = {n_particles:5d}  ESS at t=1,2,3: "
              f"{ess[0]:8.1f} {ess[1]:8.1f} {ess[2]:8.2f}   "
              f"first t with ESS < 2: {first_dead}   median ESS: {np.median(ess):.2f}")

        # the collapse is not gradual and it is not a large-N problem: a tenfold
        # increase in particles buys a step or two, because the weight variance
        # grows geometrically in t and the sample size enters logarithmically.
        assert first_dead <= 6, (n_particles, first_dead)
        assert np.median(ess) < 3.0

    # --- the bootstrap filter ------------------------------------------------
    #
    # the comparison that matters is not against the true state - that number is
    # dominated by the posterior's own spread and is nearly the same for any
    # correct filter - but against the *exact posterior mean*, which is the only
    # thing a filter can be blamed for missing.
    print(f"\nbootstrap filter (multinomial resampling)")
    print(f"  {'N':>6}  {'RMSE vs truth':>14}  {'RMSE vs exact mean':>19}  {'median ESS':>11}")

    errors_vs_exact = {}
    for n_particles in (200, 2000, 20000):
        means, ess, _ = particle_filter(observations, n_particles, seed=1, resample=True)
        errors_vs_exact[n_particles] = rmse(means, mean_estimate)
        print(f"  {n_particles:6d}  {rmse(means, states):14.3f}  "
              f"{rmse(means, mean_estimate):19.3f}  "
              f"{np.median(ess) / n_particles:10.0%}")

    # the whole argument for the grid instrument, in two asserts. the error
    # against the truth barely moves with N, because almost all of it is the
    # posterior's own uncertainty and no filter can remove it. the error against
    # the exact posterior does move, because that part is approximation error and
    # is the only part the method is responsible for.
    assert errors_vs_exact[20000] < 0.5 * errors_vs_exact[200]
    assert errors_vs_exact[20000] < 0.25 * rmse(mean_estimate, states)

    # measured: 3.781 / 3.838 / 3.891 against the truth, which is not even
    # monotone in N, against 0.751 / 0.201 / 0.122 against the exact posterior.
    # the first column is a property of the problem and the second is a property
    # of the method, and without the grid there is only the first one.
    #
    # the second column's rate is worth one caveat rather than a claim. Monte
    # Carlo says the error should fall like N^{-1/2}, so a hundredfold increase
    # should buy about 10x and buys 6.2x - with the first decade (3.7x) ahead of
    # the rate and the second (1.65x) well behind it. That is one seed and one
    # trajectory, so the ratios have error bars i have not measured, and reading
    # a convergence rate off three points from one seed is the kind of thing day
    # 2 exists to stop. Recorded, not concluded.

    # and it recovers the shape, not only the mean. take the step where the exact
    # posterior is most sharply two-lobed and check the particle cloud is two
    # lobed in the same places - the mean agreeing is compatible with the cloud
    # being one blob straddling the gap, which is exactly the failure mode the
    # whole project is about, so it needs its own check.
    #
    # measured at t = 64: exact modes at -15.75 and +15.86, particle modes at the
    # same two grid cells, and the posterior mean at 8.04 with 2.7e-20 of the peak
    # density under it. the mass splits 25/75 across that point, so the mean is
    # not even near the middle - it is a weighted average of two lobes that
    # disagree about the sign, pulled to the heavier one and landing in the void
    # between them. this is the picture the rest of the project is arguing about
    # and it is worth having it as three numbers rather than as a sentence.
    worst = int(np.argmin(np.where(bimodal, height_ratio, np.inf)))
    means, _, cloud = particle_filter(observations[: worst + 1], 20000, seed=3, resample=True)
    cloud_density = particle_density(cloud, grid, bandwidth=0.6)

    exact_modes, _ = count_modes(grid, posteriors[worst])
    cloud_modes, _ = count_modes(grid, cloud_density, relative_floor=0.15)

    print(f"\nshape at the most sharply bimodal step (t = {worst + 1})")
    print(f"  exact modes at            : {np.round(exact_modes, 2)}")
    print(f"  particle modes at         : {np.round(cloud_modes, 2)}")
    print(f"  exact posterior mean      : {mean_estimate[worst]:.2f}   "
          f"density there: {height_ratio[worst]:.2e} of peak")
    print(f"  fraction of particles on each side of the mean: "
          f"{(cloud < mean_estimate[worst]).mean():.2f} / "
          f"{(cloud > mean_estimate[worst]).mean():.2f}")

    assert len(cloud_modes) >= 2, cloud_modes
    for location in exact_modes:
        assert np.min(np.abs(cloud_modes - location)) < 1.5, (location, cloud_modes)

    # both lobes are actually populated - a filter that had collapsed onto one
    # sign would still put two bumps in a KDE if the bandwidth were small enough,
    # so the mass split is the check with teeth.
    left_share = float((cloud < mean_estimate[worst]).mean())
    assert 0.1 < left_share < 0.9, left_share

    print("\nday 1 checks passed")
