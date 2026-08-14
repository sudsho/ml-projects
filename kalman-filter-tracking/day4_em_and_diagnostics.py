"""
Day 4, and the last one: learning `Q` and `R` by EM, filtering through gaps in
the data, and finally asking whether the filter's error bars are honest.

Days 1-3 all handed the filter the exact matrices the data was generated from.
That was deliberate - it meant every disagreement was an implementation bug and
the consistency checks were checks on code rather than on modelling. It is also
the one assumption nobody gets in practice. `H` and `F` are usually known,
because they are geometry and calendar arithmetic; `Q` and `R` are almost never
known, because they are a confession about how wrong the dynamics are.

EM is the same algorithm the HMM project used for Baum-Welch, and structurally it
is identical: the E-step runs forward-backward and produces expected sufficient
statistics, the M-step maximizes as if those expectations were data. Only the
statistics change. Discrete Baum-Welch counted expected transitions `i -> j` and
expected emissions; here the counts are second moments, `E[x_t x_t']` and
`E[x_t x_{t-1}']`, which is exactly why day 3 built the lag-one smoothed
covariance and left it unused. It is the whole cross-moment term, and no amount
of smoothed marginals substitutes for it.

Three things came out of today that were not on the plan.

**The unconstrained M-step buys 1.9 nats and pays 20x in two directions of `Q`.**
`Q` for a constant-velocity model is not a free symmetric matrix - it is one
scalar, the acceleration variance, times a structure day 1 derives from `dt`
alone. Estimating all ten free parameters instead recovers `Q` to within a factor
of 20 in its two smallest eigendirections, while the one-parameter version is
uniformly within 16%. The free fit still wins on likelihood, by 1.9 nats over
1000 steps, and that tiny margin is the entire explanation: the likelihood is
nearly flat in exactly the directions the free estimator gets wrong, so the extra
freedom is spent fitting noise where nothing pushes back. See
`run_em_experiment`.

**Through a blackout the filter's covariance ramps and the smoother's arches.**
With no observation the update is skipped and the filter runs open-loop, so its
reported uncertainty grows monotonically - across a 60-step gap the position
covariance goes from 0.56 to 85. The smoother, which sees both ends, produces a
symmetric arch instead: 0.40 up to 2.19 in the middle and back to 0.40. Two
pictures of the same interval, and the difference is entirely about which
direction information is allowed to flow. Sampled at the middle of a gap across
80 independent runs, NEES comes out at 4.41 against a nominal 4 - the covariance
there is a pure prediction, corrected by nothing, and it is still telling the
truth. See `run_missing_experiment`.

**The NEES time-average is not a valid test, and I had been running it as one.**
Averaging NEES over 200 steps of one run gives the right number - 4.01 against a
state dimension of 4, unbiased - and a confidence interval that is wrong by a
factor of 3.4. Consecutive estimation errors are strongly correlated at
`dt = 0.1`, so 200 time samples carry the information of about 17 independent
ones, and the naive chi-square interval covers 37% of runs instead of 95%. NIS
does not have this problem at all: innovations are a martingale difference
sequence, hence white, so time-averaging them is legitimate and the same interval
covers 94%. Which leaves NIS as the weaker diagnostic in what it can see and
the stronger one in how it may be used - it needs no ground truth *and* no
repeated runs, and that is why it is the one that gets used. Day 1's NIS check
was sound for a reason day 1 did not state. See `run_validity_experiment`.
"""

import numpy as np

from day1_kalman_filter import (
    constant_velocity_model,
    gaussian_log_density,
    predict,
    rmse,
    simulate,
)
from day2_numerical_stability import symmetrize, update_joseph
from day3_rts_smoother import rts_smoother


def kalman_filter_missing(observations, transition, process_noise, observation,
                          observation_noise, initial_mean, initial_cov):
    """Day 2's filter, with `nan` rows in `observations` meaning "no data here".

    The handling is a deletion rather than an addition, which is the pleasant
    part. A missing observation is not a special case to be modelled - it is a
    step where the update does not happen, so the filtered belief *is* the
    predicted one and the recursion continues. Nothing needs to be imputed, which
    matters because the obvious hack of substituting the predicted observation is
    not neutral: it would leave the mean alone and shrink the covariance, telling
    the filter it had learned something from a measurement that does not exist.

    The likelihood skips those steps too. The prediction-error decomposition
    factorizes `p(y_1..y_T)` into one-step-ahead densities, and a step with no
    `y` contributes no factor. It still contributes to the *state* recursion, so
    the next observed step's predictive density is correctly widened by the gap.

    The innovation is recorded as `nan` and the innovation covariance is recorded
    anyway, as `H P^- H' + R` - the predictive covariance of an observation that
    was never taken. Meaningless for NIS, which is why the mask is returned
    alongside, and exactly right for asking how uncertain the filter would have
    been about a measurement it did not get.
    """
    n_steps = observations.shape[0]
    state_dim = transition.shape[0]
    obs_dim = observation.shape[0]

    observed = ~np.any(np.isnan(observations), axis=1)

    filtered_means = np.zeros((n_steps, state_dim))
    filtered_covs = np.zeros((n_steps, state_dim, state_dim))
    predicted_means = np.zeros((n_steps, state_dim))
    predicted_covs = np.zeros((n_steps, state_dim, state_dim))
    innovations = np.full((n_steps, obs_dim), np.nan)
    innovation_covs = np.zeros((n_steps, obs_dim, obs_dim))

    mean = np.asarray(initial_mean, dtype=float)
    cov = np.asarray(initial_cov, dtype=float)
    log_likelihood = 0.0

    for t in range(n_steps):
        mean, cov = predict(mean, cov, transition, process_noise)
        predicted_means[t] = mean
        predicted_covs[t] = symmetrize(cov)

        if observed[t]:
            mean, cov, innovation, innovation_cov = update_joseph(
                mean, cov, observations[t], observation, observation_noise
            )
            innovations[t] = innovation
            log_likelihood += gaussian_log_density(innovation, innovation_cov)
        else:
            innovation_cov = symmetrize(
                observation @ cov @ observation.T + observation_noise
            )

        innovation_covs[t] = innovation_cov
        filtered_means[t] = mean
        filtered_covs[t] = symmetrize(cov)

    return {
        "filtered_means": filtered_means,
        "filtered_covs": filtered_covs,
        "predicted_means": predicted_means,
        "predicted_covs": predicted_covs,
        "innovations": innovations,
        "innovation_covs": innovation_covs,
        "log_likelihood": log_likelihood,
        "observed": observed,
    }


def drop_observations(observations, gaps, rng):
    """Blank out `gaps` contiguous stretches of observations, returning a copy.

    Contiguous rather than scattered on purpose. Independently dropping 20% of
    the steps is nearly invisible to a filter running at `dt = 0.1` - the state
    barely moves in one step and the next observation recovers everything. A
    handful of long blackouts is what actually stresses the covariance recursion,
    and it is the realistic failure too: sensors drop out in stretches, not in
    Bernoulli draws.

    One gap per equal-width block, so they cannot overlap each other or land in
    the opening steps where the diffuse prior is still resolving.
    """
    masked = np.array(observations, dtype=float)
    n_steps = masked.shape[0]

    span = (n_steps - n_steps // 4) // len(gaps)
    for index, length in enumerate(gaps):
        block_start = n_steps // 4 + index * span
        start = block_start + int(rng.integers(0, max(1, span - length)))
        masked[start:start + length] = np.nan

    return masked


def smoothed_moments(smoothed, observations, transition, observation):
    """The E-step's output: expected sufficient statistics for the M-step.

    Two accumulations, and the first is the reason day 3 ended the way it did.

    The transition residual is `E[(x_t - F x_{t-1})(x_t - F x_{t-1})']` summed
    over steps, expanded under the smoothed posterior into

        S_tt - S_t0 F' - F S_t0' + F S_00 F'

    with `S_tt = P_t^s + m_t^s m_t^s'` and `S_t0 = P_{t,t-1}^s + m_t^s m_{t-1}^s'`.
    The middle terms need the lag-one smoothed covariance and there is no route
    to them from the marginals - `E[x_t x_{t-1}']` is an off-diagonal block of
    the joint posterior over the trajectory, and the marginals are the diagonal.
    Dropping it and using `m_t^s m_{t-1}^s'` alone is the standard bug here, and
    it does not announce itself: EM still converges, still monotonically, to the
    wrong answer, because it remains a consistent maximizer of a different
    objective. The monotonicity assertion at the bottom of this file is what
    catches it, which is the only reason that assertion is worth writing.

    The observation residual is easier - `E[(y_t - H x_t)(y_t - H x_t)']` is
    `(y_t - H m_t^s)(y_t - H m_t^s)' + H P_t^s H'`, and the second term is the
    part a plug-in estimate would omit. Omitting it biases `R` downward by
    exactly the posterior uncertainty about the state, the same error as
    estimating a variance around a fitted mean and dividing by `n`.

    Missing steps contribute to the transition sum and not to the observation
    sum, which needs no handling beyond the mask: the dynamics happened whether
    or not anyone was watching.

    The pair `(x_1, x_0)` is skipped. `x_0` is the prior rather than a state the
    smoother estimates, so its cross-moment is not in `lag_one_covs`, and with
    the prior held fixed the one dropped term is `O(1/T)`.
    """
    means = smoothed["smoothed_means"]
    covs = smoothed["smoothed_covs"]
    lag_one = smoothed["lag_one_covs"]

    n_steps = means.shape[0]
    state_dim = means.shape[1]
    obs_dim = observation.shape[0]
    observed = ~np.any(np.isnan(observations), axis=1)

    transition_residual = np.zeros((state_dim, state_dim))
    for t in range(1, n_steps):
        second_tt = covs[t] + np.outer(means[t], means[t])
        second_t0 = lag_one[t - 1] + np.outer(means[t], means[t - 1])
        second_00 = covs[t - 1] + np.outer(means[t - 1], means[t - 1])

        cross = second_t0 @ transition.T
        transition_residual += (
            second_tt - cross - cross.T + transition @ second_00 @ transition.T
        )

    observation_residual = np.zeros((obs_dim, obs_dim))
    for t in range(n_steps):
        if not observed[t]:
            continue
        residual = observations[t] - observation @ means[t]
        observation_residual += (
            np.outer(residual, residual) + observation @ covs[t] @ observation.T
        )

    return {
        "transition_residual": symmetrize(transition_residual),
        "observation_residual": symmetrize(observation_residual),
        "n_transitions": n_steps - 1,
        "n_observed": int(np.sum(observed)),
    }


def m_step_free(moments, templates=None):
    """Unconstrained M-step: `Q` and `R` are whatever averaging says they are.

    Both are the expected residual second moment over its count, which is what
    maximum likelihood for a Gaussian covariance always reduces to. `templates`
    is accepted and ignored so the two M-steps are interchangeable in `em_fit`.

    Ten free parameters in `Q` and three in `R` for a 4-state, 2-observation
    model. Not many in absolute terms, and far too many given what the data can
    see, which is the point `run_em_experiment` makes.
    """
    process_noise = moments["transition_residual"] / moments["n_transitions"]
    observation_noise = moments["observation_residual"] / moments["n_observed"]
    return symmetrize(process_noise), symmetrize(observation_noise)


def m_step_scaled(moments, templates):
    """Constrained M-step: `Q = sigma_a^2 G`, `R = sigma_r^2 I`, two scalars total.

    `G` is the structure `constant_velocity_model` builds from `dt` alone - the
    `[[dt^3/3, dt^2/2], [dt^2/2, dt]]` block per axis that comes from integrating
    one unknown acceleration across the step. It is not an approximation or a
    convenient prior. If the model is "position and velocity share a random
    acceleration", that matrix is a derivation, and only its scale is unknown.

    Maximizing over `sigma^2` with the shape fixed is one line of calculus.
    Writing `M` for the summed residual and `n` for the state dimension,

        d/d(sigma^2) [ -0.5 * ( T n log sigma^2 + tr(G^-1 M) / sigma^2 ) ] = 0
        =>  sigma^2 = tr(G^-1 M) / (T n)

    The trace against `G^-1` does the real work: it measures the residual in the
    metric `G` defines, so error in a direction `G` says should be small counts
    for more. `G` here has condition number 1206, so those directions are very
    small indeed and the weighting is severe. The unconstrained estimator has no
    such opinion and lets every direction find its own level - which is the
    freedom that hurts it, since two of those directions are ones the data has
    almost nothing to say about.
    """
    q_template, r_template = templates

    state_dim = q_template.shape[0]
    obs_dim = r_template.shape[0]

    q_scale = np.trace(
        np.linalg.solve(q_template, moments["transition_residual"])
    ) / (state_dim * moments["n_transitions"])
    r_scale = np.trace(
        np.linalg.solve(r_template, moments["observation_residual"])
    ) / (obs_dim * moments["n_observed"])

    return q_scale * q_template, r_scale * r_template


def em_fit(observations, transition, observation, initial_mean, initial_cov,
           process_noise_init, observation_noise_init, m_step=m_step_free,
           templates=None, tolerance=1e-6, max_iterations=2000):
    """Alternate E and M until the likelihood stops moving, recording every step.

    The recorded likelihood is the one the *current* parameters achieve, measured
    before they are updated, so `history[k]` is scored under iteration `k`'s
    parameters and monotonicity of the list is the real EM guarantee rather than
    a restatement of the update. Baum-Welch got the same treatment on the HMM
    project for the same reason: a monotone likelihood is nearly the only thing
    that separates a working EM from a plausible-looking broken one, and a broken
    E-step shows up as one non-monotone step early rather than as divergence.

    Stopping on a relative tolerance rather than a fixed count, because the two
    parameterizations need very different budgets and fixing the count would have
    compared a converged fit against an unconverged one. The first version of
    this file did exactly that at 80 iterations and drew the opposite conclusion
    from the one below - the structured fit was still 3x from its own optimum and
    looked like the worse estimator. Iterations-to-converge is returned as a
    result rather than hidden as a setting, since the gap between the two is
    itself a finding.
    """
    process_noise = np.array(process_noise_init, dtype=float)
    observation_noise = np.array(observation_noise_init, dtype=float)

    history = []
    previous = -np.inf

    for _ in range(max_iterations):
        filtered = kalman_filter_missing(
            observations, transition, process_noise, observation,
            observation_noise, initial_mean, initial_cov,
        )
        log_likelihood = filtered["log_likelihood"]
        history.append(log_likelihood)

        if abs(log_likelihood - previous) < tolerance * abs(log_likelihood):
            break
        previous = log_likelihood

        smoothed = rts_smoother(filtered, transition)
        moments = smoothed_moments(smoothed, observations, transition, observation)
        process_noise, observation_noise = m_step(moments, templates)

    return {
        "process_noise": process_noise,
        "observation_noise": observation_noise,
        "log_likelihood_history": np.array(history),
        "n_iterations": len(history),
    }


def covariance_log_spread(estimate, truth):
    """How wrong a covariance estimate is, in the only scale-free way available.

    `max |log lambda|` over the eigenvalues of `truth^-1 estimate`. Every
    eigenvalue is 1 exactly when the two matrices agree, and the measure is
    invariant to any linear change of coordinates - which is required here,
    because `Q`'s entries span four orders of magnitude and a Frobenius
    comparison would report only the velocity block. Two directions wrong by 20x
    are invisible in `||Q_hat - Q||_F` when the true value in those directions is
    `1e-4`, and they are the whole story.

    Reading it: 0 is exact, 0.14 is "everything within 16%", 3.0 is "some
    direction is off by a factor of 20".
    """
    chol = np.linalg.cholesky(truth)
    whitened = np.linalg.solve(chol, np.linalg.solve(chol, estimate).T).T
    return float(np.max(np.abs(np.log(np.linalg.eigvalsh(symmetrize(whitened))))))


def normalized_errors(states, filter_result):
    """NEES and NIS: the two ways to ask whether the reported covariance is real.

    NEES is `e_t' P_t^-1 e_t` against the true state, and averages the state
    dimension when the filter is consistent. It is the direct check and it is
    unavailable in practice, because it needs ground truth.

    NIS is `v_t' S_t^-1 v_t` on the innovations, and averages the observation
    dimension. It needs nothing but the filter's own output, which is what makes
    it the one that gets used - and it is weaker in a specific way. It only ever
    looks through `H`. A filter can be badly wrong about velocity, which nothing
    observes, while its position predictions stay calibrated; NIS sees the
    predictions and reports nothing.

    Missing steps are excluded from NIS and kept in NEES, which is what makes the
    gap experiment work: NEES is defined at every step whether or not an
    observation arrived, so it can be evaluated *inside* a blackout, where NIS
    has nothing to say at all.
    """
    errors = states - filter_result["filtered_means"]
    nees = np.array(
        [
            errors[t] @ np.linalg.solve(filter_result["filtered_covs"][t], errors[t])
            for t in range(len(errors))
        ]
    )

    observed = filter_result["observed"]
    innovations = filter_result["innovations"]
    nis = np.array(
        [
            innovations[t] @ np.linalg.solve(
                filter_result["innovation_covs"][t], innovations[t]
            )
            for t in range(len(innovations))
            if observed[t]
        ]
    )

    return {"nees": nees, "nis": nis, "observed": observed}


def chi_square_interval(dof, z=1.959963984540054):
    """Two-sided 95% chi-square interval, Wilson-Hilferty, no scipy.

    `(X / k)^(1/3)` is very close to normal with mean `1 - 2/(9k)` and variance
    `2/(9k)`. The cube root is not arbitrary - it is the variance-stabilizing
    transform for the chi-square family, so the approximation improves with `k`
    rather than degrading.

    Applied to a sum: `m` independent NEES values sum to `chi2(m n)`, so the
    interval on their average is this divided by `m`. Everything here depends on
    that independence, which `run_validity_experiment` shows is a real
    restriction rather than a formality.
    """
    scale = 2.0 / (9.0 * dof)
    lower = dof * (1.0 - scale - z * np.sqrt(scale)) ** 3
    upper = dof * (1.0 - scale + z * np.sqrt(scale)) ** 3
    return lower, upper


def _truth(n_steps, seed, accel_noise=0.6, obs_noise=1.2, dt=0.1):
    """Simulate a track from the true model and return it with its matrices."""
    rng = np.random.default_rng(seed)

    matrices = constant_velocity_model(dt=dt, accel_noise=accel_noise, obs_noise=obs_noise)
    transition, process_noise, observation, observation_noise = matrices

    states, observations = simulate(
        transition, process_noise, observation, observation_noise,
        np.array([0.0, 0.0, 2.0, 1.0]), n_steps=n_steps, rng=rng,
    )

    return {
        "states": states,
        "observations": observations,
        "model": matrices,
        "prior": (np.zeros(4), np.diag([5.0, 5.0, 25.0, 25.0])),
        "templates": (process_noise / accel_noise ** 2, observation_noise / obs_noise ** 2),
        "rng": rng,
    }


def run_em_experiment(n_steps=1000, seed=7):
    """Free `Q` against structured `Q`, from a deliberately bad starting point.

    Both start from `Q` inflated 20x and `R` deflated 10x - a start that gets the
    *ratio* wrong by 200x in the direction that matters, telling the filter its
    dynamics are unpredictable and its sensor is excellent, so the initial filter
    chases noise.

    What to look at is not the likelihood, which the free version wins by
    construction, but how much likelihood it wins and what it pays. The eigenvalue
    ratios of `Q_hat` against the true `Q` are the diagnostic: for the structured
    fit they are equal by definition and only their common value is in question,
    while the free fit is free to get them individually wrong.
    """
    truth = _truth(n_steps, seed)
    transition, process_noise, observation, observation_noise = truth["model"]
    initial_mean, initial_cov = truth["prior"]

    process_init = 20.0 * process_noise
    observation_init = 0.1 * observation_noise

    free = em_fit(
        truth["observations"], transition, observation, initial_mean, initial_cov,
        process_init, observation_init, m_step=m_step_free,
    )
    scaled = em_fit(
        truth["observations"], transition, observation, initial_mean, initial_cov,
        process_init, observation_init, m_step=m_step_scaled,
        templates=truth["templates"],
    )

    for fit in (free, scaled):
        fit["q_spread"] = covariance_log_spread(fit["process_noise"], process_noise)
        fit["r_spread"] = covariance_log_spread(fit["observation_noise"], observation_noise)
        fit["q_eigen_ratios"] = np.linalg.eigvalsh(
            symmetrize(np.linalg.solve(process_noise, fit["process_noise"]))
        )

    return {"free": free, "scaled": scaled, "truth": truth}


def gap_boundaries(missing):
    """Start and end indices of each contiguous run of missing observations."""
    boundaries = np.flatnonzero(np.diff(missing.astype(int)) != 0)
    starts = [index + 1 for index in boundaries if missing[index + 1]]
    ends = [index + 1 for index in boundaries if not missing[index + 1]]
    return list(zip(starts, ends))


def gap_midpoint_nees(n_steps, seed, gaps):
    """One NEES sample, from the middle of the longest gap in a fresh run.

    One, not the whole gap. Inside a blackout the error is a random walk with no
    observation pulling on it, so consecutive NEES values there are about as
    correlated as they get anywhere in the project - averaging 60 of them and
    testing the result against a chi-square interval for 60 independent samples
    is the exact mistake `run_validity_experiment` measures below. Taking a
    single value per simulation and repeating the simulation costs more compute
    and produces a test that means what it says.
    """
    truth = _truth(n_steps, seed)
    transition, process_noise, observation, observation_noise = truth["model"]
    initial_mean, initial_cov = truth["prior"]

    masked = drop_observations(truth["observations"], gaps, truth["rng"])
    filtered = kalman_filter_missing(
        masked, transition, process_noise, observation,
        observation_noise, initial_mean, initial_cov,
    )

    spans = gap_boundaries(np.any(np.isnan(masked), axis=1))
    start, end = max(spans, key=lambda span: span[1] - span[0])
    middle = (start + end) // 2

    error = truth["states"][middle] - filtered["filtered_means"][middle]
    return float(error @ np.linalg.solve(filtered["filtered_covs"][middle], error))


def run_missing_experiment(n_steps=1200, seed=11, gaps=(40, 60, 30), n_runs=80):
    """Filter and smooth through three blackouts, and check NEES inside them.

    Two questions. First, is the covariance honest where nothing was measured -
    if NEES holds its nominal value inside a gap, the answer is yes, and that
    conclusion rests on no observation at all, which is as clean as this check
    ever gets. That part is answered across `n_runs` independent simulations
    rather than along one, for the reason in `gap_midpoint_nees`.

    Second, what shape does the uncertainty take. The filter enters a gap with a
    good estimate and extrapolates, so its covariance can only grow, all the way
    to the far edge. The smoother sees both edges and interpolates, so its
    covariance has to come back down - the worst point is the middle, equidistant
    from both anchors. Those are different curves over the same interval, and the
    profile below reports both at the start, middle, and end of each gap in one
    representative run.
    """
    truth = _truth(n_steps, seed)
    transition, process_noise, observation, observation_noise = truth["model"]
    initial_mean, initial_cov = truth["prior"]

    masked = drop_observations(truth["observations"], gaps, truth["rng"])
    missing = np.any(np.isnan(masked), axis=1)

    filtered = kalman_filter_missing(
        masked, transition, process_noise, observation,
        observation_noise, initial_mean, initial_cov,
    )
    smoothed = rts_smoother(filtered, transition)

    states = truth["states"]
    diagnostics = normalized_errors(states, filtered)
    inside, outside = missing, ~missing

    # position-block covariance at the start, middle and end of every gap
    profiles = []
    for start, end in gap_boundaries(missing):
        middle = (start + end) // 2
        profiles.append(
            {
                "length": end - start,
                "filtered_trace": [
                    float(np.trace(filtered["filtered_covs"][index, :2, :2]))
                    for index in (start, middle, end - 1)
                ],
                "smoothed_trace": [
                    float(np.trace(smoothed["smoothed_covs"][index, :2, :2]))
                    for index in (start, middle, end - 1)
                ],
            }
        )

    midpoint_nees = np.array(
        [gap_midpoint_nees(n_steps, 3000 + index, gaps) for index in range(n_runs)]
    )

    return {
        "filtered": filtered,
        "smoothed": smoothed,
        "missing": missing,
        "profiles": profiles,
        "n_missing": int(np.sum(missing)),
        "midpoint_nees": midpoint_nees,
        "midpoint_nees_mean": float(np.mean(midpoint_nees)),
        "n_runs": n_runs,
        "nees_inside": float(np.mean(diagnostics["nees"][inside])),
        "nees_outside": float(np.mean(diagnostics["nees"][outside])),
        "filtered_rmse_inside": rmse(filtered["filtered_means"][inside, :2], states[inside, :2]),
        "smoothed_rmse_inside": rmse(smoothed["smoothed_means"][inside, :2], states[inside, :2]),
        "filtered_rmse_outside": rmse(filtered["filtered_means"][outside, :2], states[outside, :2]),
        "smoothed_rmse_outside": rmse(smoothed["smoothed_means"][outside, :2], states[outside, :2]),
    }


def run_consistency_experiment(n_steps=2000, seed=3, q_factors=(0.01, 1.0, 100.0)):
    """The same data through filters with `Q` scaled wrong: RMSE against NEES.

    `Q` is the filter's statement about how much it distrusts its own dynamics.
    Too small and it believes the constant-velocity model over the sensor; too
    large and it does the reverse. Neither is catastrophic for the point estimate,
    because the gain lands somewhere defensible either way and the observations
    keep pulling it back.

    The covariance has no such safety net. `P` follows the Riccati recursion using
    the model matrices and never touches the data, so it cannot be corrected by
    observations disagreeing with it. A wrong `Q` produces a wrong `P` directly,
    and the two error scales end up orders of magnitude apart.
    """
    truth = _truth(n_steps, seed)
    transition, process_noise, observation, observation_noise = truth["model"]
    initial_mean, initial_cov = truth["prior"]

    rows = []
    for factor in q_factors:
        filtered = kalman_filter_missing(
            truth["observations"], transition, factor * process_noise, observation,
            observation_noise, initial_mean, initial_cov,
        )
        diagnostics = normalized_errors(truth["states"], filtered)
        rows.append(
            {
                "factor": factor,
                "position_rmse": rmse(filtered["filtered_means"][:, :2], truth["states"][:, :2]),
                "velocity_rmse": rmse(filtered["filtered_means"][:, 2:], truth["states"][:, 2:]),
                "mean_nees": float(np.mean(diagnostics["nees"])),
                "mean_nis": float(np.mean(diagnostics["nis"])),
                "log_likelihood": float(filtered["log_likelihood"]),
            }
        )

    return {"rows": rows, "truth": truth}


def run_validity_experiment(n_runs=150, n_steps=250, burn_in=50):
    """Is the consistency test itself valid? For NEES, no. For NIS, yes.

    The chi-square interval assumes the values being averaged are independent.
    Bar-Shalom's NEES test gets that by averaging *across independent runs* at a
    fixed time. Averaging across time within one run - which is the convenient
    thing to do, and what day 1 did for NIS and what I was about to do for NEES -
    substitutes a different assumption entirely, and it is false for NEES.

    The state error is a smooth process. At `dt = 0.1` the target barely moves in
    one step, so `e_t` and `e_{t+1}` are nearly the same vector and their NEES
    values are nearly the same number. Averaging 200 of them does not average 200
    independent things.

    The innovations are different by construction. Under the true model they are a
    martingale difference sequence, hence uncorrelated at every lag - day 1
    checked exactly this and called it whiteness without connecting it to the
    consistency test that depends on it. Whiteness is precisely the licence to
    average NIS over time.

    So both diagnostics are measured two ways here: once across runs at a fixed
    time, which is valid for both, and once across time within a run, which is
    valid for one. The comparison is the empirical coverage of the naive interval
    - it should be 95% and for NEES it is not close.
    """
    ensemble_nees = []
    ensemble_nis = []
    time_average_nees = []
    time_average_nis = []

    probe = n_steps - 1
    for index in range(n_runs):
        truth = _truth(n_steps, 1000 + index)
        transition, process_noise, observation, observation_noise = truth["model"]
        initial_mean, initial_cov = truth["prior"]

        filtered = kalman_filter_missing(
            truth["observations"], transition, process_noise, observation,
            observation_noise, initial_mean, initial_cov,
        )
        diagnostics = normalized_errors(truth["states"], filtered)

        ensemble_nees.append(diagnostics["nees"][probe])
        ensemble_nis.append(diagnostics["nis"][probe])
        time_average_nees.append(float(np.mean(diagnostics["nees"][burn_in:])))
        time_average_nis.append(float(np.mean(diagnostics["nis"][burn_in:])))

    kept = n_steps - burn_in
    report = {"n_runs": n_runs, "n_kept": kept}

    for name, dim, ensemble, averages in [
        ("nees", 4, ensemble_nees, time_average_nees),
        ("nis", 2, ensemble_nis, time_average_nis),
    ]:
        ensemble = np.array(ensemble)
        averages = np.array(averages)

        lower, upper = chi_square_interval(dim * n_runs)
        naive_lower, naive_upper = chi_square_interval(dim * kept)
        naive_sd = np.sqrt(2.0 * dim * kept) / kept

        report[name] = {
            "ensemble_mean": float(np.mean(ensemble)),
            "ensemble_interval": (lower / n_runs, upper / n_runs),
            "time_average_mean": float(np.mean(averages)),
            "naive_interval": (naive_lower / kept, naive_upper / kept),
            "observed_sd": float(np.std(averages)),
            "naive_sd": float(naive_sd),
            "coverage": float(
                np.mean((averages > naive_lower / kept) & (averages < naive_upper / kept))
            ),
        }

    return report


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("=== EM for Q and R, free vs structured (1000 steps) ===")
    em = run_em_experiment()
    free, scaled = em["free"], em["scaled"]

    free_history = free["log_likelihood_history"]
    scaled_history = scaled["log_likelihood_history"]

    print(f"start log-likelihood      : {free_history[0]:.2f}")
    print(f"free    final / iters     : {free_history[-1]:.2f} / {free['n_iterations']}")
    print(f"scaled  final / iters     : {scaled_history[-1]:.2f} / {scaled['n_iterations']}")
    print(f"free    Q log-spread      : {free['q_spread']:.4f}")
    print(f"scaled  Q log-spread      : {scaled['q_spread']:.4f}")
    print(f"free    Q eigen ratios    : {free['q_eigen_ratios']}")
    print(f"scaled  Q eigen ratios    : {scaled['q_eigen_ratios']}")
    print(f"free    R log-spread      : {free['r_spread']:.4f}")
    print(f"scaled  R log-spread      : {scaled['r_spread']:.4f}")

    # the guarantee, and the only check that separates a working E-step from a
    # broken one that still converges. dropping the lag-one term breaks this.
    assert np.all(np.diff(free_history) > -1e-6)
    assert np.all(np.diff(scaled_history) > -1e-6)

    # both improved on the deliberately bad start, and both stopped on tolerance
    # rather than running out of iterations
    assert free_history[-1] > free_history[0]
    assert scaled_history[-1] > scaled_history[0]
    assert free["n_iterations"] < 2000 and scaled["n_iterations"] < 2000

    # the free fit reaches a higher likelihood - it maximizes over a set that
    # contains the structured one, so it cannot do otherwise
    margin = free_history[-1] - scaled_history[-1]
    print(f"likelihood margin, free   : {margin:.2f} nats over 1000 steps")
    assert margin >= -1e-6

    # and the structured fit recovers Q far better. more freedom, better fit,
    # much worse estimate - and the margin above is what the freedom bought.
    assert scaled["q_spread"] < 0.25
    assert free["q_spread"] > 2.0
    assert margin < 5.0

    # the damage is confined to the two directions Q says are nearly empty. the
    # other two the free fit gets about as well as the structured one does.
    assert np.max(free["q_eigen_ratios"]) > 10.0
    assert np.min(free["q_eigen_ratios"]) > 0.5

    # R is a different story - estimated from residuals against coordinates that
    # are directly observed, so both parameterizations recover it
    assert free["r_spread"] < 0.15
    assert scaled["r_spread"] < 0.15

    # the structured fit needs more iterations to get there. one scalar couples
    # every direction, so the slowest mode sets the rate for all of them.
    assert scaled["n_iterations"] > free["n_iterations"]

    print()
    print("=== filtering through blackouts ===")
    gaps = run_missing_experiment()

    print(f"missing steps             : {gaps['n_missing']} of 1200")
    print(f"position RMSE  outside    : {gaps['filtered_rmse_outside']:.4f}")
    print(f"position RMSE  inside     : {gaps['filtered_rmse_inside']:.4f}")
    print(f"smoothed RMSE  outside    : {gaps['smoothed_rmse_outside']:.4f}")
    print(f"smoothed RMSE  inside     : {gaps['smoothed_rmse_inside']:.4f}")
    print(f"time-avg NEES  outside    : {gaps['nees_outside']:.4f}  (not testable)")
    print(f"time-avg NEES  inside     : {gaps['nees_inside']:.4f}  (not testable)")
    print(
        f"gap-midpoint NEES         : {gaps['midpoint_nees_mean']:.4f}  "
        f"over {gaps['n_runs']} independent runs"
    )
    print()
    print("  position-block trace(P) at gap start / middle / end")
    for profile in gaps["profiles"]:
        filtered_trace = profile["filtered_trace"]
        smoothed_trace = profile["smoothed_trace"]
        print(
            f"  len {profile['length']:3d}  filtered {filtered_trace[0]:7.2f} "
            f"{filtered_trace[1]:7.2f} {filtered_trace[2]:7.2f}   "
            f"smoothed {smoothed_trace[0]:6.2f} {smoothed_trace[1]:6.2f} "
            f"{smoothed_trace[2]:6.2f}"
        )

    # open-loop is worse and less certain, both unsurprising
    assert gaps["filtered_rmse_inside"] > gaps["filtered_rmse_outside"]

    # the filter's covariance can only grow inside a gap, so it is monotone from
    # the first missing step to the last, and the growth is large
    for profile in gaps["profiles"]:
        start, middle, end = profile["filtered_trace"]
        assert start < middle < end
        assert end > 10.0 * start

    # the smoother's covariance is an arch instead - up in the middle, back down
    # at both edges, and roughly symmetric because both anchors are equally good
    for profile in gaps["profiles"]:
        start, middle, end = profile["smoothed_trace"]
        assert middle > start and middle > end
        assert 0.5 < start / end < 2.0

    # so at the far edge of a gap the two disagree enormously about how much is
    # known, having seen exactly the same observations
    worst = max(gaps["profiles"], key=lambda p: p["length"])
    ratio = worst["filtered_trace"][2] / worst["smoothed_trace"][2]
    print(f"  end-of-gap trace ratio, filtered / smoothed : {ratio:.1f}x")
    assert ratio > 50.0

    # and the check that needed no observations: NEES at the middle of a gap sits
    # at its nominal 4, so the covariance grew by exactly as much as the error
    # did. one sample per run across independent runs - the two time-averages
    # printed above are both below 4 and neither is evidence of anything, which
    # is the subject of the last experiment in this file.
    n_runs = gaps["n_runs"]
    lower, upper = chi_square_interval(4 * n_runs)
    assert lower / n_runs < gaps["midpoint_nees_mean"] < upper / n_runs

    print()
    print("=== wrong Q: what RMSE sees and what NEES sees ===")
    consistency = run_consistency_experiment()

    print("  Q factor   pos RMSE   vel RMSE   mean NEES   mean NIS   log-lik")
    for row in consistency["rows"]:
        print(
            f"  {row['factor']:8.2f}   {row['position_rmse']:8.4f}   "
            f"{row['velocity_rmse']:8.4f}   {row['mean_nees']:9.3f}   "
            f"{row['mean_nis']:8.3f}   {row['log_likelihood']:9.1f}"
        )

    small, correct, large = consistency["rows"]

    # RMSE degrades by a factor of three for a factor of 100 in Q either way.
    # NEES degrades by a factor of fifty. that ratio is the whole point: the
    # estimate stays usable and the error bars stop being true.
    rmse_damage = small["position_rmse"] / correct["position_rmse"]
    nees_damage = small["mean_nees"] / correct["mean_nees"]
    print(f"  Q too small by 100x: position RMSE {rmse_damage:.1f}x, NEES {nees_damage:.0f}x")
    assert rmse_damage < 4.0
    assert nees_damage > 10.0 * rmse_damage

    # NIS catches the same fault, and much more faintly - it only ever looks
    # through H, and position is the coordinate the filter is least wrong about
    nis_damage = small["mean_nis"] / correct["mean_nis"]
    print(f"                       NIS {nis_damage:.1f}x")
    assert 1.5 < nis_damage < nees_damage / 10.0

    # too much Q is the opposite failure: an underconfident filter, NEES below
    # nominal rather than above, and error bars that are merely wasteful
    assert large["mean_nees"] < correct["mean_nees"]

    # the likelihood ranks all three correctly without ever seeing the truth,
    # which is what lets EM find Q in the first place
    assert correct["log_likelihood"] > small["log_likelihood"]
    assert correct["log_likelihood"] > large["log_likelihood"]

    print()
    print("=== is the consistency test itself valid? ===")
    validity = run_validity_experiment()
    print(f"runs / steps kept per run : {validity['n_runs']} / {validity['n_kept']}")

    for name, dim in [("nees", 4), ("nis", 2)]:
        stats = validity[name]
        lower, upper = stats["ensemble_interval"]
        naive_lower, naive_upper = stats["naive_interval"]
        print(f"  {name.upper()}  (nominal {dim})")
        print(
            f"    across runs, fixed t  : {stats['ensemble_mean']:.4f}   "
            f"interval [{lower:.4f}, {upper:.4f}]"
        )
        print(
            f"    across time, per run  : {stats['time_average_mean']:.4f}   "
            f"interval [{naive_lower:.4f}, {naive_upper:.4f}]"
        )
        print(
            f"    sd observed / naive   : {stats['observed_sd']:.4f} / "
            f"{stats['naive_sd']:.4f}  = {stats['observed_sd'] / stats['naive_sd']:.2f}x"
        )
        print(f"    naive interval covers : {stats['coverage']:.1%}  (should be 95%)")

    # the ensemble test is valid for both - independent runs, so the chi-square
    # interval means what it says
    for name in ("nees", "nis"):
        stats = validity[name]
        lower, upper = stats["ensemble_interval"]
        assert lower < stats["ensemble_mean"] < upper, name

    # time-averaging is unbiased for both. nothing is wrong with the number.
    assert abs(validity["nees"]["time_average_mean"] - 4.0) < 0.2
    assert abs(validity["nis"]["time_average_mean"] - 2.0) < 0.1

    # what is wrong is the spread. NEES values are autocorrelated, so the true
    # variance of the time average is several times the independent-sampling
    # value and the interval built on it is far too narrow.
    nees_inflation = validity["nees"]["observed_sd"] / validity["nees"]["naive_sd"]
    assert nees_inflation > 2.0
    assert validity["nees"]["coverage"] < 0.75

    effective = validity["n_kept"] / nees_inflation ** 2
    print(f"  effective independent NEES samples: {effective:.0f} of {validity['n_kept']}")

    # innovations are white, so for NIS the same move is legitimate and the
    # interval covers what it claims
    nis_inflation = validity["nis"]["observed_sd"] / validity["nis"]["naive_sd"]
    assert nis_inflation < 1.3
    assert validity["nis"]["coverage"] > 0.88

    print()
    print("day 4 checks passed")
