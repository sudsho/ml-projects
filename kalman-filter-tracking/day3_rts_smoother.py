"""
Day 3 of the Kalman filter from scratch: the RTS backward smoother.

The filter answers `p(x_t | y_1..y_t)`. The smoother answers `p(x_t | y_1..y_T)`
- the same state, conditioned on the whole sequence including everything that
came after it. Offline, that is strictly more information and the estimate is
strictly better, and the interesting part is where the improvement lands.

The HMM project's backward pass is the obvious reference point and it is the
place I got today wrong before I got it right. There, the backward recursion
computed `beta_t(k) = p(y_{t+1..T} | x_t = k)` as its own quantity, running
independently of the forward pass, and the smoothed posterior came out of
combining them at the end: `gamma_t ∝ alpha_t * beta_t`. Two separate sweeps
multiplied together pointwise.

RTS does not do that, and my first attempt at today was an effort to make it.
The reason it cannot work is worth the day:

  `beta_t` is not a distribution. It is a likelihood - a function of `x_t`, not a
  density over it. In the discrete case that distinction costs nothing, because a
  likelihood over K states is still just K numbers and you can carry it in the
  same array you would carry a distribution in. In the continuous case it is
  fatal. `p(y_{t+1..T} | x_t)` as a function of `x_t` is Gaussian-shaped but not
  normalizable: `H` observes position only, so future observations say nothing
  whatsoever about some directions in state space, and the "covariance" of that
  object is infinite along them. There is no `(mean, covariance)` pair to
  propagate.

The standard fix is the information filter - carry `S = P^-1` and `y = P^-1 m`
instead, since the offending directions have precision zero rather than variance
infinity, which is representable. That is the two-filter smoother and it is a
real algorithm. RTS sidesteps the whole problem instead: recurse directly on the
*smoothed* quantities, backward, using the filtered ones as the starting point at
each step. Every quantity in the recursion is an honest distribution and nothing
improper is ever formed.

The price is that RTS is inherently sequential in a way the HMM backward pass was
not - it needs `x_{t+1}`'s smoothed estimate to produce `x_t`'s, whereas `beta`
only ever needed `beta`. The HMM's two sweeps could in principle have run
concurrently. These cannot.

Day 2 ended by pointing out that the smoother inverts the predicted covariance at
every backward step, so conditioning that was merely tolerable going forward gets
a second look coming back. It got one, and the worry turns out to have been aimed
at the wrong matrix - the diffuse prior at the start of the run produces the
*best* conditioned predicted covariance in the whole sequence, not the worst.
Day 2's dangerous quantity was the ratio of prior uncertainty to measurement
noise, which is a relationship between `P` and `R`; conditioning is a property of
`P` by itself, and I ran the two together writing that paragraph. What actually
moves the condition number is `H` being partial. See
`run_conditioning_experiment`.

Today also produces the lag-one smoothed covariance, which nothing here consumes.
Day 4's EM needs `E[x_t x_{t-1}' | y_1..y_T]` to re-estimate the transition and
`Q`, and that cross-moment is not recoverable from the smoothed marginals alone -
the marginals give the diagonal blocks of the joint posterior and this is the
first off-diagonal. Building it now, next to the recursion it falls out of, is
easier than retrofitting it later. Same reasoning as day 1's log-likelihood.
"""

import numpy as np

from day1_kalman_filter import (
    constant_velocity_model,
    kalman_filter,
    rmse,
    simulate,
)
from day2_numerical_stability import symmetrize


def rts_smoother(filter_result, transition):
    """Run the backward pass over a completed forward pass.

    The recursion, initialized at `T-1` where smoothed and filtered coincide
    because there is no future left to condition on:

        J_t   = P_t F' (P_{t+1}^-)^-1
        m_t^s = m_t + J_t (m_{t+1}^s - m_{t+1}^-)
        P_t^s = P_t + J_t (P_{t+1}^s - P_{t+1}^-) J_t'

    The gain `J_t` is worth reading rather than transcribing. `P_t F'` is the
    covariance between `x_t` and the *prediction* of `x_{t+1}`, and dividing by
    that prediction's own covariance turns it into a regression coefficient: how
    much of a surprise about `x_{t+1}` should be attributed backward to `x_t`.
    It is the Kalman gain with time reversed. The forward gain asks how much of
    an observation's surprise to believe; this asks how much of the future's
    correction to inherit.

    The two update terms are both *differences against the prediction*, which is
    what makes the recursion self-terminating: where the future turned out
    exactly as predicted, `m_{t+1}^s = m_{t+1}^-`, the correction is zero and the
    smoothed estimate is the filtered one. Nothing is being added on faith.

    `P_t^s - P_t` is negative semidefinite because `P_{t+1}^s - P_{t+1}^-` is:
    conditioning on more data cannot increase the covariance, and the congruence
    by `J_t` preserves that sign. So the covariance shrinks everywhere, which is
    asserted below rather than assumed.

    `J` is solved rather than inverted, transposing the same way day 1's Kalman
    gain did: `J_t' = (P_{t+1}^-)^-1 F P_t` using the symmetry of the predicted
    covariance. Unlike day 1's, this solve is against an `n x n` state-space
    system rather than the smaller observation-space one, so it is the expensive
    step here and the one day 2's conditioning worry attaches to.
    """
    filtered_means = filter_result["filtered_means"]
    filtered_covs = filter_result["filtered_covs"]
    predicted_means = filter_result["predicted_means"]
    predicted_covs = filter_result["predicted_covs"]

    n_steps, state_dim = filtered_means.shape

    smoothed_means = np.zeros_like(filtered_means)
    smoothed_covs = np.zeros_like(filtered_covs)
    gains = np.zeros((n_steps - 1, state_dim, state_dim))

    # the last step has no future to condition on, so smoothed is filtered. this
    # is the base case and also the cheapest correctness check in the file.
    smoothed_means[-1] = filtered_means[-1]
    smoothed_covs[-1] = filtered_covs[-1]

    for t in range(n_steps - 2, -1, -1):
        gain = np.linalg.solve(
            symmetrize(predicted_covs[t + 1]), transition @ filtered_covs[t]
        ).T
        gains[t] = gain

        smoothed_means[t] = filtered_means[t] + gain @ (
            smoothed_means[t + 1] - predicted_means[t + 1]
        )
        smoothed_covs[t] = symmetrize(
            filtered_covs[t]
            + gain @ (smoothed_covs[t + 1] - predicted_covs[t + 1]) @ gain.T
        )

    return {
        "smoothed_means": smoothed_means,
        "smoothed_covs": smoothed_covs,
        "gains": gains,
        "lag_one_covs": lag_one_smoothed_covariance(smoothed_covs, gains),
    }


def lag_one_smoothed_covariance(smoothed_covs, gains):
    """`Cov(x_t, x_{t-1} | y_1..y_T)` for `t = 1..T-1`, one block per step.

    Falls out of the same backward recursion as `P_t^s J_{t-1}'`, which is short
    enough to look like a coincidence and is not. `J_{t-1}` is the regression
    coefficient of `x_{t-1}` on `x_t` given everything up to `t-1`; multiplying
    the smoothed covariance of `x_t` through it transports that covariance back
    across one step of the same regression.

    Nothing today consumes this. Day 4's EM does, and cannot be written without
    it: the M-step for `F` and `Q` needs `E[x_t x_{t-1}']`, and the smoothed
    marginals only ever give `E[x_t x_t']`. Marginals are the diagonal blocks of
    the joint posterior over the whole trajectory and this is the first
    off-diagonal - it is genuinely absent from what the smoother otherwise
    returns, not merely inconvenient to extract.

    Index `t - 1` of the returned array is the block for the pair `(t, t-1)`.
    """
    n_steps = smoothed_covs.shape[0]
    state_dim = smoothed_covs.shape[1]

    lag_one = np.zeros((n_steps - 1, state_dim, state_dim))
    for t in range(1, n_steps):
        lag_one[t - 1] = smoothed_covs[t] @ gains[t - 1].T

    return lag_one


def joint_posterior_brute(observations, transition, process_noise, observation,
                          observation_noise, initial_mean, initial_cov):
    """The smoothed posterior by building the joint Gaussian and conditioning it.

    A linear-Gaussian state-space model is one big joint Gaussian over
    `(x_1..x_T, y_1..y_T)`. So the smoothed posterior is just the conditional of
    one block given the other, which is a textbook formula and a single dense
    solve. `O(T^3)` and hopeless past a few dozen steps, which is exactly why the
    RTS recursion exists.

    It is here because it shares *nothing* with the smoother. No backward pass,
    no gain, no filtered covariance, not even the forward recursion - it is built
    straight from the model matrices. When the two agree, that is evidence about
    the recursion rather than a restatement of it. Same role the heap merge
    played against the two-pointer sweep in the LeetCode notes: an independent
    route to the same number, kept precisely because it is too slow to be the
    answer.

    It also delivers the lag-one covariance for free as an off-diagonal block of
    the same posterior, which is the only check available on that quantity -
    nothing in the project consumes it until tomorrow, so it would otherwise ship
    untested.

    Note the time convention inherited from day 1: `x_0` is the prior and is
    never observed, so array index `t` holds `x_{t+1}`. The first state has mean
    `F m_0` and covariance `F P_0 F' + Q`.
    """
    n_steps = observations.shape[0]
    state_dim = transition.shape[0]
    obs_dim = observation.shape[0]

    # marginal mean and the diagonal covariance blocks of the state trajectory
    means = np.zeros((n_steps, state_dim))
    blocks = np.zeros((n_steps, state_dim, state_dim))

    means[0] = transition @ initial_mean
    blocks[0] = transition @ initial_cov @ transition.T + process_noise
    for t in range(1, n_steps):
        means[t] = transition @ means[t - 1]
        blocks[t] = transition @ blocks[t - 1] @ transition.T + process_noise

    # Cov(x_t, x_s) = F^(t-s) Cov(x_s, x_s) for t > s, since the extra noise
    # injected between s and t is independent of x_s
    size = n_steps * state_dim
    joint_cov = np.zeros((size, size))
    for s in range(n_steps):
        block = blocks[s]
        for t in range(s, n_steps):
            rows = slice(t * state_dim, (t + 1) * state_dim)
            cols = slice(s * state_dim, (s + 1) * state_dim)
            joint_cov[rows, cols] = block
            joint_cov[cols, rows] = block.T
            block = transition @ block

    joint_mean = means.reshape(-1)

    obs_map = np.zeros((n_steps * obs_dim, size))
    obs_cov = np.zeros((n_steps * obs_dim, n_steps * obs_dim))
    for t in range(n_steps):
        rows = slice(t * obs_dim, (t + 1) * obs_dim)
        cols = slice(t * state_dim, (t + 1) * state_dim)
        obs_map[rows, cols] = observation
        obs_cov[rows, rows] = observation_noise

    cross = joint_cov @ obs_map.T
    marginal_obs_cov = obs_map @ joint_cov @ obs_map.T + obs_cov
    residual = observations.reshape(-1) - obs_map @ joint_mean

    posterior_mean = joint_mean + cross @ np.linalg.solve(marginal_obs_cov, residual)
    posterior_cov = joint_cov - cross @ np.linalg.solve(marginal_obs_cov, cross.T)

    smoothed_means = posterior_mean.reshape(n_steps, state_dim)
    smoothed_covs = np.zeros((n_steps, state_dim, state_dim))
    lag_one = np.zeros((max(n_steps - 1, 0), state_dim, state_dim))

    for t in range(n_steps):
        block = slice(t * state_dim, (t + 1) * state_dim)
        smoothed_covs[t] = symmetrize(posterior_cov[block, block])
        if t > 0:
            previous = slice((t - 1) * state_dim, t * state_dim)
            lag_one[t - 1] = posterior_cov[block, previous]

    return {
        "smoothed_means": smoothed_means,
        "smoothed_covs": smoothed_covs,
        "lag_one_covs": lag_one,
    }


def _tracking_run(n_steps, seed, accel_noise=0.6, obs_noise=1.2, dt=0.1):
    """Simulate a track and run the forward filter over it."""
    rng = np.random.default_rng(seed)

    transition, process_noise, observation, observation_noise = constant_velocity_model(
        dt=dt, accel_noise=accel_noise, obs_noise=obs_noise
    )

    states, observations = simulate(
        transition, process_noise, observation, observation_noise,
        np.array([0.0, 0.0, 2.0, 1.0]), n_steps=n_steps, rng=rng,
    )

    initial_mean = np.zeros(4)
    initial_cov = np.diag([5.0, 5.0, 25.0, 25.0])

    result = kalman_filter(
        observations, transition, process_noise, observation,
        observation_noise, initial_mean, initial_cov,
    )

    return {
        "states": states,
        "observations": observations,
        "filter": result,
        "model": (transition, process_noise, observation, observation_noise),
        "prior": (initial_mean, initial_cov),
    }


def run_smoothing_experiment(n_steps=400, seed=0):
    """Where the smoother's improvement actually lands.

    The headline is that smoothed beats filtered, which is guaranteed and
    therefore uninteresting. The distribution of the gain is the point.

    Position is observed directly at every step, so the filter is already close
    and the future has little to add. Velocity is never observed at all - the
    filter infers it only from the position/velocity correlation that `Q` builds
    up, and at any given step it has only the past to build it from. The smoother
    gets to look at where the target subsequently went, which is a far more
    direct statement about how fast it was moving. So the improvement should be
    lopsided toward velocity, and by a lot.

    The other lopsidedness is in time. At the last step smoothed *equals*
    filtered by construction, and near the start the filter is still recovering
    from a diffuse prior while the smoother is not. So the benefit is
    concentrated at the beginning of the sequence and vanishes at the end.
    """
    run = _tracking_run(n_steps, seed)
    transition = run["model"][0]
    smoothed = rts_smoother(run["filter"], transition)

    states = run["states"]
    filtered_means = run["filter"]["filtered_means"]
    smoothed_means = smoothed["smoothed_means"]

    report = {
        "filtered_position_rmse": rmse(filtered_means[:, :2], states[:, :2]),
        "smoothed_position_rmse": rmse(smoothed_means[:, :2], states[:, :2]),
        "filtered_velocity_rmse": rmse(filtered_means[:, 2:], states[:, 2:]),
        "smoothed_velocity_rmse": rmse(smoothed_means[:, 2:], states[:, 2:]),
        "first_tenth_filtered": rmse(filtered_means[: n_steps // 10, :2],
                                     states[: n_steps // 10, :2]),
        "first_tenth_smoothed": rmse(smoothed_means[: n_steps // 10, :2],
                                     states[: n_steps // 10, :2]),
    }
    report["result"] = smoothed
    report["run"] = run
    return report


def run_conditioning_experiment(n_steps=400, seed=0):
    """Day 2's worry, checked - and the worry was aimed at the wrong matrix.

    The backward gain solves against `P_{t+1}^-`, the *predicted* covariance,
    once per step. Day 2 closed by suggesting that conditioning which was merely
    tolerable going forward would get a second look coming back, on the grounds
    that the diffuse prior at the start of the run makes for a nasty matrix.

    I went looking for that and it is not there. The prediction I wrote down was
    that step 0 would be the worst-conditioned matrix in the run and the backward
    pass would be safe because it never touches step 0. Both halves are wrong,
    and the correction is the useful part of today's second half.

    Step 0 is the *best*-conditioned predicted covariance in the entire run. A
    diffuse prior is diffuse in every direction at once, and `cond` measures the
    spread of the eigenvalues rather than their size - `diag(5, 5, 25, 25)` is a
    huge covariance and a well conditioned one. Day 2's dangerous quantity was
    the ratio of prior uncertainty to *measurement* noise, which governs how much
    cancellation the short-form update suffers. That is a relationship between
    `P` and `R`. This is a property of `P` alone. Two different quantities and I
    ran them together writing that closing paragraph.

    What actually drives the spread is that `H` is partial, and the two blocks of
    the state contract on completely different timescales. Position is observed
    directly: its predicted variance falls 5.25 -> 1.48 in a single step and then
    sits flat near 1.1, because it has already hit the floor that `Q` and `R`
    jointly set. Velocity is never observed and can only contract through the
    correlation `Q` builds up, so it barely moves on the first step - 25.0 ->
    24.1 - and then declines slowly for dozens of steps.

    The condition number is the gap between those two schedules. It is small at
    step 0 because nothing has contracted yet, rises as position collapses and
    velocity does not, peaks a few steps in with the fast direction finished and
    the slow one barely started, and decays back as velocity catches up. Steady
    state is milder than the peak by a factor of six.

    So the peak is real, early, transient, and about forty - which is nothing.
    The backward pass does reach it last, which was the one part of the original
    guess that survives, but it no longer matters that it does.
    """
    run = _tracking_run(n_steps, seed)
    predicted_covs = run["filter"]["predicted_covs"]

    conditions = np.array([np.linalg.cond(cov) for cov in predicted_covs])

    # the mechanism, measured rather than asserted: the observed and unobserved
    # blocks of the predicted covariance, tracked separately. the condition
    # number is essentially the ratio between them.
    observed_variance = np.array([np.trace(cov[:2, :2]) / 2.0 for cov in predicted_covs])
    unobserved_variance = np.array([np.trace(cov[2:, 2:]) / 2.0 for cov in predicted_covs])

    return {
        "conditions": conditions,
        "worst_overall": float(conditions.max()),
        "worst_index": int(conditions.argmax()),
        "first_step": float(conditions[0]),
        "steady_state": float(np.median(conditions[n_steps // 2:])),
        "observed_variance": observed_variance,
        "unobserved_variance": unobserved_variance,
    }


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    report = run_smoothing_experiment()
    smoothed = report["result"]
    run = report["run"]

    print("=== smoothing vs filtering (400 steps) ===")
    print(f"position RMSE  filtered  : {report['filtered_position_rmse']:.4f}")
    print(f"position RMSE  smoothed  : {report['smoothed_position_rmse']:.4f}")
    print(f"velocity RMSE  filtered  : {report['filtered_velocity_rmse']:.4f}")
    print(f"velocity RMSE  smoothed  : {report['smoothed_velocity_rmse']:.4f}")
    print(f"first 10% pos  filtered  : {report['first_tenth_filtered']:.4f}")
    print(f"first 10% pos  smoothed  : {report['first_tenth_smoothed']:.4f}")

    # the guarantee. conditioning on more data cannot make the estimate worse in
    # expectation, and over 400 steps the sample version had better agree.
    assert report["smoothed_position_rmse"] < report["filtered_position_rmse"]
    assert report["smoothed_velocity_rmse"] < report["filtered_velocity_rmse"]

    # and the lopsidedness. velocity is never observed, so the future is doing
    # most of the work there and comparatively little for position.
    position_gain = 1.0 - report["smoothed_position_rmse"] / report["filtered_position_rmse"]
    velocity_gain = 1.0 - report["smoothed_velocity_rmse"] / report["filtered_velocity_rmse"]
    print(f"relative gain  position  : {position_gain:.1%}")
    print(f"relative gain  velocity  : {velocity_gain:.1%}")
    assert velocity_gain > position_gain

    # base case: the last step has no future, so smoothing is a no-op there
    assert np.allclose(smoothed["smoothed_means"][-1], run["filter"]["filtered_means"][-1])
    assert np.allclose(smoothed["smoothed_covs"][-1], run["filter"]["filtered_covs"][-1])

    # every smoothed covariance is a valid covariance, and no larger than the
    # filtered one it refines - the difference has to be negative semidefinite,
    # which is the matrix statement of "more data cannot hurt"
    for t in range(len(smoothed["smoothed_covs"])):
        cov = smoothed["smoothed_covs"][t]
        assert np.allclose(cov, cov.T, atol=1e-9)
        assert np.all(np.linalg.eigvalsh(cov) > 0), t
        difference = cov - run["filter"]["filtered_covs"][t]
        assert np.all(np.linalg.eigvalsh(symmetrize(difference)) < 1e-9), t

    print()
    print("=== against the joint posterior, built and conditioned directly ===")

    # short sequence, because the dense route is O(T^3) in a 4-dimensional state
    for short_steps, seed in [(3, 4), (12, 5), (25, 6)]:
        short = _tracking_run(short_steps, seed)
        transition, process_noise, observation, observation_noise = short["model"]
        initial_mean, initial_cov = short["prior"]

        recursive = rts_smoother(short["filter"], transition)
        direct = joint_posterior_brute(
            short["observations"], transition, process_noise, observation,
            observation_noise, initial_mean, initial_cov,
        )

        mean_gap = np.max(np.abs(recursive["smoothed_means"] - direct["smoothed_means"]))
        cov_gap = np.max(np.abs(recursive["smoothed_covs"] - direct["smoothed_covs"]))
        lag_gap = np.max(np.abs(recursive["lag_one_covs"] - direct["lag_one_covs"]))

        print(f"T={short_steps:3d}  mean {mean_gap:.2e}  cov {cov_gap:.2e}  lag-one {lag_gap:.2e}")

        assert mean_gap < 1e-8, short_steps
        assert cov_gap < 1e-8, short_steps
        # the lag-one blocks are the only check this quantity gets before day 4
        # picks it up, and it is the reason the dense route is in this file
        assert lag_gap < 1e-8, short_steps

    # a single-step sequence has no backward step at all, so the smoother must
    # return the filter untouched and produce an empty lag-one array
    single = _tracking_run(1, seed=7)
    single_smoothed = rts_smoother(single["filter"], single["model"][0])
    assert np.allclose(single_smoothed["smoothed_means"], single["filter"]["filtered_means"])
    assert single_smoothed["lag_one_covs"].shape[0] == 0

    print()
    print("=== conditioning of the matrices the backward gain solves against ===")
    conditioning = run_conditioning_experiment()
    print(f"step 0 (the diffuse prior)      : {conditioning['first_step']:.2f}")
    print(f"worst overall                   : {conditioning['worst_overall']:.2f}"
          f"  at step {conditioning['worst_index']}")
    print(f"steady state (median)           : {conditioning['steady_state']:.2f}")

    observed = conditioning["observed_variance"]
    unobserved = conditioning["unobserved_variance"]
    print("observed (position) variance    :",
          " ".join(f"{v:7.3f}" for v in observed[:6]))
    print("unobserved (velocity) variance  :",
          " ".join(f"{v:7.3f}" for v in unobserved[:6]))

    # the guess this experiment was written to confirm, kept as a failing
    # assertion would have been - the diffuse prior is the best conditioned
    # matrix in the run, not the worst. a diffuse prior is diffuse in every
    # direction, and cond measures spread rather than size.
    assert conditioning["first_step"] < conditioning["worst_overall"]

    # what actually drives it, in two steps. first: one observation collapses
    # position and leaves velocity essentially untouched, because H sees one and
    # not the other.
    assert conditioning["worst_index"] < 20
    assert observed[1] / observed[0] < 0.35
    assert unobserved[1] / unobserved[0] > 0.9

    # second: past that first step position is already at its floor and stays
    # flat, while velocity keeps contracting for dozens of steps. the peak is
    # where those two schedules are furthest apart, so it is early and transient
    # rather than at either boundary.
    assert observed[5] / observed[1] > 0.6
    assert unobserved[5] / unobserved[1] < 0.5
    assert conditioning["steady_state"] < conditioning["worst_overall"]

    # and the magnitude is the reason day 2's worry can be closed rather than
    # defended against. Q is full rank and floors every eigenvalue each predict
    # step, so nothing here is within orders of magnitude of dangerous.
    assert conditioning["worst_overall"] < 1e3

    print()
    print("day 3 checks passed")
