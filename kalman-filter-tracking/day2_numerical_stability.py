"""
Day 2 of the Kalman filter from scratch: why the textbook covariance update rots.

Day 1's `update` ended on a comment calling `P - K S K.T` algebraically correct
and numerically fragile. Today is that claim, made to actually happen, and then
fixed - and the shape of the failure is not the one I went in expecting, which
is most of what the day was worth.

Two unrelated failure modes are both called "filter divergence", they look
nothing alike, and only one of them is a numerics problem.

  1. Covariance corruption. `P` stops being positive definite. The short-form
     update is a difference of two positive semidefinite matrices, and when they
     nearly cancel the round-off in the small remainder can carry an eigenvalue
     past zero. After that the filter holds a "covariance" that is not one: the
     Cholesky in the log-likelihood raises, the gain is nonsense, the estimate
     follows. The Joseph form fixes this structurally.

  2. Overconfidence. `P` stays perfectly valid and shrinks toward zero, so the
     gain shrinks too, so the filter stops listening to observations. The
     estimate drifts away from the truth while the reported uncertainty says it
     is doing fine. This is a modelling error - `Q` too small for the motion
     that is actually happening - and no amount of numerical care touches it.

The expectation I had about the first one was wrong. I assumed slow rot: error
accumulating over thousands of steps until definiteness gives out. It does not
accumulate, and the reason is `Q`. Every predict step adds a positive definite
matrix to `P`, which is a regularizer whether or not it was meant as one - it
resets the smallest eigenvalue off the floor and discards the previous step's
round-off along with it. A filter with healthy process noise does not degrade
over time; the long run below confirms it, 4000 steps in float32 with the naive
form matching Joseph to the last digit.

What actually kills it is a single bad step. The cancellation in `P - K S K.T`
is severe exactly when the prior uncertainty dwarfs the measurement noise,
because then the gain is nearly `H^+` and the subtraction removes nearly all of
`P`. That ratio is at its worst at the very first update after a diffuse prior,
which is the most ordinary way in the world to start a filter. So the danger is
concentrated at step 0 rather than spread over the run, and "it survived a long
sequence" is not evidence of anything.

Day 3's RTS smoother inverts the predicted covariance at every backward step, so
conditioning that is merely tolerable on the way forward gets a second look on
the way back.
"""

import numpy as np

from day1_kalman_filter import (
    constant_velocity_model,
    predict,
    rmse,
    simulate,
    update,
)


def symmetrize(matrix):
    """Project a matrix onto the symmetric ones: `(A + A.T) / 2`.

    Cheap and worth doing after every covariance write. `P` is symmetric in
    exact arithmetic at every point in the recursion, so any asymmetry that
    appears is pure round-off - which means discarding the antisymmetric part
    throws away error and nothing else.

    This is a projection in the Frobenius sense: `(A + A.T)/2` is the closest
    symmetric matrix to `A`. It is not a fix for indefiniteness. An asymmetric
    matrix and an indefinite one are different diseases and this only treats the
    first, which is exactly why the Joseph form below is a separate change.
    """
    return 0.5 * (matrix + matrix.T)


def update_joseph(mean, cov, obs, observation, observation_noise):
    """The Joseph-form update: `(I - KH) P (I - KH).T + K R K.T`.

    Algebraically identical to day 1's `P - K S K.T` *when K is the optimal
    gain*, and that qualifier is the entire point.

    Expanding the Joseph form with `K = P H.T S^-1` and `S = H P H.T + R`
    collapses it to the short form, so on paper there is nothing to choose
    between them. Numerically they are not the same computation:

      - The short form is a subtraction of two positive semidefinite matrices.
        The result is the small difference of two larger things, so the relative
        error in it is amplified by their ratio, and once that error exceeds the
        smallest eigenvalue the result leaves the positive-definite cone.

      - The Joseph form is a sum of two positive semidefinite terms - a
        congruence `M P M.T` and `K R K.T`. Congruence preserves definiteness
        exactly, and a sum of PSD matrices is PSD, so the result is structurally
        non-negative rather than accidentally so. Round-off perturbs the value,
        not the sign.

    That structural property is also why Joseph is valid for a *suboptimal* `K`,
    where the short form is simply wrong rather than merely fragile. Any filter
    that clamps or scales its gain - a fading-memory filter, a robust filter
    that discounts outliers - has to use this form. The optimal-gain case is the
    special case, not the general one.

    Cost is a couple of extra `n x n` products per step. On a 4-state model that
    is nothing, and it is not much on a big one either, because the expensive
    part of a Kalman step scales with the observation dimension through `S`.
    """
    innovation = obs - observation @ mean
    innovation_cov = symmetrize(observation @ cov @ observation.T + observation_noise)

    gain = np.linalg.solve(innovation_cov, observation @ cov).T

    new_mean = mean + gain @ innovation

    factor = np.eye(cov.shape[0]) - gain @ observation
    new_cov = factor @ cov @ factor.T + gain @ observation_noise @ gain.T

    return new_mean, symmetrize(new_cov), innovation, innovation_cov


def kalman_filter_diagnosed(observations, transition, process_noise, observation,
                            observation_noise, initial_mean, initial_cov,
                            joseph=True, dtype=np.float64):
    """Forward pass instrumented with the health of `P`, and no likelihood.

    Day 1's filter accumulates the log-likelihood through a Cholesky, which
    raises the moment `P` goes indefinite. That is the right behaviour there and
    useless here - the whole experiment is to watch the corruption happen, so it
    has to be allowed to happen. This one records instead of raising:

      min_eigenvalue  the smallest eigenvalue of the filtered `P`. Positive on a
                      healthy run; crossing zero is the corruption.
      asymmetry       `max|P - P.T|` relative to `max|P|`. Round-off in the
                      short-form update is not symmetric, so this moves before
                      the eigenvalue does and is the cheaper warning.
      condition       `cond(P)`. The mechanism rather than the symptom - the
                      cancellation is dangerous exactly when the eigenvalue
                      spread is wide enough that error at the top swamps the
                      bottom.

    `dtype` exists so this can be run in float32. Single precision is not a
    rigged demo, it is what an embedded filter actually runs in, and it puts the
    same arithmetic within reach of a test instead of several orders of
    magnitude out of it.
    """
    n_steps = observations.shape[0]
    state_dim = transition.shape[0]

    transition = transition.astype(dtype)
    process_noise = process_noise.astype(dtype)
    observation = observation.astype(dtype)
    observation_noise = observation_noise.astype(dtype)

    step = update_joseph if joseph else update

    filtered_means = np.zeros((n_steps, state_dim))
    min_eigenvalues = np.zeros(n_steps)
    asymmetries = np.zeros(n_steps)
    conditions = np.zeros(n_steps)

    mean = np.asarray(initial_mean, dtype=dtype)
    cov = np.asarray(initial_cov, dtype=dtype)

    for t in range(n_steps):
        mean, cov = predict(mean, cov, transition, process_noise)
        mean, cov, _, _ = step(mean, cov, observations[t].astype(dtype),
                               observation, observation_noise)

        filtered_means[t] = mean
        # eigenvalues in double regardless of the filter's own precision - the
        # instrument should not share the failure mode it is measuring
        eigenvalues = np.linalg.eigvalsh(symmetrize(cov.astype(np.float64)))
        min_eigenvalues[t] = eigenvalues[0]
        asymmetries[t] = np.max(np.abs(cov - cov.T)) / max(np.max(np.abs(cov)), 1e-30)
        conditions[t] = (
            abs(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] != 0 else np.inf
        )

    return {
        "filtered_means": filtered_means,
        "min_eigenvalues": min_eigenvalues,
        "asymmetries": asymmetries,
        "conditions": conditions,
    }


def stiff_observation_noise(precise, sloppy):
    """`R` for a sensor that is far more certain about x than about y.

    Not a contrived matrix. A radar measuring range precisely and bearing poorly
    gives this shape once projected into cartesian coordinates, as does any
    fused pair of sensors with different accuracies on different axes.

    A tiny variance on one axis drives the gain there to nearly 1, so `K S K.T`
    nearly cancels `P` in that direction and the surviving entry is a small
    difference of large numbers - the worst case for the short form. The wider
    the ratio between the two variances, the wider the eigenvalue spread of `P`
    and the less room there is between round-off and zero.
    """
    return np.diag([precise ** 2, sloppy ** 2])


def _tracking_setup(accel_noise, precise, sloppy, n_steps, seed):
    """Shared data for both experiments below, so only the update form differs."""
    rng = np.random.default_rng(seed)

    transition, process_noise, observation, _ = constant_velocity_model(
        dt=0.05, accel_noise=accel_noise, obs_noise=1.0
    )
    observation_noise = stiff_observation_noise(precise, sloppy)

    initial_state = np.array([0.0, 0.0, 1.0, -0.5])
    _, observations = simulate(
        transition, process_noise, observation, observation_noise,
        initial_state, n_steps, rng,
    )

    return observations, transition, process_noise, observation, observation_noise


def run_diffuse_prior_experiment(n_steps=4000, seed=1):
    """A diffuse prior against a precise sensor - the step that actually breaks it.

    `P0` says the position is unknown to within a hundred units of variance; `R`
    says the x measurement is good to `1e-4`. The first update therefore has to
    remove almost all of `P` in the x direction, and the short form removes it by
    subtracting two nearly equal matrices in float32.

    This is the ordinary way to start a filter. Nobody initializes a tracker
    with a tight prior on a target they have not seen yet, so the most dangerous
    step in the whole run is the one every filter takes first.

    Both branches get identical observations, an identical initial belief and
    identical precision. The only variable is which covariance update runs.
    """
    setup = _tracking_setup(
        accel_noise=1e-3, precise=1e-4, sloppy=30.0, n_steps=n_steps, seed=seed
    )
    initial_mean = np.zeros(4)
    initial_cov = np.diag([100.0, 100.0, 100.0, 100.0])

    naive = kalman_filter_diagnosed(
        *setup, initial_mean, initial_cov, joseph=False, dtype=np.float32
    )
    joseph = kalman_filter_diagnosed(
        *setup, initial_mean, initial_cov, joseph=True, dtype=np.float32
    )

    return naive, joseph


def run_long_run_experiment(n_steps=4000, seed=1):
    """The same filter started from a sane prior, run long, in float32.

    This is the control, and it is the one that corrected the expectation. If
    round-off accumulated across steps, four thousand of them at single
    precision with a condition number in the 1e8 range would show it. Nothing
    shows. The two update forms end up a few float32 epsilon apart in relative
    terms, which is the difference between two ways of writing the same sum and
    not the signature of anything building up.

    `Q` is why. Every predict step adds a positive definite matrix to `P`, so
    the smallest eigenvalue is pushed back off the floor before each update and
    the previous step's error goes with it. Process noise is a regularizer, and
    it was not put there to be one.

    The consequence for testing is the part worth keeping: a long clean run is
    not evidence that the short form is safe. It is evidence that `Q` is doing
    its job, which it will keep doing right up until a step arrives where the
    prior/measurement ratio is bad enough to blow through it in one go.
    """
    setup = _tracking_setup(
        accel_noise=1e-2, precise=1e-3, sloppy=30.0, n_steps=n_steps, seed=seed
    )
    initial_mean = np.zeros(4)
    initial_cov = np.diag([10.0, 10.0, 10.0, 10.0])

    naive = kalman_filter_diagnosed(
        *setup, initial_mean, initial_cov, joseph=False, dtype=np.float32
    )
    joseph = kalman_filter_diagnosed(
        *setup, initial_mean, initial_cov, joseph=True, dtype=np.float32
    )

    return naive, joseph


def run_overconfidence_experiment(n_steps=600, seed=2):
    """The other divergence: a filter that is wrong and certain, with clean arithmetic.

    The truth manoeuvres - a steady turn, which a constant-velocity model cannot
    represent - while the filter is handed a `Q` saying the motion is almost
    exactly constant velocity. Nothing here is numerically hard. It runs in
    float64 with the Joseph form and `P` stays positive definite throughout.

    What happens instead is that `P` converges to something tiny, so the gain
    converges to something tiny, so observations stop moving the estimate. The
    filter has concluded it knows the trajectory and stops taking evidence. The
    error grows and the reported uncertainty does not.

    The tell is entirely in the innovations. They are supposed to be white and
    zero-mean; here they become large and strongly autocorrelated, because the
    filter makes the same signed mistake for many consecutive steps. Nothing
    else in the filter's own output says anything is wrong, which is the point.
    """
    rng = np.random.default_rng(seed)

    dt = 0.1
    transition, _, observation, observation_noise = constant_velocity_model(
        dt=dt, accel_noise=1.0, obs_noise=0.5
    )
    # what the filter believes: essentially no process noise
    filter_process_noise = 1e-9 * np.eye(4)

    state = np.array([0.0, 0.0, 2.0, 0.0])
    states = np.zeros((n_steps, 4))
    observations = np.zeros((n_steps, 2))
    turn_rate = 0.05

    for t in range(n_steps):
        speed = np.hypot(state[2], state[3])
        heading = np.arctan2(state[3], state[2]) + turn_rate * dt
        state = np.array([
            state[0] + state[2] * dt,
            state[1] + state[3] * dt,
            speed * np.cos(heading),
            speed * np.sin(heading),
        ])
        states[t] = state
        observations[t] = observation @ state + 0.5 * rng.standard_normal(2)

    mean = np.zeros(4)
    cov = np.diag([5.0, 5.0, 5.0, 5.0])

    means = np.zeros((n_steps, 4))
    traces = np.zeros(n_steps)
    innovations = np.zeros((n_steps, 2))

    for t in range(n_steps):
        mean, cov = predict(mean, cov, transition, filter_process_noise)
        mean, cov, innovation, _ = update_joseph(
            mean, cov, observations[t], observation, observation_noise
        )
        means[t] = mean
        traces[t] = np.trace(cov)
        innovations[t] = innovation

    return states, means, traces, innovations


def innovation_autocorrelation(innovations, lag=1):
    """Lag-`lag` correlation of the innovation sequence, averaged over components.

    Under a correctly specified model the innovations are a martingale
    difference sequence, so this is zero up to sampling noise. A filter that is
    systematically wrong repeats its mistake, and the repetition shows up here
    long before it shows up in anything the filter reports about itself.
    """
    centered = innovations - innovations.mean(axis=0)
    numerator = np.sum(centered[:-lag] * centered[lag:], axis=0)
    denominator = np.sum(centered * centered, axis=0)

    return float(np.mean(numerator / denominator))


if __name__ == "__main__":
    print("=== 1. diffuse prior, precise sensor: short form vs joseph, float32 ===")
    naive, joseph = run_diffuse_prior_experiment()

    first_bad = int(np.argmax(naive["min_eigenvalues"] <= 0))

    print(f"short form  min eigenvalue over run : {naive['min_eigenvalues'].min():.3e}")
    print(f"short form  first indefinite step   : {first_bad}")
    print(f"short form  worst asymmetry         : {naive['asymmetries'].max():.3e}")
    print(f"joseph      min eigenvalue over run : {joseph['min_eigenvalues'].min():.3e}")
    print(f"joseph      worst asymmetry         : {joseph['asymmetries'].max():.3e}")
    print(f"joseph      worst condition number  : {joseph['conditions'].max():.3e}")

    # the short form loses definiteness, and it loses it immediately - not after
    # a long grind. the second assert is the one that surprised me
    assert np.any(naive["min_eigenvalues"] <= 0), "expected the short form to break"
    assert first_bad == 0, "expected the very first update to be the fatal one"

    # joseph holds through the same step on the same data
    assert np.all(joseph["min_eigenvalues"] > 0), "joseph form lost positive definiteness"

    # symmetry is enforced explicitly in the joseph path, so this checks the
    # enforcement rather than a happy accident. the short form is left
    # unenforced on purpose - its asymmetry is the cheap early warning
    assert joseph["asymmetries"].max() == 0.0
    assert naive["asymmetries"].max() > joseph["asymmetries"].max()

    print()
    print("=== 2. control: sane prior, 4000 steps, float32 ===")
    long_naive, long_joseph = run_long_run_experiment()

    print(f"short form  min eigenvalue over run : {long_naive['min_eigenvalues'].min():.3e}")
    print(f"short form  worst condition number  : {long_naive['conditions'].max():.3e}")
    print(f"joseph      min eigenvalue over run : {long_joseph['min_eigenvalues'].min():.3e}")

    estimate_gap = np.max(np.abs(long_naive["filtered_means"] - long_joseph["filtered_means"]))
    estimate_scale = np.max(np.abs(long_joseph["filtered_means"]))
    relative_gap = estimate_gap / estimate_scale

    print(f"|naive - joseph| on the estimate    : {estimate_gap:.3e}")
    print(
        f"  relative to state scale           : {relative_gap:.3e} "
        f"({relative_gap / np.finfo(np.float32).eps:.1f} x float32 eps)"
    )

    # no accumulation. Q resets the floor every predict step and takes the
    # previous step's round-off with it, so a long clean run says nothing about
    # whether the short form is safe
    assert np.all(long_naive["min_eigenvalues"] > 0)

    # a few epsilon apart after 4000 steps. this is two spellings of the same
    # sum disagreeing at the precision they are written in, not error growing
    assert relative_gap < 100 * np.finfo(np.float32).eps

    print()
    print("=== 3. overconfidence: valid covariance, useless filter ===")
    states, means, traces, innovations = run_overconfidence_experiment()

    half = len(states) // 2
    early_error = rmse(means[:half, :2], states[:half, :2])
    late_error = rmse(means[half:, :2], states[half:, :2])
    autocorr = innovation_autocorrelation(innovations)

    print(f"trace(P) at start / end   : {traces[0]:.4e} -> {traces[-1]:.4e}")
    print(f"position RMSE first half  : {early_error:.4f}")
    print(f"position RMSE second half : {late_error:.4f}")
    print(f"innovation lag-1 autocorr : {autocorr:.4f}")

    # the covariance never breaks. that is the whole distinction from part 1 -
    # this failure is not visible in the arithmetic anywhere
    assert np.all(traces > 0)
    assert traces[-1] < traces[0]

    # the filter stops listening, so the error grows while its own uncertainty
    # shrinks. reported confidence and actual accuracy move in opposite
    # directions, which is the definition of the failure
    assert late_error > early_error

    # and the only diagnostic that sees it is the innovation sequence
    assert abs(autocorr) > 0.5, "expected strongly autocorrelated innovations"

    print()
    print("day 2 checks passed")
