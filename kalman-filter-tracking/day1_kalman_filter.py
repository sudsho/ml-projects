"""
Day 1 of the Kalman filter from scratch.

The HMM project a few weeks ago spent four days on forward-backward over a
discrete state: alpha_t propagated a distribution over K hidden states through a
transition matrix, reweighted it by an emission probability, renormalized, and
moved on. The Kalman filter is that same recursion with the discrete state
replaced by a continuous one and the sum over states replaced by an integral.
Everything structural carries over. What changes is that the integral has a
closed form only because the model is linear and the noise is Gaussian, and that
closed form is what makes the whole thing O(1) per step instead of intractable.

The correspondence, made concrete, because it is the reason this project exists
directly after that one:

  HMM                                  Kalman
  ---                                  ------
  alpha_t(k), a vector over K states   (mean, covariance) of a Gaussian
  sum_j alpha_{t-1}(j) A[j,k]          predict: F @ mean, F @ cov @ F.T + Q
  multiply by B[k, obs_t]              update: reweight by N(obs; H @ mean, R)
  renormalize, keep the scale factor   the normalizer is the innovation density
  product of scale factors = P(obs)    sum of log innovation densities = P(obs)

That last row is not decoration. In the HMM the sequence likelihood fell out of
the forward pass for free as the product of the per-step normalizers, and it was
the thing Baum-Welch monotonically increased. The same quantity falls out here as
the sum of Gaussian log-densities of the innovations, and day 4's EM will lean on
it the same way. It is computed below even though nothing on day 1 consumes it,
because getting it right is easier now than retrofitting it later.

Today: the linear-Gaussian state-space model, a constant-velocity tracking
simulation, and the predict/update recursions with the innovation covariance
written out explicitly. Day 2 handles the numerical stability of the covariance
update, which is where naive implementations quietly rot.
"""

import numpy as np


def constant_velocity_model(dt, accel_noise, obs_noise):
    """Matrices for a 2-d point tracked by position alone.

    State is (x, y, vx, vy). The transition is exact Newtonian motion over a
    step of `dt` with no acceleration term, which is a lie the process noise is
    there to cover: `Q` is the covariance of a random acceleration held constant
    across the interval, so it is not diagonal. Position and velocity both
    inherit the same unknown acceleration, which correlates them.

    That off-diagonal structure is the piece worth not fudging. A diagonal `Q`
    is the usual shortcut and it tells the filter that a velocity error and the
    position error it necessarily produces are independent, which they are not.
    The filter still runs, it just trusts its own position estimate more than it
    has any right to.

    `H` observes position only. Velocity is never measured and has to be inferred
    from the sequence, which is the entire point of the exercise - a filter that
    observed the full state would have nothing to do.
    """
    transition = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # continuous white-noise acceleration integrated over the step. the block
    # is [[dt^3/3, dt^2/2], [dt^2/2, dt]] per axis, scaled by the acceleration
    # variance, and the dt^2/2 corners are the position/velocity correlation.
    q_pos = dt ** 3 / 3.0
    q_cross = dt ** 2 / 2.0
    q_vel = dt
    process_noise = accel_noise ** 2 * np.array(
        [
            [q_pos, 0.0, q_cross, 0.0],
            [0.0, q_pos, 0.0, q_cross],
            [q_cross, 0.0, q_vel, 0.0],
            [0.0, q_cross, 0.0, q_vel],
        ]
    )

    observation = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    observation_noise = obs_noise ** 2 * np.eye(2)

    return transition, process_noise, observation, observation_noise


def simulate(transition, process_noise, observation, observation_noise,
             initial_state, n_steps, rng):
    """Draw a trajectory and its noisy observations from the model itself.

    Sampling from the same model the filter assumes is deliberate for day 1. It
    means any error is an implementation error rather than a modelling error,
    which is the only way the consistency checks at the bottom of this file mean
    anything. Day 4 breaks the assumption on purpose and looks at what the
    diagnostics do.

    The noise is drawn via Cholesky factors rather than `multivariate_normal`
    so the correlated structure of `Q` is visible as a matrix multiply instead
    of hidden inside a library call.
    """
    state_dim = transition.shape[0]
    obs_dim = observation.shape[0]

    chol_q = np.linalg.cholesky(process_noise + 1e-12 * np.eye(state_dim))
    chol_r = np.linalg.cholesky(observation_noise)

    states = np.zeros((n_steps, state_dim))
    observations = np.zeros((n_steps, obs_dim))

    state = np.asarray(initial_state, dtype=float)
    for t in range(n_steps):
        state = transition @ state + chol_q @ rng.standard_normal(state_dim)
        states[t] = state
        observations[t] = observation @ state + chol_r @ rng.standard_normal(obs_dim)

    return states, observations


def predict(mean, cov, transition, process_noise):
    """Push the belief forward one step through the dynamics.

    The HMM's `sum_j alpha(j) A[j,k]`, in closed form. A Gaussian pushed through
    a linear map stays Gaussian, so the whole distribution is carried by two
    moments: the mean maps through `F`, and the covariance picks up the
    congruence `F P F.T` plus the process noise.

    The covariance grows here and only here. Prediction is the step that loses
    information, and `Q` is the statement of how fast it is lost. With `Q = 0`
    the filter eventually stops listening to observations entirely, which is a
    real failure mode and not a hypothetical one.
    """
    return transition @ mean, transition @ cov @ transition.T + process_noise


def update(mean, cov, obs, observation, observation_noise):
    """Fold one observation into the belief.

    The HMM's multiply-by-emission-then-renormalize. Three quantities, and the
    middle one is the one worth naming:

      innovation  v = y - H @ mean      what the observation said that the
                                        prediction did not expect
      innovation covariance
                  S = H P H.T + R       how surprised the filter is entitled to
                                        be, given its own uncertainty and the
                                        sensor's
      gain        K = P H.T S^-1        how much of the surprise to believe

    Writing `S` out explicitly rather than inlining it is not cosmetic. It is
    the normalizer of the HMM's forward step, so the sequence log-likelihood is
    built from it; and `v.T S^-1 v` is the normalized squared innovation that
    day 4 uses to decide whether the filter's own error bars are honest. Both
    consumers want the same matrix, and neither is visible if it stays inline.

    The gain is solved rather than inverted. `K = P H.T S^-1` transposes to
    `K.T = S^-1 H P` using the symmetry of `S` and `P`, so one `solve` on the
    smaller observation-space system does it. On a 2-d observation that is not
    a speed argument, it is a habit for the case where it is.
    """
    innovation = obs - observation @ mean
    innovation_cov = observation @ cov @ observation.T + observation_noise

    gain = np.linalg.solve(innovation_cov, observation @ cov).T

    new_mean = mean + gain @ innovation
    # the textbook form. it is algebraically correct and numerically fragile,
    # which is day 2's entire subject - the subtraction can drive the result
    # out of the positive-definite cone over a long run.
    new_cov = cov - gain @ innovation_cov @ gain.T

    return new_mean, new_cov, innovation, innovation_cov


def gaussian_log_density(residual, cov):
    """log N(residual; 0, cov), via Cholesky rather than an explicit inverse.

    `logdet` comes off the Cholesky diagonal and the quadratic form from solving
    against the factor: if `L L.T = cov` and `L z = residual`, then
    `residual.T cov^-1 residual = z.T z`. Both are cheaper than the naive route
    and, more to the point, both fail loudly if `cov` has drifted out of
    positive-definiteness - which is exactly the corruption day 2 is about. An
    `inv`-based version would happily return a finite number for an indefinite
    covariance.

    (`np.linalg` has no triangular solve - that is SciPy - so this is a general
    solve on a matrix that happens to be triangular. Dependency-free is worth
    more here than the constant factor.)
    """
    dim = residual.shape[0]
    chol = np.linalg.cholesky(cov)
    solved = np.linalg.solve(chol, residual)

    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    return -0.5 * (dim * np.log(2.0 * np.pi) + log_det + solved @ solved)


def kalman_filter(observations, transition, process_noise, observation,
                  observation_noise, initial_mean, initial_cov):
    """Run the forward recursion over a whole sequence.

    Returns the filtered means and covariances, the innovations and their
    covariances, and the sequence log-likelihood.

    The log-likelihood is the prediction-error decomposition: the joint density
    of the observations factorizes into one-step-ahead predictive densities, and
    each of those is `N(y_t; H @ predicted_mean, S_t)`. This is literally the
    HMM's "product of the per-step normalizers", written additively. Nothing on
    day 1 uses it. Day 4's EM will, and a likelihood that only appears once
    something depends on it tends to appear with a bug in it.

    The predicted quantities are kept alongside the filtered ones because the
    RTS smoother on day 3 needs both - it runs backward through the same steps
    and requires the prediction that each update corrected.
    """
    n_steps = observations.shape[0]
    state_dim = transition.shape[0]
    obs_dim = observation.shape[0]

    filtered_means = np.zeros((n_steps, state_dim))
    filtered_covs = np.zeros((n_steps, state_dim, state_dim))
    predicted_means = np.zeros((n_steps, state_dim))
    predicted_covs = np.zeros((n_steps, state_dim, state_dim))
    innovations = np.zeros((n_steps, obs_dim))
    innovation_covs = np.zeros((n_steps, obs_dim, obs_dim))

    mean = np.asarray(initial_mean, dtype=float)
    cov = np.asarray(initial_cov, dtype=float)
    log_likelihood = 0.0

    for t in range(n_steps):
        mean, cov = predict(mean, cov, transition, process_noise)
        predicted_means[t] = mean
        predicted_covs[t] = cov

        mean, cov, innovation, innovation_cov = update(
            mean, cov, observations[t], observation, observation_noise
        )

        filtered_means[t] = mean
        filtered_covs[t] = cov
        innovations[t] = innovation
        innovation_covs[t] = innovation_cov
        log_likelihood += gaussian_log_density(innovation, innovation_cov)

    return {
        "filtered_means": filtered_means,
        "filtered_covs": filtered_covs,
        "predicted_means": predicted_means,
        "predicted_covs": predicted_covs,
        "innovations": innovations,
        "innovation_covs": innovation_covs,
        "log_likelihood": log_likelihood,
    }


def rmse(estimates, truth):
    """Root mean squared error over a trajectory of vectors."""
    return float(np.sqrt(np.mean(np.sum((estimates - truth) ** 2, axis=1))))


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    dt = 0.1
    transition, process_noise, observation, observation_noise = constant_velocity_model(
        dt=dt, accel_noise=0.6, obs_noise=1.2
    )

    true_initial = np.array([0.0, 0.0, 2.0, 1.0])
    states, observations = simulate(
        transition, process_noise, observation, observation_noise,
        true_initial, n_steps=400, rng=rng,
    )

    # the filter is not told the true initial state. a diffuse prior on the
    # unobserved velocities is the honest starting point, and watching how fast
    # it recovers them from position observations alone is the demo.
    initial_mean = np.array([0.0, 0.0, 0.0, 0.0])
    initial_cov = np.diag([5.0, 5.0, 25.0, 25.0])

    result = kalman_filter(
        observations, transition, process_noise, observation,
        observation_noise, initial_mean, initial_cov,
    )

    filtered_pos = result["filtered_means"][:, :2]
    true_pos = states[:, :2]
    true_vel = states[:, 2:]

    print(f"steps                     : {len(observations)}")
    print(f"raw observation RMSE      : {rmse(observations, true_pos):.4f}")
    print(f"filtered position RMSE    : {rmse(filtered_pos, true_pos):.4f}")
    print(f"filtered velocity RMSE    : {rmse(result['filtered_means'][:, 2:], true_vel):.4f}")
    print(f"sequence log-likelihood   : {result['log_likelihood']:.2f}")

    # the filter must beat its own input. if it does not, something is wrong
    # with the gain - a filter that cannot outperform the raw sensor is just an
    # expensive copy of it.
    assert rmse(filtered_pos, true_pos) < rmse(observations, true_pos)

    # covariances stay symmetric positive definite on a clean run. this passes
    # here and is exactly the assertion that starts failing on long runs with
    # a badly conditioned R, which is day 2.
    for cov in result["filtered_covs"]:
        assert np.allclose(cov, cov.T, atol=1e-9)
        assert np.all(np.linalg.eigvalsh(cov) > 0)

    # velocity is never observed, so the only reason the estimate exists is the
    # position/velocity correlation the filter accumulates. it should be well
    # inside the diffuse prior it started from.
    final_vel_error = np.linalg.norm(result["filtered_means"][-1, 2:] - true_vel[-1])
    print(f"final velocity error      : {final_vel_error:.4f}")
    assert final_vel_error < 5.0

    # innovations are a martingale difference sequence under the true model, so
    # they should look white and zero-mean. a filter with the wrong Q shows up
    # here as autocorrelated innovations long before it shows up in the RMSE.
    innovations = result["innovations"]
    print(f"innovation mean           : {np.mean(innovations, axis=0)}")
    assert np.all(np.abs(np.mean(innovations, axis=0)) < 0.3)

    # normalized innovation squared should average the observation dimension.
    # this is the NIS consistency check day 4 uses in earnest; running it now
    # confirms S is the right matrix rather than merely a plausible one.
    nis = np.array(
        [
            innovations[t] @ np.linalg.solve(result["innovation_covs"][t], innovations[t])
            for t in range(len(innovations))
        ]
    )
    print(f"mean NIS (expect ~2.0)    : {np.mean(nis):.4f}")
    assert 1.5 < np.mean(nis) < 2.5

    print("day 1 checks passed")
