"""
Day 1 of a hidden Markov model from scratch.

An HMM is two coupled sequences. The one you want, z_1..z_T, is a Markov chain
over K discrete states - z_t depends on z_{t-1} and on nothing earlier. The one
you actually observe, x_1..x_T, is emitted independently from whichever state is
active at that step. The whole model is three matrices:

    pi[i]     = P(z_1 = i)              initial distribution, length K
    A[i, j]   = P(z_t = j | z_{t-1} = i)  transition matrix, K x K
    B[i, o]   = P(x_t = o | z_t = i)      emission matrix, K x M

Every row of all three is a probability distribution and must sum to one. That
constraint is worth enforcing in code rather than trusting, because a row that
quietly sums to 0.999 produces likelihoods that are wrong by a factor that grows
exponentially in T and looks like a subtle bug for a long time.

The central quantity today is the sequence likelihood P(x_1..x_T). Written
directly it is a sum over every possible state path, of which there are K^T -
already 10^30 paths for K=3, T=63. The forward recursion computes the same sum
in O(T K^2) by exploiting the Markov property: the paths only interact through
which state they are in at time t, so they can be merged there. That is the
entire trick, and it is the same dynamic-programming move as the diff-array
collapse - a pile of things that would have to be enumerated separately turns
out to compose, so it can be summarized by a single running quantity.

The forward variable is

    alpha[t, i] = P(x_1..x_t, z_t = i)

a JOINT, not a conditional - it carries the probability mass of the evidence so
far along with it. That matters numerically: each step multiplies by an emission
probability, so alpha decays roughly geometrically and underflows float64 in a
few hundred steps. The fix here is to carry log alpha and replace every sum with
log-sum-exp, which is exact up to rounding and never underflows. (The other
standard fix, per-step rescaling with saved normalizers, comes up tomorrow when
the backward pass has to use the same scale factors.)

Run this file directly to see the naive version underflow to exactly zero while
the log-space version keeps reporting a sensible per-step log-likelihood.
"""

import numpy as np


def check_hmm_params(pi, a, b, tol=1e-9):
    """Validate shapes and row-stochasticity of the three HMM matrices.

    Returns (K, M). Raises rather than warns - every downstream quantity is a
    probability, and a model whose rows do not sum to one produces numbers that
    still look like probabilities while being silently wrong.
    """
    pi = np.asarray(pi, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if pi.ndim != 1:
        raise ValueError(f"pi must be 1-D, got shape {pi.shape}")
    n_states = pi.shape[0]
    if a.shape != (n_states, n_states):
        raise ValueError(f"A must be {(n_states, n_states)}, got {a.shape}")
    if b.ndim != 2 or b.shape[0] != n_states:
        raise ValueError(f"B must have {n_states} rows, got shape {b.shape}")

    for name, m in (("pi", pi[None, :]), ("A", a), ("B", b)):
        if np.any(m < -tol):
            raise ValueError(f"{name} has negative entries")
        sums = m.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=tol):
            worst = int(np.argmax(np.abs(sums - 1.0)))
            raise ValueError(
                f"{name} row {worst} sums to {sums[worst]:.12f}, not 1"
            )

    return n_states, b.shape[1]


def sample_sequence(pi, a, b, n_steps, rng=None):
    """Draw (states, observations) of length n_steps from the model.

    Ancestral sampling straight down the generative story: pick z_1 from pi,
    then alternate emitting x_t from B[z_t] and stepping z_{t+1} from A[z_t].
    Having a sampler before any inference code is what makes the inference
    testable at all - it is the only way to compare a decoded path against the
    truth, since in a real application the states are by definition unseen.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n_states, n_obs = check_hmm_params(pi, a, b)

    states = np.empty(n_steps, dtype=int)
    observations = np.empty(n_steps, dtype=int)

    states[0] = rng.choice(n_states, p=pi)
    observations[0] = rng.choice(n_obs, p=b[states[0]])
    for t in range(1, n_steps):
        states[t] = rng.choice(n_states, p=a[states[t - 1]])
        observations[t] = rng.choice(n_obs, p=b[states[t]])

    return states, observations


def forward_naive(pi, a, b, obs):
    """Forward recursion in probability space. Underflows, kept for contrast.

    alpha[t, i] = P(x_1..x_t, z_t = i), built by

        alpha[0, i] = pi[i] * B[i, x_0]
        alpha[t, j] = (sum_i alpha[t-1, i] * A[i, j]) * B[j, x_t]

    Returns (alpha, likelihood). The likelihood is alpha[-1].sum(), which
    marginalizes out the final state.
    """
    pi = np.asarray(pi, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    obs = np.asarray(obs, dtype=int)

    n_steps = len(obs)
    alpha = np.zeros((n_steps, pi.shape[0]))
    alpha[0] = pi * b[:, obs[0]]
    for t in range(1, n_steps):
        # alpha[t-1] @ A sums over the previous state; the emission is a
        # pointwise gate applied after the transition has been marginalized.
        alpha[t] = (alpha[t - 1] @ a) * b[:, obs[t]]

    return alpha, alpha[-1].sum()


def logsumexp(x, axis=None):
    """Stable log(sum(exp(x))): factor out the max before exponentiating.

    Without the shift, exp of a log-probability around -800 is exactly 0 and the
    log of that is -inf. With it, the largest term is exp(0) = 1 and everything
    else is a positive-or-smaller correction, so the only loss is in terms that
    were negligible anyway. The all-(-inf) row is handled explicitly because
    -inf - -inf is nan, and a state with zero probability is a normal thing to
    have rather than an error.
    """
    x = np.asarray(x, dtype=float)
    x_max = np.max(x, axis=axis, keepdims=True)
    x_max = np.where(np.isfinite(x_max), x_max, 0.0)
    out = np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return np.squeeze(out, axis=axis) if axis is not None else out.item()


def forward_log(pi, a, b, obs):
    """Forward recursion in log space. This is the one everything else uses.

    Identical recursion with products becoming sums and the marginalizing sum
    becoming a log-sum-exp:

        log_alpha[t, j] = logsumexp_i(log_alpha[t-1, i] + logA[i, j]) + logB[j, x_t]

    Returns (log_alpha, log_likelihood). Runs in O(T K^2) - the K^2 is the
    all-pairs transition term, and it is what replaces the K^T path enumeration.
    """
    pi = np.asarray(pi, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    obs = np.asarray(obs, dtype=int)

    # log(0) is a legitimate -inf here (a forbidden transition), so silence the
    # divide warning rather than nudging zeros to epsilon - the arithmetic below
    # is written to survive -inf, and faking the zeros would hide real structure.
    with np.errstate(divide="ignore"):
        log_pi, log_a, log_b = np.log(pi), np.log(a), np.log(b)

    n_steps = len(obs)
    log_alpha = np.empty((n_steps, pi.shape[0]))
    log_alpha[0] = log_pi + log_b[:, obs[0]]
    for t in range(1, n_steps):
        # broadcast to (K_prev, K_next), reduce over the previous state
        scores = log_alpha[t - 1][:, None] + log_a
        log_alpha[t] = logsumexp(scores, axis=0) + log_b[:, obs[t]]

    return log_alpha, logsumexp(log_alpha[-1])


def brute_force_log_likelihood(pi, a, b, obs):
    """Enumerate all K^T state paths and sum their joint probabilities.

    Only usable for tiny T, which is the point - it is the ground truth that
    proves the forward recursion is computing the marginal and not something
    that merely looks plausible.
    """
    from itertools import product

    pi = np.asarray(pi, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_states = pi.shape[0]

    total = 0.0
    for path in product(range(n_states), repeat=len(obs)):
        p = pi[path[0]] * b[path[0], obs[0]]
        for t in range(1, len(obs)):
            p *= a[path[t - 1], path[t]] * b[path[t], obs[t]]
        total += p

    return np.log(total)


def make_toy_model():
    """Two-state weather model with a three-symbol emission alphabet.

    State 0 is "rainy", state 1 is "sunny"; both are sticky, so the chain has
    real temporal structure to recover. Emissions are walk / shop / clean, the
    standard Wikipedia toy, chosen because the states are only weakly
    identifiable from a single observation - inference has to use the sequence.
    """
    pi = np.array([0.6, 0.4])
    a = np.array([[0.7, 0.3],
                  [0.4, 0.6]])
    b = np.array([[0.1, 0.4, 0.5],
                  [0.6, 0.3, 0.1]])
    return pi, a, b


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pi, a, b = make_toy_model()
    n_states, n_obs = check_hmm_params(pi, a, b)
    print(f"model validated: K={n_states} states, M={n_obs} symbols")

    states, obs = sample_sequence(pi, a, b, n_steps=12, rng=rng)
    print(f"\nsampled states: {states}")
    print(f"sampled obs   : {obs}")
    # stationary-ish occupancy check: with these transitions the chain should
    # spend a bit more time rainy than sunny over a long run.
    long_states, _ = sample_sequence(pi, a, b, n_steps=20000, rng=rng)
    occupancy = np.bincount(long_states, minlength=n_states) / len(long_states)
    print(f"empirical occupancy over 20k steps: {np.round(occupancy, 4)}")

    # correctness: the recursion must agree with explicit path enumeration.
    short = obs[:8]
    ll_forward = forward_log(pi, a, b, short)[1]
    ll_brute = brute_force_log_likelihood(pi, a, b, short)
    print(f"\nT=8, {n_states ** 8} paths enumerated")
    print(f"  forward  log-lik = {ll_forward:.12f}")
    print(f"  brute    log-lik = {ll_brute:.12f}")
    print(f"  abs diff         = {abs(ll_forward - ll_brute):.3e}")
    assert np.isclose(ll_forward, ll_brute), "forward recursion disagrees"

    # and the naive version must agree too, while it still can.
    ll_naive = np.log(forward_naive(pi, a, b, short)[1])
    print(f"  naive    log-lik = {ll_naive:.12f}")
    assert np.isclose(ll_naive, ll_brute)

    # now the reason log space exists. alpha decays by roughly one emission
    # probability per step, so the mass falls off the bottom of float64.
    print("\nnaive vs log-space as T grows:")
    print(f"  {'T':>6}  {'naive P':>12}  {'log-space ll':>14}  {'ll / T':>9}")
    for n_steps in (10, 50, 100, 400, 1500):
        _, seq = sample_sequence(pi, a, b, n_steps=n_steps, rng=rng)
        p_naive = forward_naive(pi, a, b, seq)[1]
        ll = forward_log(pi, a, b, seq)[1]
        print(f"  {n_steps:>6}  {p_naive:>12.4e}  {ll:>14.4f}  {ll / n_steps:>9.4f}")

    print("\nnaive underflows to exactly 0 somewhere past T=400; the log-space")
    print("per-step likelihood stays flat, which is what it should do for a")
    print("stationary chain - it is an average, not a total.")
