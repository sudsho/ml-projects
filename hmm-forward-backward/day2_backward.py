"""
Day 2 of the HMM: the backward recursion and the posterior state marginals.

Yesterday's forward variable answered "what is the probability of the evidence",
and it does that by carrying a joint over the past:

    alpha[t, i] = P(x_1..x_t, z_t = i)

which is everything that happened up to and including t. But the question you
usually want answered is not about the evidence, it is about the states: given
the WHOLE sequence, what was the chain doing at step t? Alpha cannot answer that
on its own, because at time t it has not seen x_{t+1}..x_T yet, and those
observations are informative about z_t - the chain is sticky, so a run of
rain-ish emissions after t is evidence that it was raining at t.

The missing half is the backward variable

    beta[t, i] = P(x_{t+1}..x_T | z_t = i)

which is a CONDITIONAL, not a joint - it is the likelihood of the future given
the state, with no prior on the state attached. That asymmetry with alpha is the
detail worth getting right, and it is exactly why the two multiply cleanly:

    alpha[t, i] * beta[t, i] = P(x_1..x_t, z_t = i) * P(x_{t+1}..x_T | z_t = i)
                             = P(x_1..x_T, z_t = i)

The conditional independence that licenses the split is the Markov property
itself - given z_t, the past and the future of the sequence are independent, so
the evidence factors into a piece before t and a piece after t. Every quantity
today is a consequence of that one line.

Two things fall out immediately:

  * the posterior marginal, gamma[t, i] = P(z_t = i | x_1..x_T), obtained by
    normalizing that product over i. This is the smoothed estimate of the state,
    and it is strictly better informed than the filtered estimate alpha gives.

  * the likelihood computed at ANY t, since summing alpha * beta over i
    marginalizes z_t out and returns P(x_1..x_T) regardless of which t you
    picked. That is a genuinely strong self-check: T independent recomputations
    of the same number, one per time step, all of which must agree. A sign error
    or a transposed A survives a lot of eyeballing but not this.

Everything is in log space again for the same underflow reason as day 1 - beta
decays away from the ends just as alpha does.
"""

import numpy as np

from day1_forward import (
    check_hmm_params,
    forward_log,
    logsumexp,
    make_toy_model,
    sample_sequence,
)


def backward_log(pi, a, b, obs):
    """Backward recursion in log space.

    beta[t, i] = P(x_{t+1}..x_T | z_t = i), built from the end backwards:

        beta[T-1, i] = 1                      (log 0, empty future)
        beta[t, i]   = sum_j A[i,j] B[j,x_{t+1}] beta[t+1, j]

    Note the emission index inside the recursion is t+1, not t. That is the
    single most common place to put an off-by-one in this algorithm, and it
    follows directly from beta conditioning on z_t while covering observations
    strictly after t - x_t belongs to alpha's half of the factorization, and
    including it here would double-count it in every product alpha * beta.

    The initialization at 1 (not at the emission, not at pi) is the same fact
    stated at the boundary: after the last step there is no future evidence, so
    the conditional probability of observing it is trivially one.
    """
    pi = np.asarray(pi, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    obs = np.asarray(obs, dtype=int)

    with np.errstate(divide="ignore"):
        log_a, log_b = np.log(a), np.log(b)

    n_steps = len(obs)
    log_beta = np.empty((n_steps, pi.shape[0]))
    log_beta[-1] = 0.0  # log 1

    for t in range(n_steps - 2, -1, -1):
        # broadcast to (K_current, K_next) and reduce over the NEXT state.
        # mirror image of the forward step, which reduced over the previous
        # one - hence axis=1 here against axis=0 there.
        scores = log_a + log_b[:, obs[t + 1]] + log_beta[t + 1]
        log_beta[t] = logsumexp(scores, axis=1)

    return log_beta


def posterior_marginals(log_alpha, log_beta):
    """gamma[t, i] = P(z_t = i | x_1..x_T), the smoothed state posterior.

    log gamma[t, i] = log alpha[t, i] + log beta[t, i] - log P(x_1..x_T)

    The normalizer is the same scalar at every t, which is the consistency
    property above; it is recomputed per row anyway because dividing by the
    row's own logsumexp is what forces each row to sum to exactly one rather
    than to one-plus-rounding-error.
    """
    log_gamma = log_alpha + log_beta
    log_gamma = log_gamma - logsumexp(log_gamma, axis=1)[:, None]
    return np.exp(log_gamma)


def log_likelihood_at_each_t(log_alpha, log_beta):
    """Marginalize z_t out of alpha * beta at every t; all entries must match.

    Returns a length-T vector. Constant to floating-point precision if the two
    recursions are consistent, and the shape of the deviation is diagnostic
    when they are not - an error at one end that grows toward the other usually
    means a transposed transition matrix, since A and A.T agree only on the
    boundary conditions.
    """
    return logsumexp(log_alpha + log_beta, axis=1)


def filtered_marginals(log_alpha):
    """P(z_t = i | x_1..x_t) - the online estimate, using only the past.

    Kept for the comparison against gamma. This is what a real-time system
    would have available at step t; gamma is what you get once the sequence is
    complete. The gap between them is the value of hindsight, and it is largest
    exactly where the observation at t is ambiguous and the resolution comes
    from what happened afterwards.
    """
    return np.exp(log_alpha - logsumexp(log_alpha, axis=1)[:, None])


def posterior_decode(gamma):
    """Most likely state at each step independently: argmax_i gamma[t, i].

    Maximizes the expected number of individually-correct states, which is not
    the same objective as the most likely PATH - this decode can emit a
    transition that A gives zero probability to, because nothing here couples
    adjacent choices. That failure mode is the reason Viterbi exists, and it is
    tomorrow's problem; the two decodes are compared directly there.
    """
    return np.argmax(gamma, axis=1)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    pi, a, b = make_toy_model()
    n_states, n_obs = check_hmm_params(pi, a, b)

    states, obs = sample_sequence(pi, a, b, n_steps=200, rng=rng)
    log_alpha, ll_forward = forward_log(pi, a, b, obs)
    log_beta = backward_log(pi, a, b, obs)

    # check 1: the backward recursion reproduces the forward likelihood at t=0.
    # here beta carries almost everything and alpha almost nothing, so it is
    # the least forgiving place to compare them.
    ll_backward = logsumexp(np.log(pi) + np.log(b[:, obs[0]]) + log_beta[0])
    print(f"forward  log-lik  = {ll_forward:.10f}")
    print(f"backward log-lik  = {ll_backward:.10f}")
    print(f"abs diff          = {abs(ll_forward - ll_backward):.3e}")
    assert np.isclose(ll_forward, ll_backward)

    # check 2: the stronger one. alpha * beta summed over states must give the
    # same likelihood at every single t, not just at the two ends.
    per_t = log_likelihood_at_each_t(log_alpha, log_beta)
    spread = per_t.max() - per_t.min()
    print(f"\nlog-lik recomputed at all T={len(obs)} time steps")
    print(f"  min {per_t.min():.10f}  max {per_t.max():.10f}")
    print(f"  spread = {spread:.3e} (should be float noise)")
    assert spread < 1e-8, "alpha and beta disagree in the interior"

    gamma = posterior_marginals(log_alpha, log_beta)
    filtered = filtered_marginals(log_alpha)
    print(f"\ngamma rows sum to one: {np.allclose(gamma.sum(axis=1), 1.0)}")

    # smoothing should beat filtering, because it conditions on strictly more
    # evidence. comparing accuracy against the sampled states is legitimate
    # here only because this is synthetic data where the truth is known.
    smoothed_path = posterior_decode(gamma)
    filtered_path = np.argmax(filtered, axis=1)
    print(f"\nfiltered  accuracy vs truth: {(filtered_path == states).mean():.4f}")
    print(f"smoothed  accuracy vs truth: {(smoothed_path == states).mean():.4f}")

    # and the confidence should be better calibrated too - mean posterior
    # probability assigned to the state that actually occurred.
    idx = np.arange(len(states))
    print(f"mean P(true state) filtered: {filtered[idx, states].mean():.4f}")
    print(f"mean P(true state) smoothed: {gamma[idx, states].mean():.4f}")

    # where does hindsight actually help? at the steps where the filtered
    # estimate was least certain, which is what you would hope.
    disagree = np.flatnonzero(smoothed_path != filtered_path)
    print(f"\nthe two decodes disagree at {len(disagree)} of {len(obs)} steps")
    if len(disagree):
        conf = filtered.max(axis=1)
        print(f"  mean filtered confidence there : {conf[disagree].mean():.4f}")
        print(f"  mean filtered confidence overall: {conf.mean():.4f}")
        t = int(disagree[0])
        print(f"  e.g. t={t}: obs={obs[t]} truth={states[t]}"
              f" filtered={np.round(filtered[t], 3)}"
              f" smoothed={np.round(gamma[t], 3)}")

    # the last step is the one place the two must agree exactly, since beta is
    # 1 there and gamma collapses to the normalized alpha. cheap boundary test.
    assert np.allclose(gamma[-1], filtered[-1]), "gamma != filtered at t=T"
    print("\ngamma == filtered at the final step, as it must be (beta = 1 there)")
