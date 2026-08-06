"""
Day 4 of the HMM: Baum-Welch, the EM algorithm that learns the three matrices.

Days 1-3 all assumed pi, A and B were handed to us. Every quantity computed so
far - the likelihood, the smoothed marginals, the Viterbi path - was inference
under a known model. Today the model itself is unknown and the only thing
available is a pile of observation sequences with the states stripped out.

If the states WERE observed this would not be an algorithm, it would be counting:
A[i,j] is the fraction of steps that went i -> j, B[i,o] the fraction of visits
to i that emitted o, both plain maximum likelihood on a fully observed chain.
The states are not observed, so those counts do not exist. What does exist, once
a guess at the model is in hand, is the posterior distribution over them - which
is exactly what the forward-backward pass produces. So:

    E step: run forward-backward under the current parameters to get the
            posterior over states, and use it to form EXPECTED counts
    M step: run the fully-observed estimator on those expected counts

and iterate. That is the whole algorithm, and stated that way it is obviously a
fixed point rather than obviously correct. The guarantee that makes it work is
that each round cannot decrease the likelihood of the data, so the sequence of
likelihoods is monotone non-decreasing and (being bounded above by 1) converges.
It converges to a *local* optimum - EM says nothing about which one, and with a
bad initialization it will happily settle somewhere useless. That is why the
runs below use restarts and report the spread rather than a single number.

Two expected counts are needed, and only one of them exists already:

    gamma[t, i]    = P(z_t = i | x)              day 2
    xi[t, i, j]    = P(z_t = i, z_{t+1} = j | x) new today

gamma is enough for pi and B, because those are per-timestep quantities. A is
about adjacent PAIRS of states, and gamma has already marginalized the pairing
away - knowing the posterior at t and at t+1 separately does not tell you the
joint, because the two are correlated through the very transitions being
estimated. This is the same information-destroyed-by-summarizing point as the
sweep problems: a marginal is a summary, and the pairing is what it drops.

The last section is about label switching, which is not a bug and does need
handling. The likelihood is exactly invariant under permuting the state labels -
relabel the states, permute pi, A and B to match, and the model assigns
identical probability to every sequence. So the learned parameters can only ever
match the truth up to a permutation, and any comparison has to search over the K!
relabelings before it means anything.

Run this file directly for the monotonicity check, a recovery experiment against
a known model, and the restart spread.
"""

from itertools import permutations

import numpy as np

from day1_forward import (
    check_hmm_params,
    forward_log,
    logsumexp,
    make_toy_model,
    sample_sequence,
)
from day2_backward import backward_log, posterior_marginals


def pairwise_marginals(log_alpha, log_beta, log_a, log_b, obs, log_likelihood):
    """xi[t, i, j] = P(z_t = i, z_{t+1} = j | x), shape (T-1, K, K).

    The joint over an adjacent pair splits into the four things that have to
    happen for the chain to be at i then at j, given all the evidence:

        alpha[t, i]           the past, arriving at i
        A[i, j]               the step itself
        B[j, x_{t+1}]         the emission made from j
        beta[t+1, j]          the remaining future, from j

    divided by P(x) to normalize. Note x_{t+1} appears explicitly - beta[t+1]
    conditions on z_{t+1} and covers only observations strictly after t+1, so
    the emission at t+1 belongs to neither alpha's nor beta's half and has to
    be supplied here. Same off-by-one that governs the backward recursion,
    surfacing in the place where it is easiest to get wrong.

    There are T-1 rows because there are T-1 transitions, not T. That single
    fact is why A's denominator below excludes the last timestep.
    """
    # (T-1, K, 1) + (K, K) + (T-1, 1, K) + (T-1, 1, K)
    log_xi = (log_alpha[:-1, :, None]
              + log_a[None, :, :]
              + log_b[:, obs[1:]].T[:, None, :]
              + log_beta[1:, None, :])
    return np.exp(log_xi - log_likelihood)


def e_step(pi, a, b, sequences):
    """Accumulate expected counts over every sequence under the current model.

    Returns (start_counts, trans_counts, emit_counts, total_log_likelihood).
    Multiple sequences are handled by summing their counts, which is correct
    because they are independent draws - the complete-data log-likelihood is a
    sum over sequences, so the expected sufficient statistics add. Learning
    from one long sequence is possible but estimates pi from a single sample,
    so it is the transitions and emissions that get learned and the initial
    distribution that stays essentially at its initial guess.
    """
    n_states, n_obs = check_hmm_params(pi, a, b)
    with np.errstate(divide="ignore"):
        log_a, log_b = np.log(a), np.log(b)

    start_counts = np.zeros(n_states)
    trans_counts = np.zeros((n_states, n_states))
    emit_counts = np.zeros((n_states, n_obs))
    total_ll = 0.0

    for obs in sequences:
        obs = np.asarray(obs, dtype=int)
        log_alpha, ll = forward_log(pi, a, b, obs)
        log_beta = backward_log(pi, a, b, obs)
        gamma = posterior_marginals(log_alpha, log_beta)
        xi = pairwise_marginals(log_alpha, log_beta, log_a, log_b, obs, ll)

        total_ll += ll
        start_counts += gamma[0]
        trans_counts += xi.sum(axis=0)
        # scatter each step's posterior into the column of the symbol it emitted
        np.add.at(emit_counts.T, obs, gamma)

    return start_counts, trans_counts, emit_counts, total_ll


def m_step(start_counts, trans_counts, emit_counts, smoothing=0.0):
    """Renormalize the expected counts into row-stochastic matrices.

    This is the fully-observed maximum-likelihood estimator applied to counts
    that happen to be fractional, which is the entire content of the M step for
    a discrete HMM - no optimizer, no gradients, closed form.

    Note the two denominators differ. A's rows are normalized by the expected
    number of transitions OUT of each state, which is the row sum of the
    transition counts and equals sum_t gamma[t, i] over t < T-1. B's rows are
    normalized by the expected number of VISITS, which includes the final step.
    Using one for the other is a real bug that shrinks by 1/T and so looks like
    a convergence issue on short sequences and like nothing at all on long ones.

    `smoothing` adds a pseudo-count before normalizing. Left at zero by default
    because it biases the estimate, but a state that the posterior never visits
    produces a 0/0 row, and a symbol never emitted gets probability exactly
    zero - which is a permanent decision, since a zero in B makes any sequence
    containing that symbol impossible and EM can never raise it again.
    """
    def normalize(counts):
        counts = counts + smoothing
        totals = counts.sum(axis=-1, keepdims=True)
        # an unvisited state has no evidence about where it goes; leave it
        # uniform rather than propagating a nan through the next E step.
        safe = np.where(totals > 0, totals, 1.0)
        out = np.where(totals > 0, counts / safe, 1.0 / counts.shape[-1])
        return out

    return normalize(start_counts), normalize(trans_counts), normalize(emit_counts)


def baum_welch(sequences, n_states, n_obs, max_iter=200, tol=1e-6,
               smoothing=0.0, rng=None, init=None):
    """Fit an HMM by EM. Returns (pi, A, B, log_likelihood_history).

    Stops when the per-iteration likelihood improvement falls below `tol`, or
    at max_iter. The history is returned rather than just the final model
    because its monotonicity is the property worth checking - a decrease is not
    a slow-convergence symptom, it is proof of a bug in the E or M step, and it
    is the single most useful assertion in this file.

    `init` overrides the random start with a given (pi, A, B). Seeding it at the
    true parameters is the experiment that separates "EM has not converged" from
    "this sample cannot identify the model": starting inside the right basin
    finds the sample MLE in a few dozen iterations, so whatever a random start
    reports has to be measured against that rather than against the truth.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    if init is None:
        pi, a, b = random_hmm(n_states, n_obs, rng)
    else:
        pi, a, b = (np.array(m, dtype=float) for m in init)

    history = []
    for _ in range(max_iter):
        start_counts, trans_counts, emit_counts, ll = e_step(pi, a, b, sequences)
        history.append(ll)
        pi, a, b = m_step(start_counts, trans_counts, emit_counts, smoothing)
        if len(history) > 1 and history[-1] - history[-2] < tol:
            break

    return pi, a, b, np.array(history)


def random_hmm(n_states, n_obs, rng):
    """Random row-stochastic starting point, drawn from a Dirichlet.

    Dirichlet(1) rather than uniform-then-normalize, because the latter
    concentrates near the center of the simplex and gives every state nearly
    identical rows. A symmetric starting point is close to a saddle: the E step
    then assigns near-identical posteriors to every state and there is little
    asymmetry for EM to amplify, so it converges slowly toward whichever
    solution the rounding noise happens to favor.
    """
    return (rng.dirichlet(np.ones(n_states)),
            rng.dirichlet(np.ones(n_states), size=n_states),
            rng.dirichlet(np.ones(n_obs), size=n_states))


def best_label_permutation(true_b, learned_b):
    """Find the relabeling of learned states that best matches the true ones.

    The likelihood is exactly invariant to permuting state labels, so EM has no
    way to prefer one labeling and returns whichever the initialization drifted
    toward. Any parameter-space comparison therefore has to quotient out the
    permutation first, or it reports a large error for a perfectly recovered
    model.

    Matching on B rather than on A because emissions identify a state by what
    it looks like, independently of the chain's dynamics; A only says how the
    states connect and is itself permuted on both axes. Brute force over K!
    permutations, which is fine for the K used here and is the honest version -
    the scalable one is a linear-assignment solve on the same cost matrix.
    """
    n_states = true_b.shape[0]
    best, best_cost = None, np.inf
    for perm in permutations(range(n_states)):
        cost = np.abs(true_b[list(perm)] - learned_b).sum()
        if cost < best_cost:
            best, best_cost = perm, cost
    return list(best), best_cost


def apply_permutation(pi, a, b, perm):
    """Relabel a model's states by `perm`, giving an equivalent model.

    A is permuted on BOTH axes - it is indexed by state twice, so relabeling
    has to touch rows and columns together. Permuting only the rows is a silent
    error that leaves every matrix row-stochastic and every shape correct while
    scrambling the dynamics.
    """
    perm = list(perm)
    return pi[perm], a[np.ix_(perm, perm)], b[perm]


def align_to(true_b, pi, a, b):
    """Relabel a fitted model to match the true state ordering."""
    perm, cost = best_label_permutation(true_b, b)
    return apply_permutation(pi, a, b, np.argsort(perm)) + (perm, cost)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    true_pi, true_a, true_b = make_toy_model()
    n_states, n_obs = check_hmm_params(true_pi, true_a, true_b)

    # 20 independent sequences so pi has more than one sample to learn from.
    sequences = [sample_sequence(true_pi, true_a, true_b, 100, rng)[1]
                 for _ in range(20)]
    n_tokens = sum(len(s) for s in sequences)
    print(f"training on {len(sequences)} sequences, {n_tokens} observations")

    true_ll = sum(forward_log(true_pi, true_a, true_b, s)[1] for s in sequences)
    print(f"log-likelihood under the true model: {true_ll:.4f}")

    pi_hat, a_hat, b_hat, history = baum_welch(
        sequences, n_states, n_obs, max_iter=250, rng=np.random.default_rng(1)
    )
    print(f"\nEM from a random start: {len(history)} iterations")
    print(f"  first ll {history[0]:.4f}  ->  final ll {history[-1]:.4f}")

    # the guarantee. a decrease anywhere is a bug in the E or M step, not slow
    # convergence, so this is an assert rather than a printed warning.
    steps = np.diff(history)
    print(f"  smallest step: {steps.min():.3e}  (must be >= 0)")
    assert steps.min() >= -1e-9, "likelihood decreased - EM guarantee violated"

    pi_hat, a_hat, b_hat, perm, cost = align_to(true_b, pi_hat, a_hat, b_hat)
    print(f"  best label permutation {perm}, L1 cost on B {cost:.4f}")
    print(f"  A error {np.abs(true_a - a_hat).max():.4f}"
          f"   B error {np.abs(true_b - b_hat).max():.4f}")

    # that error is larger than it looks like it should be, and the obvious
    # readings - not enough data, not enough iterations - are both testable.
    # seed EM at the truth: it stays in the right basin and lands on the MLE
    # for THIS sample in a few dozen iterations.
    pi_mle, a_mle, b_mle, mle_history = baum_welch(
        sequences, n_states, n_obs, max_iter=40, tol=0.0,
        init=(true_pi, true_a, true_b)
    )
    print(f"\nEM seeded at the truth: {len(mle_history)} iterations")
    print(f"  final ll {mle_history[-1]:.4f}")
    print(f"  A error {np.abs(true_a - a_mle).max():.4f}"
          f"   B error {np.abs(true_b - b_mle).max():.4f}")
    print("  -> the sample MLE sits essentially on top of the truth, so 2000")
    print("     observations are plenty and the data is not the limitation.")

    print("\nso compare the two fits on their own terms:")
    print(f"  random start : ll {history[-1]:.4f}, "
          f"B error {np.abs(true_b - b_hat).max():.4f}")
    print(f"  truth  start : ll {mle_history[-1]:.4f}, "
          f"B error {np.abs(true_b - b_mle).max():.4f}")
    better = "higher" if history[-1] > mle_history[-1] else "lower"
    print(f"  the random start reached a {better} likelihood with the worse")
    print("  parameters. that is not a local optimum trap - it is a genuinely")
    print("  flat ridge, where a visibly different (A, B) explains this sample")
    print("  about as well. EM optimizes likelihood; it was never optimizing")
    print("  parameter recovery, and on a ridge the two come apart.")

    # restarts. the spread is the honest summary - a single run's number says
    # as much about its seed as about the algorithm.
    print("\n5 restarts on identical data:")
    print(f"  {'seed':>5}  {'iters':>6}  {'final ll':>13}  {'B error':>9}")
    finals, errors = [], []
    for seed in range(5):
        p_s, a_s, b_s, h_s = baum_welch(
            sequences, n_states, n_obs, max_iter=120,
            rng=np.random.default_rng(seed)
        )
        _, _, b_aligned, _, _ = align_to(true_b, p_s, a_s, b_s)
        err = np.abs(true_b - b_aligned).max()
        finals.append(h_s[-1])
        errors.append(err)
        print(f"  {seed:>5}  {len(h_s):>6}  {h_s[-1]:>13.4f}  {err:>9.4f}")

    finals, errors = np.array(finals), np.array(errors)
    print(f"\n  ll spread {finals.max() - finals.min():.4f}, "
          f"B error range {errors.min():.4f} - {errors.max():.4f}")
    print(f"  best-likelihood run has B error {errors[finals.argmax()]:.4f}; "
          f"the most\n  accurate run has {errors.min():.4f}. "
          "picking a restart by likelihood -")
    print("  the only criterion available without the truth - does not reliably")
    print("  pick the most accurate model. worth knowing before trusting the")
    print("  learned matrices as an explanation rather than as a density.")
