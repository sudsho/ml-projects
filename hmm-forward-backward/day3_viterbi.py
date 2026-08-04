"""
Day 3 of the HMM: Viterbi decoding, and why it is not the same as yesterday.

Yesterday ended on gamma, the posterior marginal, and the decode that falls out
of it - take argmax_i gamma[t, i] independently at every step. That decode is
not wrong, it is answering a different question, and the difference is the whole
content of today.

The posterior decode maximizes the EXPECTED NUMBER OF CORRECT STATES. It picks
each step to be individually right as often as possible, and it is optimal for
that objective. But it chooses each step in isolation, so nothing in it knows
that consecutive states are coupled by A. Viterbi maximizes the probability of
the WHOLE PATH:

    posterior : z*_t = argmax_i  P(z_t = i | x)          for each t separately
    viterbi   : z*   = argmax_z  P(z_1..z_T | x)         over all K^T paths

Those two disagree, and not only marginally. The sharpest case is a transition
that A gives zero probability to. The posterior decode can emit it, because it
never evaluates the joint - each step was individually the most likely, and the
pair was never scored. So the "most likely state at every step" can be a path
of probability exactly zero. Viterbi cannot do this by construction, since it
only ever extends paths that exist.

The recursion is the forward recursion with sum replaced by max:

    delta[t, i] = max over paths ending at (t, i) of P(x_1..x_t, z_1..z_t)
    delta[0, i] = pi[i] B[i, x_0]
    delta[t, j] = max_i delta[t-1, i] A[i,j] * B[j, x_t]

That substitution is legitimate for the same reason the sum version was: the
paths interact only through which state they occupy at t, so a path that is not
the best way to arrive at (t, i) can never become the best way to arrive
anywhere later. Bellman's principle, and it is exactly the argument that lets
the forward pass merge paths at all - sum and max are both semiring operations
over the same structure, and the algorithm does not care which one it is given.

What the max version needs extra is a backpointer. Summing throws away which
predecessor contributed and does not need it back; maximizing has a single
winner per cell, and the path is recovered by walking those winners backwards
from the best final state. Storing psi[t, j] = argmax_i is the entire cost.

Log space again, but for a nicer reason than before. Under max there is no
log-sum-exp at all - log turns the products into sums and commutes with max
directly, so the recursion becomes pure additions and maxima. Viterbi in log
space is not an approximation of anything, it is the same arithmetic on a
different scale, and it is faster than the forward pass because of it.
"""

import numpy as np

from day1_forward import (
    check_hmm_params,
    forward_log,
    make_toy_model,
    sample_sequence,
)
from day2_backward import (
    backward_log,
    posterior_decode,
    posterior_marginals,
)

NEG_INF = -np.inf


def viterbi_log(pi, a, b, obs):
    """Most likely state path in log space, with backpointers.

    Returns (path, log_prob) where log_prob is the joint log P(x, z*) of that
    single best path - NOT the sequence likelihood, which sums over all paths
    and is therefore always larger. The gap between the two is a useful measure
    of how concentrated the posterior over paths is: if Viterbi's joint is close
    to the forward log-likelihood, one path carries most of the mass.

    np.errstate is needed because zero transitions are the interesting case and
    log(0) = -inf is the correct value here, not a warning. -inf propagates
    through the additions and loses every max against a finite score, which is
    precisely the behaviour wanted - a forbidden edge is never extended.
    """
    pi = np.asarray(pi, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    obs = np.asarray(obs, dtype=int)
    n_states, _ = check_hmm_params(pi, a, b)

    with np.errstate(divide="ignore"):
        log_pi, log_a, log_b = np.log(pi), np.log(a), np.log(b)

    n_steps = len(obs)
    delta = np.full((n_steps, n_states), NEG_INF)
    psi = np.zeros((n_steps, n_states), dtype=int)

    delta[0] = log_pi + log_b[:, obs[0]]

    for t in range(1, n_steps):
        # (K_prev, K_next): row i is the score of arriving at j through i.
        # the emission at t is constant down each column so it can be added
        # after the max instead of inside it - same result, K^2 fewer adds.
        scores = delta[t - 1][:, None] + log_a
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = scores[psi[t], np.arange(n_states)] + log_b[:, obs[t]]

    best_last = int(np.argmax(delta[-1]))
    best_log_prob = float(delta[-1, best_last])

    if not np.isfinite(best_log_prob):
        # every path through this observation sequence has probability zero,
        # which means the model cannot have generated it at all. worth being
        # loud about rather than returning a backtrace through garbage.
        raise ValueError("no path with nonzero probability generates this obs")

    # walk the winners backwards. psi[t, j] is who we came from to reach j at t,
    # so the loop reads the pointer at t and writes the state at t-1.
    path = np.empty(n_steps, dtype=int)
    path[-1] = best_last
    for t in range(n_steps - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]

    return path, best_log_prob


def path_log_probability(pi, a, b, obs, path):
    """Joint log P(x_1..x_T, z_1..z_T) for one specific state path.

    Scores a path the model already committed to, which is what makes it the
    referee between the two decodes: run it on Viterbi's output and on the
    posterior decode's output and compare. Returns -inf when the path uses a
    zero-probability transition, and that is the headline result of the day.
    """
    pi = np.asarray(pi, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    obs = np.asarray(obs, dtype=int)
    path = np.asarray(path, dtype=int)

    with np.errstate(divide="ignore"):
        total = np.log(pi[path[0]]) + np.log(b[path[0], obs[0]])
        for t in range(1, len(obs)):
            total += np.log(a[path[t - 1], path[t]])
            total += np.log(b[path[t], obs[t]])

    return float(total)


def viterbi_brute_force(pi, a, b, obs):
    """Enumerate all K^T paths and keep the best. Ground truth for tiny T.

    Same role as brute_force_log_likelihood on day 1 - it is the check that the
    recursion computes the argmax over paths rather than something that merely
    scores well. Deliberately shares no code with viterbi_log beyond the
    scoring helper, so a shared bug cannot hide in both.
    """
    from itertools import product

    n_states = np.asarray(pi).shape[0]
    best_path, best_score = None, NEG_INF

    for candidate in product(range(n_states), repeat=len(obs)):
        score = path_log_probability(pi, a, b, obs, candidate)
        if score > best_score:
            best_path, best_score = np.array(candidate), score

    return best_path, best_score


def make_forbidden_transition_model():
    """Three-state chain in which state 0 can never follow state 2.

    Built specifically to separate the two decodes, and the construction is the
    interesting part. States 0 and 2 are near-duplicates in emission - rows that
    differ by two hundredths - so no single observation tells them apart and all
    of the information separating them lives in A. That is the regime where the
    per-step decode is weakest, because it is the one thing gamma summarizes
    away: gamma[t] and gamma[t+1] each know about the transitions, but nothing
    downstream of them knows the two argmaxes have to be compatible.

    So the marginals genuinely do put their mass on 2 at one step and on 0 at
    the next, and the posterior decode emits the 2 -> 0 edge that A forbids.
    """
    pi = np.array([0.35, 0.50, 0.15])
    a = np.array([[0.40, 0.40, 0.20],
                  [0.70, 0.05, 0.25],
                  [0.00, 0.40, 0.60]])   # the forbidden edge is a[2, 0]
    b = np.array([[0.50, 0.08, 0.42],
                  [0.52, 0.22, 0.26],
                  [0.50, 0.06, 0.44]])   # rows 0 and 2 all but identical
    return pi, a, b


def decode_comparison(pi, a, b, obs):
    """Run both decoders on the same sequence and score each under the model.

    Returns a dict rather than a tuple because there are five things worth
    looking at and positional unpacking of five items is unreadable at the call
    site. The two joint scores are the comparison that matters; Viterbi's is
    the maximum by definition, so the only question is by how much and whether
    the posterior path is even admissible.
    """
    log_alpha, _ = forward_log(pi, a, b, obs)
    log_beta = backward_log(pi, a, b, obs)
    gamma = posterior_marginals(log_alpha, log_beta)

    viterbi_path, viterbi_score = viterbi_log(pi, a, b, obs)
    marginal_path = posterior_decode(gamma)

    return {
        "gamma": gamma,
        "viterbi_path": viterbi_path,
        "viterbi_score": viterbi_score,
        "marginal_path": marginal_path,
        "marginal_score": path_log_probability(pi, a, b, obs, marginal_path),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(3)

    # part 1: correctness. the recursion against exhaustive enumeration on a
    # sequence short enough to enumerate (3^8 = 6561 paths).
    pi, a, b = make_toy_model()
    _, short_obs = sample_sequence(pi, a, b, 8, rng=rng)

    fast_path, fast_score = viterbi_log(pi, a, b, short_obs)
    slow_path, slow_score = viterbi_brute_force(pi, a, b, short_obs)

    print(f"observations      : {short_obs}")
    print(f"viterbi (dp)      : {fast_path}  logP = {fast_score:.10f}")
    print(f"viterbi (brute)   : {slow_path}  logP = {slow_score:.10f}")
    assert np.array_equal(fast_path, slow_path)
    assert np.isclose(fast_score, slow_score)

    # the backtrace has to actually score what delta claimed it scores. this
    # catches an off-by-one in the psi walk that a matching argmax would not.
    rescored = path_log_probability(pi, a, b, short_obs, fast_path)
    print(f"backtrace rescored: {rescored:.10f}")
    assert np.isclose(rescored, fast_score)

    # part 2: the best single path against the sum over all paths. viterbi's
    # joint is a lower bound on the likelihood, and the gap says how much mass
    # lives off the best path.
    _, total_ll = forward_log(pi, a, b, short_obs)
    print(f"\nforward log-lik (all paths) : {total_ll:.6f}")
    print(f"viterbi  log-prob (one path): {fast_score:.6f}")
    print(f"best path holds {np.exp(fast_score - total_ll):.2%} of the mass")

    # part 3: the two decodes on a longer sequence from the same toy model.
    # here they mostly agree, which is the boring and typical case.
    truth, obs = sample_sequence(pi, a, b, 400, rng=rng)
    cmp_toy = decode_comparison(pi, a, b, obs)
    agree = (cmp_toy["viterbi_path"] == cmp_toy["marginal_path"]).mean()
    print(f"\ntoy model, T={len(obs)}")
    print(f"  viterbi  accuracy vs truth : "
          f"{(cmp_toy['viterbi_path'] == truth).mean():.4f}")
    print(f"  marginal accuracy vs truth : "
          f"{(cmp_toy['marginal_path'] == truth).mean():.4f}")
    print(f"  the two decodes agree on {agree:.2%} of steps")
    print(f"  joint logP  viterbi={cmp_toy['viterbi_score']:.4f}  "
          f"marginal={cmp_toy['marginal_score']:.4f}")

    # the marginal decode should win on per-step accuracy and lose on joint
    # probability. that is the objective mismatch stated as two numbers, and it
    # is the reason "which decode is better" has no answer without a question.
    assert cmp_toy["viterbi_score"] >= cmp_toy["marginal_score"]

    # part 4: the failure mode. a model with a forbidden transition, searched
    # for a sequence on which the posterior decode actually walks off the edge.
    pi_f, a_f, b_f = make_forbidden_transition_model()
    for attempt in range(200):
        _, obs_f = sample_sequence(pi_f, a_f, b_f, 60,
                                   rng=np.random.default_rng(attempt))
        cmp_f = decode_comparison(pi_f, a_f, b_f, obs_f)
        if not np.isfinite(cmp_f["marginal_score"]):
            break
    else:
        raise RuntimeError("no seed produced an impossible marginal path")

    bad = cmp_f["marginal_path"]
    edge = next(t for t in range(1, len(bad))
                if a_f[bad[t - 1], bad[t]] == 0.0)
    print(f"\nforbidden-transition model, seed {attempt}, T={len(obs_f)}")
    print(f"  marginal decode uses {bad[edge - 1]} -> {bad[edge]} at t={edge},"
          f" which A gives probability {a_f[bad[edge - 1], bad[edge]]}")
    print(f"  joint logP of the marginal path : {cmp_f['marginal_score']}")
    print(f"  joint logP of the viterbi  path : {cmp_f['viterbi_score']:.6f}")

    # and at that step both choices really were the individually-best ones -
    # the marginal decode is not making a mistake on its own terms.
    print(f"  gamma[t-1] = {np.round(cmp_f['gamma'][edge - 1], 4)}")
    print(f"  gamma[t]   = {np.round(cmp_f['gamma'][edge], 4)}")
    assert np.isfinite(cmp_f["viterbi_score"])
    print("\nviterbi path is admissible; the per-step decode is not.")
