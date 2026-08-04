"""
Unit tests for Viterbi decoding. Small set, aimed at the failures that survive
eyeballing - a backtrace that is off by one still returns a plausible path of
the right length, and a max taken over the wrong axis still returns a number.
"""

from __future__ import annotations

import numpy as np
import pytest

from day1_forward import forward_log, make_toy_model, sample_sequence
from day3_viterbi import (
    make_forbidden_transition_model,
    path_log_probability,
    viterbi_brute_force,
    viterbi_log,
)


def _toy_sequence(n_steps: int = 8, seed: int = 0):
    pi, a, b = make_toy_model()
    _, obs = sample_sequence(pi, a, b, n_steps, rng=np.random.default_rng(seed))
    return pi, a, b, obs


@pytest.mark.parametrize("seed", range(6))
def test_matches_brute_force(seed):
    """The recursion must find the same path enumeration does."""
    pi, a, b, obs = _toy_sequence(8, seed)
    path, score = viterbi_log(pi, a, b, obs)
    best_path, best_score = viterbi_brute_force(pi, a, b, obs)
    assert np.array_equal(path, best_path)
    assert np.isclose(score, best_score)


def test_backtrace_scores_what_delta_claimed():
    """Rescoring the returned path must reproduce the returned log-prob.

    Separate from the brute-force check on purpose: an off-by-one in the psi
    walk can still produce the correct argmax path while the reported score
    came from a different cell, and only one of these two tests sees that.
    """
    pi, a, b, obs = _toy_sequence(20, seed=1)
    path, score = viterbi_log(pi, a, b, obs)
    assert np.isclose(path_log_probability(pi, a, b, obs, path), score)


def test_single_step_is_just_the_argmax():
    """T=1 collapses to argmax_i pi[i] B[i, x_0], with no transitions at all."""
    pi, a, b = make_toy_model()
    for symbol in range(b.shape[1]):
        path, score = viterbi_log(pi, a, b, [symbol])
        expected = int(np.argmax(pi * b[:, symbol]))
        assert path.tolist() == [expected]
        assert np.isclose(score, np.log(pi[expected] * b[expected, symbol]))


def test_best_path_never_beats_the_full_likelihood():
    """P(x, z*) <= P(x), since the likelihood sums over every path."""
    pi, a, b, obs = _toy_sequence(50, seed=2)
    _, score = viterbi_log(pi, a, b, obs)
    _, log_likelihood = forward_log(pi, a, b, obs)
    assert score <= log_likelihood + 1e-9


def test_deterministic_chain_recovers_the_true_path():
    """With near-deterministic emissions the decode must return the truth.

    The one case where the answer is knowable without a reference
    implementation, which is what makes it worth a test of its own.
    """
    pi = np.array([1.0, 0.0])
    a = np.array([[0.5, 0.5],
                  [0.5, 0.5]])
    b = np.array([[0.999, 0.001],
                  [0.001, 0.999]])
    states, obs = sample_sequence(pi, a, b, 40, rng=np.random.default_rng(4))
    path, _ = viterbi_log(pi, a, b, obs)
    assert np.array_equal(path, states)


def test_never_emits_a_forbidden_transition():
    """Zero transitions must be unreachable however the sequence comes out.

    The point of the whole day: the per-step posterior decode can walk across
    a zero-probability edge, and Viterbi structurally cannot.
    """
    pi, a, b = make_forbidden_transition_model()
    for seed in range(20):
        _, obs = sample_sequence(pi, a, b, 60, rng=np.random.default_rng(seed))
        path, score = viterbi_log(pi, a, b, obs)
        assert np.isfinite(score)
        assert all(a[i, j] > 0 for i, j in zip(path[:-1], path[1:]))


def test_impossible_observation_raises():
    """A symbol no state can emit means no path exists - say so, do not decode.

    Without the check the backtrace would run over a delta row of all -inf and
    return argmax's arbitrary index 0, which looks like an answer.
    """
    pi = np.array([0.5, 0.5])
    a = np.array([[0.5, 0.5],
                  [0.5, 0.5]])
    b = np.array([[0.6, 0.4, 0.0],
                  [0.3, 0.7, 0.0]])   # symbol 2 is emitted by nobody
    with pytest.raises(ValueError):
        viterbi_log(pi, a, b, [0, 1, 2])
