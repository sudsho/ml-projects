# Hidden Markov Model with Forward-Backward and Viterbi

A discrete hidden Markov model written from scratch in NumPy. The forward and
backward recursions in log space, the smoothed posteriors that come from pairing
them, Viterbi decoding with backpointers, and Baum-Welch EM to learn the model
from sequences whose states were never observed.

## Idea

An HMM is two coupled sequences. The one you want, `z_1..z_T`, is a Markov chain
over `K` discrete states. The one you observe, `x_1..x_T`, is emitted
independently from whichever state is active. The entire model is three matrices:

    pi[i]    = P(z_1 = i)                 initial distribution
    A[i, j]  = P(z_t = j | z_{t-1} = i)   transitions
    B[i, o]  = P(x_t = o | z_t = i)       emissions

The quantity everything is built on is `P(x_1..x_T)`. Written directly it is a
sum over `K^T` state paths — already `10^30` for `K=3, T=63`. The forward
recursion computes the same sum in `O(T K^2)` because the paths interact only
through which state they occupy at time `t`, so they can be merged there. That
single observation is reused three more times in the project: with `max` instead
of `sum` it becomes Viterbi, run backwards it becomes `beta`, and paired with
`alpha` it becomes the expected counts that EM needs.

## Layout

- `day1_forward.py` — the three matrices and their validation, ancestral
  sampling, the forward recursion in probability space and in log space, and
  brute-force path enumeration as ground truth. Includes the demonstration that
  the naive version underflows to exactly `0.0` somewhere past `T = 400` while
  the log-space per-step likelihood stays flat.
- `day2_backward.py` — the backward recursion, the smoothed posterior `gamma`,
  the filtered posterior for contrast, and the likelihood recomputed at every
  `t` as a consistency check on both passes.
- `day3_viterbi.py` — Viterbi in log space with backpointers, path scoring,
  brute-force decoding for verification, and a model with a forbidden transition
  that makes the posterior decode emit a zero-probability path.
- `day4_baum_welch.py` — the pairwise posterior `xi`, the E and M steps, the EM
  loop with its monotonicity assertion, label-permutation alignment, and the
  restart study.
- `test_viterbi.py` — the properties that must hold: agreement with brute force,
  the decoded path having non-zero probability, backtrace correctness.

## Key design choices

- **Log space everywhere, with `-inf` allowed.** `alpha` decays by roughly one
  emission probability per step and underflows float64 in a few hundred steps.
  Zeros in `A` and `B` are genuine structure (a forbidden transition), so
  `log(0) = -inf` is permitted to flow through rather than being nudged to
  epsilon, and `logsumexp` handles the all-`-inf` row explicitly because
  `-inf - -inf` is `nan`.
- **`alpha` is a joint, `beta` is a conditional.** That asymmetry is why they
  multiply cleanly into `P(x, z_t = i)`, and it is why `beta`'s recursion
  indexes the emission at `t+1` — `x_t` belongs to `alpha`'s half, and including
  it in both would double-count it.
- **Row-stochasticity is asserted, not trusted.** A row summing to 0.999 gives
  likelihoods wrong by a factor that grows exponentially in `T` and looks like a
  subtle bug for a long time.
- **`sum` and `max` are the same algorithm.** Viterbi is the forward recursion
  over a different semiring. What it needs extra is a backpointer, because
  summing does not care which predecessor contributed and maximizing has a
  single winner per cell.
- **The M step normalizes `A` and `B` by different denominators.** `A`'s is the
  expected transitions out of a state (`T-1` steps); `B`'s is the expected visits
  (`T` steps). Swapping them is an error that shrinks like `1/T`, so it looks
  like a convergence problem on short sequences and like nothing at all on long
  ones.

## Results

Two-state weather model (`rainy`/`sunny`, both sticky) over a three-symbol
alphabet, trained on 20 sequences of length 100 — 2000 observations total.
Log-likelihood under the true parameters: `-2182.64`.

| Fit | Iterations | Final log-lik | max \|A - Â\| | max \|B - B̂\| |
|---|---|---|---|---|
| EM from a random start | 250 | **-2181.40** | 0.152 | 0.186 |
| EM seeded at the truth | 40 | -2181.57 | **0.012** | **0.030** |

Five restarts on identical data:

| Seed | Final log-lik | max \|B - B̂\| |
|---|---|---|
| 0 | **-2181.33** | 0.174 |
| 1 | -2181.64 | 0.202 |
| 2 | -2183.61 | 0.096 |
| 3 | -2181.61 | **0.068** |
| 4 | -2181.57 | 0.194 |

The likelihood increased at every EM iteration in every run, which is asserted
rather than reported — a decrease is not slow convergence, it is proof of a bug
in the E or M step.

## The thing worth taking away

The two tables disagree, and the disagreement is the result.

Seeding EM at the true parameters recovers them to within `0.03` in 40
iterations, so 2000 observations identify this model comfortably and the data is
not the limitation. Yet the random start reached a *higher* likelihood with
parameters five times further off — and across restarts, the best-likelihood run
(`-2181.33`) has a `B` error of `0.174` while the most accurate run (`0.068`)
scores lower. Ranking restarts by likelihood, which is the only criterion
available when the truth is unknown, does not reliably pick the most accurate
model.

This is not a local-optimum trap; those runs are not stuck in visibly worse
basins. It is a flat ridge — a visibly different `(A, B)` explains this sample
about as well, so the likelihood barely distinguishes them and EM has no reason
to prefer the true one. EM optimizes likelihood. It was never optimizing
parameter recovery, and on a ridge the two come apart.

Which matters entirely for what the fitted model is being used for. As a density
— scoring sequences, filling in missing observations, detecting anomalies — any
of these runs is fine and the likelihood ranking is the right one. As an
*explanation*, where `A[0,1] = 0.19` versus `0.30` is a claim about the world,
the learned matrices are much softer than a converged optimizer suggests. Label
switching is the trivial version of the same point (the likelihood is exactly
invariant to permuting the states, so recovery is only ever defined up to a
relabeling), and the ridge is the non-trivial version: a direction in which the
objective is nearly invariant, and about which the fit therefore says very little.

## Running

```bash
python day1_forward.py
python day2_backward.py
python day3_viterbi.py
python day4_baum_welch.py
pytest test_viterbi.py
```

Requires NumPy. `day4_baum_welch.py` takes a few minutes — the recursions are
explicit Python loops over `t` for readability, not vectorized across time.
