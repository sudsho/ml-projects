# Kalman Filter and RTS Smoother from Scratch

The linear-Gaussian state-space model built from scratch in NumPy: the Kalman
predict/update recursions, the numerical failure modes that only show up over
thousands of steps, the RTS backward smoother, EM for the noise covariances,
filtering through blackouts, and a final experiment on whether the consistency
checks used throughout are valid tests at all.

## Idea

A 2-d target is tracked by position alone. State is `(x, y, vx, vy)`, the
dynamics are exact Newtonian motion over a step of `dt`, and `H` observes the
first two coordinates. Velocity is never measured, which is the entire exercise -
a filter that saw the whole state would have nothing to do.

The model is the HMM with the discrete state replaced by a continuous one:

    x_t = F x_{t-1} + w_t       w_t ~ N(0, Q)
    y_t = H x_t     + v_t       v_t ~ N(0, R)

and the algorithms map across one for one. The forward recursion is
forward-backward's alpha pass with the sum over states replaced by an integral;
the sequence likelihood is the same product of per-step normalizers; EM is
Baum-Welch with expected counts replaced by expected second moments. Everything
stays Gaussian because a Gaussian pushed through a linear map is Gaussian and a
Gaussian conditioned on a linear observation is Gaussian, so two matrices carry
the entire belief and there is nothing to approximate.

Written against the HMM project deliberately, because the correspondence is
exact until it suddenly is not, and the places it breaks are where the content
is. The clearest is the backward pass. In the HMM, `beta_t` is computed as its
own independent sweep and combined with `alpha_t` at the end. That is impossible
here: `beta_t` is a likelihood rather than a distribution, and `H` observes
position only, so `p(y_{t+1..T} | x_t)` has infinite variance along the
directions the future says nothing about. Over K discrete states a likelihood and
a distribution are both just K numbers and the distinction costs nothing; over a
continuous state one of them has no `(mean, covariance)` representation at all.
RTS avoids the problem entirely by recursing on the *smoothed* quantities
backward, so every object in it is an honest distribution.

## Layout

- `day1_kalman_filter.py` — the constant-velocity model with its correlated `Q`,
  simulation from the model itself, `predict`/`update` with the innovation
  covariance written out explicitly, the sequence log-likelihood by the
  prediction-error decomposition, and the first NIS check.
- `day2_numerical_stability.py` — Joseph-form covariance update, symmetry
  projection, what a filter divergence actually looks like over 4000 steps, and
  the diffuse-prior and stiff-`R` experiments that produce one.
- `day3_rts_smoother.py` — the backward recursion and its gain, the lag-one
  smoothed covariance, an `O(T^3)` dense conditioning of the full joint Gaussian
  as an independent check, and where the smoother's improvement actually lands.
- `day4_em_and_diagnostics.py` — EM for `Q` and `R` in free and structured
  parameterizations, filtering through blackouts, NEES/NIS consistency, and the
  validity of the consistency test itself.

## What the days actually show

**Day 1: the correlation in `Q` is not a detail.** `Q` for a constant-velocity
model is the covariance of one random acceleration held across the interval, so
position and velocity inherit the *same* unknown and the matrix has `dt^2/2`
off-diagonal corners. The usual diagonal shortcut tells the filter that a
velocity error and the position error it necessarily causes are independent. The
filter still runs; it just trusts its own position estimate more than it has any
right to. Over 400 steps the filter takes position RMSE from 1.70 raw to 0.73 and
recovers unobserved velocity from a diffuse prior to 0.65 by the end.

**Day 2: the short covariance update dies on the first step, not slowly.** The
textbook `P - K S K'` is a subtraction that can leave the positive-definite cone,
and the expected story is a slow rot over thousands of steps. It is not what
happens. A diffuse prior against a precise sensor breaks it immediately, on step
one, where the ratio of prior uncertainty to measurement noise is at its largest
- and every subsequent step is *safer*, because `Q` adds a full-rank floor to the
eigenvalues at every predict. The danger is a transient at the start of the run.
The Joseph form, which is a sum of two positive-semidefinite terms rather than a
difference, is unbothered, and it is additionally valid for suboptimal gains
where the short form is not merely fragile but wrong.

**Day 3: smoothing helps least where you can already see.** Smoothed beats
filtered by construction, so the distribution of the gain is the content: 40.5%
on position, 48.0% on velocity. Position is observed directly at every step and
the future has little to add; velocity is inferred only through the correlation
`Q` builds, so knowing where the target subsequently went is a far more direct
statement about how fast it was moving. The improvement is also concentrated at
the start of the sequence and vanishes at the end, where smoothed equals filtered
by construction because there is no future left to condition on. Day 2 closed by
worrying that the backward pass inverts a predicted covariance every step and
that the diffuse prior would make that badly conditioned; it does not — a diffuse
prior is diffuse in every direction, so it is the *best*-conditioned predicted
covariance in the run. What actually moves the condition number is `H` being
partial, and the worst point is a transient around step 10.

**Day 4: more parameters, better fit, much worse estimate.** From a start with
`Q` inflated 20x and `R` deflated 10x, EM recovers both. The unconstrained
M-step reaches a higher likelihood than the one-scalar version — it must, since
it maximizes over a superset — and the margin is 1.9 nats over 1000 steps. What
it pays for that margin:

    fit          log-lik    iters   Q eigenvalue ratios vs truth
    free       -3370.07      140    0.86, 1.06, 20.01, 20.41
    structured -3371.99      245    1.15, 1.15, 1.15, 1.15

`Q` here has condition number 1206, and the free estimator inflates its two
smallest eigendirections by a factor of twenty. Those are exactly the directions
the likelihood is flat in, which is why the error is nearly free in nats and why
nothing in the fit pushes back on it. `R`, estimated from residuals against
directly observed coordinates, is recovered well by both.

## The consistency checks are not equally usable

This is the part of day 4 that changed how the rest of the project reads.

NEES (`e' P^-1 e` against ground truth, nominal 4) and NIS (`v' S^-1 v` on
innovations, nominal 2) both ask whether the reported covariance is honest. NEES
is the direct question and needs truth; NIS only ever looks through `H` and needs
nothing. The standard summary is that NIS is the weaker but practical one.

It is weaker in what it sees and *stronger* in how it may be used. Over 150 runs
of 200 steps:

    diagnostic   time-avg   sd observed / naive   naive 95% interval covers
    NEES           4.0052      0.679 / 0.200            36.7%
    NIS            2.0094      0.149 / 0.141            94.0%

Both time-averages are unbiased. The NEES interval is wrong by a factor of 3.4 in
width, because the chi-square interval assumes the averaged values are
independent and consecutive estimation errors at `dt = 0.1` are nearly the same
vector — 200 time samples carry the information of about 17. Innovations are a
martingale difference sequence, hence white at every lag, so for NIS the same
move is exactly legitimate.

Day 1 time-averaged NIS and was right to. Doing the same to NEES produces a
number that looks fine and a test that fails 63% of the time. The fix is
Bar-Shalom's original form — average across independent runs at a fixed time —
which is what day 4 does everywhere it asserts on NEES.

## Blackouts

Dropping three contiguous stretches of observations, rather than scattering
missing steps at random, since sensors fail in stretches and a filter at
`dt = 0.1` barely notices an isolated dropout. A missing step is a deletion
rather than a special case: the update does not happen, the filtered belief *is*
the predicted one, and nothing is imputed. Substituting the predicted observation
would leave the mean alone and shrink the covariance, claiming information from a
measurement that does not exist.

Position-block `trace(P)` at the start, middle and end of each gap:

    gap length   filtered                  smoothed
        30       0.56 →  4.77 → 16.42      0.31 → 0.63 → 0.31
        40       0.56 →  7.86 → 31.56      0.35 → 1.00 → 0.35
        60       0.56 → 17.65 → 85.20      0.40 → 2.19 → 0.40

The filter extrapolates, so its uncertainty can only grow, all the way to the far
edge. The smoother interpolates between two known ends, so its uncertainty is a
symmetric arch peaking in the middle. At the last missing step of the 60-step gap
the two disagree by 214x about how much is known, having seen identical data.

Sampled at gap midpoints across 80 independent runs, NEES is 4.41 against a
nominal 4. The covariance inside a blackout is a pure prediction corrected by
nothing, and it is still honest — which is the cleanest form this check takes
anywhere in the project.

## Key design choices

- **The lag-one smoothed covariance is built on day 3 and used on day 4.** EM's
  M-step needs `E[x_t x_{t-1}']` and the smoothed marginals cannot supply it —
  marginals are the diagonal blocks of the joint posterior over the trajectory
  and this is the first off-diagonal. Dropping the term and using
  `m_t^s m_{t-1}^s'` is the standard bug, and it is silent: EM still converges,
  still monotonically, to the wrong answer. The monotonicity assertion is what
  catches it, which is the only reason that assertion is worth writing.
- **Day 3 checks the recursion against an `O(T^3)` dense solve.** A
  linear-Gaussian state-space model is one big joint Gaussian, so the smoothed
  posterior is a textbook conditional. It shares nothing with the RTS recursion —
  no backward pass, no gain, not even the forward filter — so agreement to `1e-14`
  is evidence about the recursion rather than a restatement of it. Hopeless past
  a few dozen steps, which is exactly why it is trustworthy as a reference.
- **EM stops on a relative tolerance, not a fixed iteration count.** The first
  version of day 4 used 80 iterations for both fits and concluded the structured
  parameterization was the worse estimator. It was still 3x from its own optimum;
  the two parameterizations need 140 and 245 iterations respectively. Fixing the
  count compared a converged fit against an unconverged one.
- **Covariance error is measured as `max |log lambda|` of `Q_true^-1 Q_hat`.**
  `Q`'s entries span four orders of magnitude, so a Frobenius comparison reports
  only the velocity block. Two directions wrong by 20x are invisible in
  `||Q_hat - Q||_F` when the truth there is `1e-4`, and they are the whole story.
- **Chi-square intervals via Wilson-Hilferty rather than scipy.** The cube root is
  the variance-stabilizing transform for the family, so the approximation
  improves with the degrees of freedom rather than degrading, and the project
  keeps its single dependency.

## Where it breaks

Everything here assumes the model is linear and the noise is Gaussian, and both
assumptions are load-bearing rather than convenient. Linearity is what keeps the
belief Gaussian through `predict`; Gaussianity is what makes two matrices a
sufficient summary. Drop either and the posterior stops having a finite
parameterization, which is where the EKF, UKF and particle filters start — all of
them approximations to a distribution this project represents exactly.

The narrower gap is that `F` and `H` are assumed known throughout, including in
day 4's EM. Estimating `F` jointly is a small change to the M-step and a large
change to the problem: `F` and `Q` trade off against each other, the likelihood
develops a ridge rather than a peak, and the label-switching-style
non-identifiability the HMM project ran into has a continuous analogue here.

And EM finds a local maximum. Day 4 starts it from a single deliberately bad
point and it recovers, which is evidence about this likelihood surface and not a
general guarantee.

## Running

```bash
python day1_kalman_filter.py
python day2_numerical_stability.py
python day3_rts_smoother.py
python day4_em_and_diagnostics.py
```

Requires NumPy and nothing else. Days 1-3 run in seconds. Day 4 takes about three
minutes — the two EM fits run 385 iterations of forward-backward between them,
and the validity experiment simulates 230 independent tracks.

Every file asserts its own claims at the bottom rather than printing numbers for
inspection, so a silent exit is the pass condition.
