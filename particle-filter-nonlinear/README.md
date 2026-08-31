# Particle Filtering for Nonlinear State-Space Models

Where the Kalman project stops. Sequential Monte Carlo from scratch in NumPy on a
model whose observation is even in the state, so the filtering posterior is
genuinely bimodal and no Gaussian belief can represent it at all: the bootstrap
filter, weight degeneracy and the effective sample size, three resampling schemes,
the extended and unscented Kalman filters on the same tracks, backward-simulation
smoothing, and the likelihood estimator - every claim scored against an exact grid
filter rather than against the truth.

## Idea

The Kalman project's filter is exact. For a linear-Gaussian model the posterior
*is* Gaussian, the recursions carry its two sufficient statistics without
discarding anything, and nothing that project measured was ever about the belief
being the wrong shape, because there was no shape to be wrong about. Drop
linearity or Gaussianity and the posterior stops belonging to any parametric
family - it is a function, and the only exact representation of it is the
function.

The EKF's answer is to linearize and keep the Gaussian anyway. That answers a
question about *representation* with a change to the *model*, and when the true
posterior has two separated modes no linearization repairs it, because the failure
is not that the mean is in the wrong place. The failure is that a mean is the
wrong object.

The model is the univariate nonlinear growth benchmark:

    x_t = 0.5 x_{t-1} + 25 x_{t-1} / (1 + x_{t-1}^2) + 8 cos(1.2 t) + v_t
    y_t = x_t^2 / 20 + w_t

with `v_t ~ N(0, 10)`, `w_t ~ N(0, 1)`. The observation is **even in the state**,
so it constrains `|x_t|` and says nothing about the sign. Only the dynamics can
break the tie, and near zero they barely do. This is not a hard nonlinearity
picked to embarrass a Gaussian - it is a structural ambiguity, and a unimodal
belief cannot represent it however well it is fitted.

The one design decision the whole project rests on: **build the exact filter
first.** The state is one-dimensional, so the true recursion can be integrated on
a fine grid to a precision set by the mesh rather than by a seed. That does not
generalize past two or three dimensions - which is the entire reason particle
filters exist - but it means every claim here about what the posterior looks like
is measured rather than asserted, and every claim about the particle filter's
error separates into approximation error and irreducible posterior spread.

## Layout

- `day1_bootstrap_filter.py` - the model, the grid filter as the instrument,
  sequential importance sampling as the base case, and the bootstrap filter with
  multinomial resampling. Bimodality measured rather than assumed, and the
  posterior mean interrogated as a summary of a two-mode density.
- `day2_resampling.py` - multinomial vs stratified vs systematic, offspring-count
  variance on a single step against RMSE over a hundred of them, the
  particle-ordering effect, adaptive resampling by ESS threshold, and what ESS
  does and does not measure.
- `day3_gaussian_filters.py` - EKF and UKF on the same tracks, every Gaussian
  moment checked against the closed form (`h` is a quadratic, so there is one),
  and a linear-transition control to locate where the damage actually is.
- `day4_smoothing_and_likelihood.py` - forward-backward smoothing on the grid,
  FFBSi backward simulation, the likelihood estimator and its unbiasedness, and a
  linear-Gaussian control where "exact" needs no qualifier.

## What the days actually show

**Day 1: the posterior mean lands where the posterior says nothing is, and is
still the better point estimate.** Bimodal at 33 of 100 steps, and the two modes
are near reflections (median `|x_lo + x_hi| / |x|` of 0.029), so it is the sign
ambiguity and not some other structure wearing its clothes. At those steps the
reported mean sits at 0.255 of the peak density on average and at 2.7e-20 of it at
the worst step, against 0.968 on the unimodal steps. The obvious conclusion is
that the mean is the wrong summary. It is not: RMSE 3.859 against the MAP's 4.520,
and the margin is *widest* on exactly the bimodal steps, 6.274 vs 7.514. Squared
error is minimized by the mean whether or not the mean is a plausible state, and
"the summary is in a place the density excludes" and "the summary is bad" are
different sentences.

SIS without resampling reaches `ESS < 2` at `t = 3` with 200 particles and at
`t = 3` with 2000. A tenfold increase buys nothing, which is the sharpest form the
argument for resampling takes.

**Day 2: three of the four things I expected about resampling are wrong.** On a
single degenerate step the summed offspring-count variances are 152.7 /
18.2 / 13.7 for multinomial / stratified / systematic, so 8.4x and 11.2x
reductions. Then:

- the particle-ordering effect is *invisible* to that statistic, provably:
  `Var(count_i) = frac(N w_i)(1 - frac(N w_i))` with no other weight in it, so the
  ordering lives entirely in the covariances and a sum of marginal variances is
  exactly the statistic that cannot see them. Measure `Var(sum count_i x_i / N)`
  instead and it is 104x between best and worst ordering for stratified;
- that 104x is worth nothing in the filter. Sorting particles before resampling
  moves RMSE by `-0.032 +/- 0.028`, against a between-scheme gap of 0.876 -> 0.782.
  A hundred steps of a mixing transition destroy the correlation the sort creates;
- adaptive resampling is worse at every threshold, monotonically (0.782 always,
  0.840 at `N/2`, 1.409 at `N/10`). With `R = 1` against `Q = 10` the weights
  degenerate within a couple of steps, and by the time ESS has fallen the damage
  is in the particle *locations*, which resampling can redistribute but not undo;
- low-variance schemes preserve *more* distinct particles, not fewer (87.3 vs
  70.7 of 200). Survival needs `count_i >= 1` at fixed `E[count_i] = N w_i`, so
  lowering variance moves mass off `{2, 3, ...}` onto `{0, 1}` - it suppresses
  duplication, and duplication is what costs diversity.

ESS correlates `-0.208` with absolute error overall, which looks like it working.
Split by bimodality it is `+0.067` on the 67 unimodal steps and `-0.410` on the 33
bimodal ones - the overall figure is entirely a between-group effect. ESS is a
diagnostic for the weight vector, which is what it was defined to be.

**Day 3: the even observation is not what hurts the Gaussian filters.** The story
the model was built for is that `dh/dx = x/10` vanishes at the origin so the gain
collapses. Three things killed it. The gain collapse is *correct* - `Cov(x, h(x))`
for a centred Gaussian and any even `h` is exactly zero, and both filters compute
`m P / 10` to 1e-15, so there is nothing to linearize better. The errors are flat
across the axis the story needs (EKF RMSE 19.6 bimodal, 21.0 unimodal). And
swapping `f` for a linear one takes the EKF from 12.4x the exact filter to 1.01x
*within each model's unimodal steps*, where the posterior shape is held fixed. The
damage is the Jensen gap in the prediction: `Q = 10` keeps the belief at `sd ~ 3.2`
and `f`'s rational term turns over at `|x| ~ 1`, so `f(E[x]) != E[f(x)]` at every
step.

The UKF's `beta` closes it out. On this `h` the transform is *exact* at `beta = 0`
and doubles the `P^2` term at the standard `beta = 2` - and the exact one is
worse, RMSE 13.54 against 10.67. The inflated `S` was shrinking the gain and
compensating for a prediction bias the transform never modelled. Two errors partly
cancelling, and removing the removable one leaves the other bare.

**Day 4: smoothing is worth 66% here and the reason is not the one I would have
given.** Backward-pass RMSE 1.315 against the filter's 3.859, concentrated exactly
where day 1 said the difficulty was - 6.274 -> 1.560 on the bimodal steps against
1.687 -> 1.175 on the rest. Two-thirds of the filter's error is not irreducible,
just early: the backward pass sees the sign because the next few states are not
sign-symmetric once the forcing moves.

The obvious explanation - "the filter is very uncertain, so smoothing has room" -
is wrong, and the control says so. Weaken the linear model's observation gain and
its filter variance rises 12x with linearity and unimodality left alone:

| model | filter var | smoother var | smoothing gain |
|---|---|---|---|
| linear, `c=1.00` | 0.911 | 0.893 | 0.3% |
| linear, `c=0.50` | 2.914 | 2.769 | 0.7% |
| linear, `c=0.20` | 8.118 | 7.621 | 0.7% |
| linear, `c=0.10` | 11.368 | 11.016 | 1.3% |
| nonlinear | 21.652 | 2.206 | **65.9%** |

Twelve-fold more filter variance buys one point of smoothing gain. The gain is not
a function of how uncertain the filter is - posterior *width* is the wrong
variable. What matters is the kind of uncertainty: a Gaussian filter's leftover is
a continuous spread the future shrinks by a small gain, while this model's is a
discrete question (which branch) that one look at the future answers outright.

Two smaller results that change how the earlier days should be read. The smoothed
posteriors are bimodal at **more** steps than the filtered ones, 36 against 33,
while being 4x more accurate and 10x tighter - so the mode count is a shape
statistic and has never been measuring ambiguity, which is what every previous
day's tables used it for. And backward simulation's `M = 800` RMSE of 0.123
against the exact smoother is *not* better than the filter's 0.323 against the
exact filter: divided by each posterior's own sd the order reverses, 0.083 vs
0.069. The smoothed target is tighter, so there is less to get wrong before any
method is involved.

## The likelihood estimator is unbiased and that is nearly useless

`Zhat = prod_t (1/N) sum_j p(y_t | x_j^t)`, against the grid filter's exact
`log Z = -67.428` over 25 steps, 400 independent runs each:

| `N` | mean `Zhat/Z` | sd `Zhat/Z` | mean `log Zhat - log Z` |
|---|---|---|---|
| 100 | 0.996 +/- 0.100 | 2.010 | -3.501 |
| 400 | 1.012 +/- 0.038 | 0.768 | -0.250 |
| 1600 | 1.005 +/- 0.019 | 0.374 | -0.058 |

Unbiasedness holds at every `N` with no `N`-dependence to it, exactly as
advertised - and at `N = 100` the estimator's standard deviation is twice the
quantity it is estimating, so a single run carries essentially no information and
only the average over 400 of them is worth anything. The property is about the
ensemble and says nothing about the run you have, which is the whole reason it is
enough for pseudo-marginal MCMC and not enough for reading a number off one fit.

`log Zhat` is biased *down* by 3.5 nats there - Jensen, since the raw estimator is
unbiased and `log` is concave - falling off roughly like `1/N` (3.501, 0.250,
0.058 across 16x). Everything that consumes a likelihood consumes the log of it,
so everything downstream is reading a downward-biased number whose bias depends on
a tuning parameter.

## The degeneracy check comes out backwards

The reason to pay `O(M N T)` for backward simulation instead of tracing resampling
ancestry is that ancestral paths collapse in the *past* - every trajectory shares
one ancestor a few dozen steps back, and the early smoothed marginals sit on one
or two distinct values at any `N`. These do not: 160 distinct values at step 0,
median 390 across steps, out of `M = 800`.

The thin step is the **last** one, 56 of 800. The cause is in the filter rather
than the smoother: ESS at step `T-1` is 29.6 of 2000 against a median of 587, and
`x_T` is drawn straight from those weights with no backward reweighting to flatten
them. So the method fixes the degeneracy it is advertised against and inherits a
different one at the single step where it is doing nothing.

`M` also buys much less than it should - 50, 200, 800 trajectories give 0.192,
0.197, 0.123, neither `M^-1/2` nor reliably monotone. It cannot be: the backward
pass only re-weights the cloud the forward pass produced, so the error floor is
set by `N`. Spending on trajectories while the filter is the bottleneck buys noise.

## The linear control, where "exact" needs no qualifier

Everything above is scored against a grid filter, exact only up to a quadrature
and a truncation window. On a linear-Gaussian model the Kalman filter *is* the
posterior, so the whole gap is Monte Carlo error:

| `N` | RMSE vs Kalman mean | ratio to previous | 95% coverage |
|---|---|---|---|
| 50 | 0.2474 | | 0.922 |
| 200 | 0.1031 | 2.40 | 0.969 |
| 800 | 0.0504 | 2.05 | 0.972 |
| 3200 | 0.0249 | 2.02 | 0.971 |
| Kalman | - | | 0.970 |

Textbook `N^-1/2` once `N` is past 200, and coverage converges onto the exact
filter's own 0.970 from below - at `N = 50` the intervals undercover because
particle quantiles cannot resolve a 2.5% tail out of 50 points. The nonlinear
model's exact filter covers 0.950 on the nose, so nothing above is a calibration
problem in the model.

## Key design choices

**The grid filter is written twice.** Day 1's returns the filtering posteriors and
nothing else. Day 4 needs the one-step predictive densities (the smoothing
recursion divides by them) and the normalising constants (their product is the
likelihood), and neither is recoverable after the fact, so the recursion is
written out again and checked against day 1's - they agree to 5.6e-16.

**The likelihood constant matters in exactly one place.** Day 1 drops the
`1/sqrt(2 pi R)` factor because a constant divides out of a normalised posterior.
It does not divide out of the evidence. Leaving it off would shift `log Z` by
`T * 0.919` and silently break every comparison in day 4.

**The smoother masks rather than floors.** In the tails `predicted` is numerically
zero where `smoothed` is zero too, and `0/0` has to become `0`. Flooring the
denominator instead biases the tails upward by manufacturing a ratio where there
is no density.

**Filtering clouds are stored pre-resampling.** The `(particles, weights)` pair
before resampling is the filtering approximation; the post-resampling cloud is
consistent but strictly noisier, since resampling is a bootstrap sample of the
weighted average it replaces.

## Where it breaks

**One dimension, and that is load-bearing.** The grid filter is `O(T n^2)` in 1D
and `O(T n^2d)` in `d`. Every "measured rather than asserted" claim here rests on
an instrument that does not exist for the problems particle filters are actually
for.

**One trajectory, one seed, for the structural claims.** The resampling
comparisons are paired over 40 seeds and the likelihood study over 400 runs, but
the smoothing gain, the mode counts and the coverage figures are one 100-step
track at `seed=7`. The 65.9% has no error bar on it.

**The bootstrap proposal only.** Every filter here proposes from the prior. The
locally-optimal proposal and auxiliary particle filters would change the weight
degeneracy that days 1 and 2 are largely about, and nothing here tests them.

**The linear control is not a clean isolation.** Swapping `f` changes the Jensen
gap *and* collapses the bimodal steps, because the odd rational term is what kept
the prior nearly sign-symmetric. Day 3 works around it by restricting to each
model's unimodal steps; day 4's `c` sweep is the cleaner version, varying one
thing, and it is the only reason the smoothing conclusion is stated as strongly as
it is.

## Running

```bash
python day1_bootstrap_filter.py
python day2_resampling.py
python day3_gaussian_filters.py
python day4_smoothing_and_likelihood.py
```

NumPy only - no SciPy, no PyTorch. Days 1 and 3 take a few seconds each, day 2
about a minute for the 40-seed paired comparisons, day 4 about two minutes, most
of it the 1200 filter runs behind the likelihood table.
