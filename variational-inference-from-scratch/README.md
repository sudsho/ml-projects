# Variational Inference from Scratch: CAVI, Black-Box Gradients and the ELBO's Blind Spot

The other way to do the integral the MCMC project spent four days sampling.
Mean-field CAVI and black-box VI built from scratch in NumPy, on targets whose
normalisers are known in closed form - the ELBO as a bound with an exactly
computable gap, reverse KL measured against forward KL on the same mixture, the
score-function and reparameterisation gradients with their variances printed,
Pareto-smoothed importance sampling with k-hat scored against a tail shape that
can be written down, and then Neal's funnel, the target the MCMC project's
diagnostics were scored on, so the two projects' failure reports sit in one
table.

## Idea

The MCMC project ended on one sentence: a statistic of the samples you have
cannot report on the samples you did not get. The ELBO is an expectation under
`q`, so it has that shape by construction, and VI was picked as the next project
because it should fail in the same place for the same reason.

That is only a checkable claim if the thing being missed is known. So no target
here is a posterior from data. Every one has a closed-form `log Z`, which turns
`KL(q || p) = log Z - ELBO` into a number, every fitted `q` is scored against its
closed-form optimum before anything is read off it, and every diagnostic is
scored against the quantity it claims to estimate.

**The ELBO and k-hat are both computed from draws of q, and the four days are
about what each of them can say about where q is not.**

## Layout

- `day1_cavi.py` - conjugate Gaussian targets, CAVI and its fixed point, the
  ELBO gap identified with KL by two separate derivations, the mean-field
  variance underestimate under AR(1) and equicorrelated structure, and reverse
  against forward KL on a two-component mixture.
- `day2_blackbox_gradients.py` - the score-function and pathwise gradients of the
  same ELBO checked against the exact gradient, their variances against
  dimension, Rao-Blackwellisation and a control variate, and what the trace shows.
- `day3_fullrank_iw_khat.py` - full-rank against mean-field, PSIS from scratch
  with k-hat scored against `k = 1 - m`, two q's with identical KL and opposite
  tails, and the importance-weighted bound's rate.
- `day4_funnel.py` - Neal's funnel with the reverse-KL Gaussian in closed form,
  the trace's level and its jitter, k-hat on three q's that share an exact shape
  of 1, and a `sigma_v` sweep beside MCMC day 4's divergence counts.

## What the days actually show

**Day 1: the mean-field underestimate is exact, and it is a property of the
precision's shape.** The ELBO gap matches the closed-form KL to 2.2e-16, and the
fixed point gets the mean exactly and the variance wrong by the
conditional-over-marginal ratio, 0.19 against 1.00 at `rho = 0.9`, which is
`1 - rho^2`. Both predictions I wrote down before the run were wrong in the same
place. I had the underestimate worsening with dimension for both structures.
Equicorrelated does, saturating at `1 - r` from above. AR(1) is exactly 0.342282
at every `d` from 4 to 64, because its precision is tridiagonal and an interior
coordinate's conditional variance only ever sees two neighbours.

On a two-component mixture the reverse-KL fit collapses onto the heavier
component, sd 0.711 against the component's 0.700, sitting on 65% of the mass and
12.1x worse on forward KL. Not a fit the ELBO failed to improve: the optimum of
the thing the ELBO is.

**Day 2: the two gradients differ only in variance, and the standard recipe for
reducing it made it worse.** Both estimators sit inside 1.8 standard errors of
the exact gradient. The reparameterisation gradient's advertised orders of
magnitude are 1.7x on the mean coordinate at `d = 4`, and the ratio is a
statement about dimension - 1.6x, 3.4x, 11.2x, 38.1x as `d` goes 2, 4, 8, 16.
Dropping the `log q_j` terms for `j != i` is unbiased and takes the per-sample
variance from 6.73 to 22.81, because `Var(B) = 16.29` against
`Cov(A, B) = 16.19`: the deleted term was working as a control variate.
Unbiasedness was the only property checked before deleting it. The ELBO trace
sees none of it - two runs finish 6.3e-03 apart and differ only in step-to-step
jitter, 1.0e-03 against 2.1e-04.

**Day 3: k-hat is a diagnostic of the tail as it looks at this sample size.**
Every q and target is Gaussian, so the weight tail's shape is `k = 1 - m` and
k-hat can be scored. Full-rank gets KL to 1.2e-02 against mean-field's 2.6156
and still has `k = 0.092` in one slightly narrow direction. At `d = 2` k-hat
separates two q's with identical KL the way their tails say, 0.481 against
-0.906. At `d = 32` the q with bounded weights reads 0.531 and falls to 0.412 as
`S` goes from 4000 to 64000 - it is reading the sample - and the heavy one reads
0.97 against an exact 0.51. The importance-weighted gap closes at slopes -0.973,
-0.666 and -0.189 against predicted -1, -0.667 and -0.111, and the control
variate I wrote to tighten it gives 1.158 against an exact 0.830 at `K = 1` once
`E[w^2]` diverges, so the closed form picks the estimator.

**Day 4: on the funnel the ELBO ranks the q's backwards and k-hat's estimand
cannot rank them at all.** A Gaussian q against the funnel still has an exact
ELBO, so the reverse-KL optimum is closed form: Var(v) 0.900 against 9.00,
Var(x) 0.638 against 90.02, and `KL = 0.5 log(1 + k sigma_v^2 / 2) = 1.151293`,
with the variance ratio exactly `exp(-2 KL)`. Full-rank buys nothing - max
|corr| 0.014 - because `Cov_p(x_i, v) = 0` exactly and the funnel's dependence is
not a correlation. Forcing one in costs `-0.5 log(1 - rho^2)` to five places.

The ELBO puts that q 8096 nats above the moment-matched Gaussian. As an
importance proposal the moment-matched q misses `log Z` by -0.044 against the
reverse-KL q's -0.330. k-hat reads 0.916 and 0.880 and cannot separate them, and
there is nothing for it to separate: for every Gaussian q in the centred
coordinates `E_q[w^a]` is infinite for all `a > 1`, so the exact shape is 1
across the whole family. Which of the two is heavier depends on the moment asked
about - 40% against 4.2% of q's mass sits where the conditional moment is
already infinite at `a = 2`, and 5.7e-06 against 1.2e-03 at `a = 1.01`.

## The table the two projects share

`sigma_v` sets how deep the funnel goes. HMC columns from MCMC day 4 at
`eps = 0.2`, VI columns with `q` at its closed-form optimum and k-hat over 40
seeds at `S = 4000`.

| `sigma_v` | HMC divergences | HMC `Var(v)` ratio | q `Var(v)` ratio | KL | k-hat | IS `log Z` |
|---|---|---|---|---|---|---|
| 1.0 | 0 | 1.017 | 0.500 | 0.347 | 0.662 +- 0.112 | -0.017 |
| 2.0 | 21 | 0.948 | 0.200 | 0.805 | 0.786 +- 0.117 | -0.163 |
| 3.0 | 113 | 0.728 | 0.100 | 1.151 | 0.800 +- 0.128 | -0.444 |
| 4.0 | 4204 | 0.765 | 0.059 | 1.417 | 0.861 +- 0.111 | -0.554 |

The chain's variance ratio is wrong and not monotone. q's is monotone and far
worse, and the ELBO trace behind it is flat. k-hat is monotone like the
divergence count, and for the same kind of reason: it is computed from the draws
at the edge of q, where p is larger than q, the way a divergence is produced
where the chain fails to go. But the divergence count spans three orders of
magnitude and k-hat spans 0.66 to 0.86 with a single-run sd of 0.11 to 0.13. An
edge of q is still inside q. What lies past it enters k-hat only through how fast
the weights grow toward it, and 4000 draws see that rate as it looks at 4000
draws - 0.808, 0.808, 0.840, 0.872 at `S` = 1000 to 64000, against an exact 1.

The prediction written into day 4 before the run was that k-hat would track
`k_v = 1 - s_v^2 / sigma_v^2`, day 3's formula along `v`: 0.500, 0.800, 0.900,
0.941. It sits above that at `sigma_v = 1` and below it from 3 on.

## What the trace does carry

Day 2 said the trace cannot report its own sampling noise. Day 4 corrects one
part of that. The trace's level cannot show the gap - a q that fits a target with
`log Z = -1.151` exactly prints the same level as this one. Its jitter can,
because an exact q makes `log p - log q` the constant `log Z` and every estimate
identical. On the funnel the per-draw sd is 1.260 against 0.011 for the nearly
exact non-centred fit. What the jitter lacks is a scale: if `log w` were
Gaussian, `E_q[w] = 1` would fix the gap at `Var(log w) / 2 = 0.794`. It is
1.151, and the difference is the weight tail.

## Key design choices

**Normalised targets only.** Every gap is `log Z - ELBO` against a closed-form
`log Z`, and on day 4 every importance estimate of `log Z` is its own error.

**Closed forms before fits.** Day 3's mean-field fit is scored against its
closed-form optimum (KL 2.6156 against 2.6103), and day 4's against a derived one
(ELBO -1.15135 against -1.15129), before either fit is used for anything.

**The closed form picks the estimator.** Day 3's control variate is better when
`E[w^2]` is finite and wrong when it is not, and the second-moment formula decides
which of the two gets printed.

**Predictions written into the files before running them.** Most were wrong, and
the docstrings say which and by how much rather than being rewritten to match.

**Fixed seeds throughout.** Every number in every docstring is reproducible by
running the file.

## Where it breaks

**Every gap here needs a known `log Z`.** The one number this project leans on,
`KL = log Z - ELBO`, is unavailable on any posterior with data, and the only
estimate of `log Z` built here comes from q itself: 0.33 nats low on the funnel
from the reverse-KL fit, 0.46 low after PSIS. Nothing in these four days
estimates the normaliser from outside q, and until something does the gap is a
quantity that exists only on a closed-form target.

**Gaussian q only.** On the funnel no Gaussian has a finite weight moment above
the first, so day 4's k-hat and importance-sampling results are about the family
as much as the objective. Mixtures and flows would move the tail and are not
here.

**The fix came from outside again.** Non-centred VI is exact - k-hat 0.046,
`log Z` to 2e-04 - and it is the MCMC project's reparameterisation, supplied by
someone who already knew the target was a funnel.

**k-hat at a fixed sample size.** Single-run sd 0.11 to 0.13, and every ranking
statement about it is over 20 to 40 seeds that nobody running VI on a real model
would have.

**No data, `d <= 64`.** Conjugate Gaussians, a mixture and a three-dimensional
funnel. Nothing here is a posterior from a likelihood.

## Running

```bash
python day1_cavi.py
python day2_blackbox_gradients.py
python day3_fullrank_iw_khat.py
python day4_funnel.py
```

NumPy only - no SciPy, no autodiff, nothing downloaded. Day 4 imports the PSIS
implementation from day 3. Measured on this machine: 11s, 2s, 31s, 4s, day 3's
time being almost all the `d = 32` sample-size sweep.
