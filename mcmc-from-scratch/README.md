# MCMC from Scratch: Metropolis-Hastings, HMC and the Diagnostics

Where the particle filter project points. Markov chain Monte Carlo built from
scratch in NumPy on targets whose answers are known in closed form - random-walk
Metropolis and the optimal-scaling result measured rather than cited, the cost of
conditioning against the cost of dimension, Hamiltonian Monte Carlo with the two
identities its correctness argument actually uses written as residuals, and then
the convergence diagnostics scored on whether each one detects the failure it is
sold for.

## Idea

The particle filter project ended with an unbiased likelihood estimator, which is
what pseudo-marginal MCMC consumes. That is only worth having if the MCMC half is
understood on its own, so this project never runs on a posterior. Every target
here has closed-form moments, and where the state space can be made finite the
transition kernel is written out as a matrix, so the correctness properties are
residuals rather than arguments and the mixing numbers are eigenvalues rather
than estimates.

That discipline is the same one the grid filter bought on the previous project,
and it exists to support a question that turned out to be the actual subject:

**a sampler is used exactly when its answer cannot be checked, so everything
rests on the diagnostics - and the diagnostics are the part nobody measures.**

Every day here produced at least one quantity that is supposed to report how the
sampler is doing and does not. That was not the plan for days 1 through 3. It
became the plan for day 4.

## Layout

- `day1_metropolis.py` - targets with closed-form moments, random-walk
  Metropolis, detailed balance and stationarity as residuals on a 241-point
  discretised state space, the exact relaxation time from the kernel's second
  eigenvalue, and the 0.234 optimal-scaling result swept rather than quoted.
- `day2_mixing.py` - Geyer's IACT estimator checked against a closed-form IACT
  for the first time, dimension against conditioning, preconditioning and how
  wrong a pilot covariance is allowed to be, the independence sampler and its
  uniform-ergodicity threshold, and the banana and Student-t where the fixes stop.
- `day3_hmc.py` - leapfrog integration, volume preservation and reversibility as
  testable identities, the energy-error scaling and the `E[exp(-dH)] = 1` check,
  cost per effective sample against dimension, conditioning under a mass matrix,
  and trajectory-length tuning.
- `day4_diagnostics.py` - Neal's funnel, the non-centred reparameterisation,
  divergences, R-hat and split-R-hat, and three controls that pin down what R-hat
  is a test of.

## What the days actually show

**Day 1: the quantity you are told to tune by is the worse of the two rules.**
Sweeping the proposal scale `l/sqrt(d)`, the optimal `l*` is 2.30 for every
`d >= 2` on this grid - flat, immediately, against the asymptotic 2.38 - while
the acceptance rate at that optimum is still falling at `d = 50`: 0.417, 0.369,
0.301, 0.276, 0.261, 0.255 for `d` = 1, 2, 5, 10, 25, 50. So the number you set
converges at `d = 2` and the number you are told to set it by is 9% off at
`d = 50` and still moving. The peak is also very flat - at `d = 50` every
acceptance rate from 0.153 to 0.320 is within 10% of peak ESS - so the three
decimals in "0.234" describe a limit, not a tolerance.

Two things came out of the same day that did not survive into the plan. Making
the proposal width state-dependent breaks the symmetry assumption and nothing
else, and the result is a chain whose mean is right (-0.0048 against 0) and whose
variance is 16% high: the bias is even in `x`, so a sampler checked on its mean
passes while sampling the wrong distribution. And the first version of the
efficiency estimator reported a **completely stuck chain as the most efficient
setting on the grid**, which is the first appearance of the thing day 4 is about.

**Day 2: conditioning costs more than dimension, and preconditioning is exact
rather than approximate.** Isotropic RWM runs at IACT 4.4, 7.9, 17.7, 32.8, 84.1,
179.2 for `d` = 1 to 50, linear in `d`. Two correlated parameters at condition
number 100 run at 311 - worse than fifty independent ones. Setting the proposal
covariance proportional to `Sigma` gives IACT 8.0 at kappa 19, 100 and 400 alike,
which is the isotropic `d = 2` number to the digit: affine invariance visible
rather than argued.

The part expected to fail did not. A pilot covariance estimated from the badly
mixing chain the preconditioner is meant to fix, up to 9.8% wrong, gives IACT
7.7, 8.1, 7.8 - indistinguishable from the exact matrix. A preconditioner needs
the orientation and the scale and does not need an accurate covariance, and those
had been assumed to be the same requirement.

The diagnostic failure that day is the independence sampler's uniform-ergodicity
threshold, which is binary where the performance it describes is continuous, and
the Student-t, where the three proposals rank by acceptance rate in exactly the
reverse of how they rank by correctness.

**Day 3: HMC's correctness properties are exact and its tuning is not.** The
leapfrog Jacobian determinant reads 1.000000000091 at `eps = 0.1` and
0.999999992267 at `eps = 2.0`, where the energy error is already 51.1 and every
proposal is being rejected - so the integrator being wrong about the physics does
not make the sampler wrong about the distribution. Energy error falls as `eps^2`
with measured ratios 3.47, 4.02, 4.01, 4.00.

Cost per effective sample: HMC 8.0, 8.0, 14.1, 23.0, 38.5, 58.0 against RWM's
4.4, 7.9, 17.7, 32.8, 84.1, 179.2. HMC *loses* below `d = 5`, because its floor
is `L` gradients per sample and RWM's is 1. On conditioning it pays the same
growth RWM pays at a 2.7x smaller constant, and the thing that fixes it is day 2's
preconditioner wearing a mass matrix.

The day's finding is trajectory length. On `N(0, I_10)` at `eps = 0.2`: IACT 1.00
at `L = 16`, **988.9 at `L = 31`**, 1.00 again at 48. The free Hamiltonian
trajectory on a Gaussian is a rotation of period `2 pi`, and `31 * 0.2 = 6.20`
against `2 pi = 6.283`, so the sampler is proposing the point it started from -
and accepting it 0.999 of the time, the highest acceptance rate in the sweep.

**Day 4: the marginal moment cannot see a missing tail, and R-hat is a test of
agreement.** On Neal's funnel, `Var(v)` comes back 8.49, 8.01, 8.30, 9.45, 6.54
against a true 9.00 across step sizes - four of five within 8%. The same chains
have `min v` of -7.48, -6.01, -5.07, -3.97, -2.71, and `v` has standard deviation
3, so the best of them reaches 2.5 sigma and the worst reaches 0.9. The neck is
not being visited at all. A marginal variance averages over the bulk, and the
bulk is fine.

The non-centred reparameterisation - `x_i = xt_i exp(v/2)`, which turns the
target into a product of independent Gaussians - takes `min v` to -12.4 at the
same step size on the same sampler, with **zero divergences at every step size**
and IACT down to 1.00. Nothing about the sampler changed; a person supplied the
coordinates.

R-hat on four dispersed chains at `eps = 0.10` reads 1.0041, split 1.0045, both
comfortably under the conventional 1.01 threshold, on chains with 42 divergences
that never get below `v = -6.14`. Three controls say why:

| chains | R-hat | |
|---|---|---|
| frozen at four different points | `inf` | caught, but by `W = 0` |
| four chains agreeing about `N(5,1)` when the target is `N(0,1)` | 1.0000 | not caught |
| frozen at the same point | 0.9999 | not caught |

That last value is `sqrt((n-1)/n) = 0.9998750` to seven places. `W` comes out
1.2e-32 instead of 0 only because the two-pass variance leaves rounding behind,
so the statistic is 0/0 and what gets printed is a function of the chain length
and of nothing else.

## The one diagnostic that works, and what it has that the others do not

Sweeping `sigma_v`, which sets how deep the funnel's neck goes, with the sampler
and step size held fixed:

| `sigma_v` | true `Var(v)` | measured | ratio | divergences |
|---|---|---|---|---|
| 1.0 | 1.00 | 1.02 | 1.017 | 0 |
| 2.0 | 4.00 | 3.79 | 0.948 | 21 |
| 3.0 | 9.00 | 6.55 | 0.728 | 113 |
| 4.0 | 16.00 | 12.24 | 0.765 | 4204 |

The divergence count is monotone over three orders of magnitude. The variance
ratio is wrong and *not* monotone, so it cannot even be used to rank the four
problems by difficulty.

The reason is structural rather than a property of this target. The marginal
variance, the acceptance rate and R-hat are all computed from the draws that were
collected, and a region the chain never enters leaves no trace in them. A
divergence is a trajectory the integrator lost, and it is generated in the region
being missed. It is the only diagnostic in this project that reports on where the
sampler did **not** go.

Which is also the answer to the thing that recurred all four days. A stuck chain
scores best on ESS (day 1). Acceptance rate climbs while performance collapses
(day 2), and peaks at the worst trajectory length in the sweep (day 3). Four
chains that agree about the wrong distribution pass R-hat (day 4). None of these
is a bug in the estimator. They are all the same fact: a statistic of the samples
you have cannot report on the samples you did not get.

## The estimators, twice, and both times the estimator was the problem

Day 3's sharpest available check is `E[exp(-dH)] = 1`, an exact identity over
proposals. It reads 1.0000 at every stable step size, then 0.9398 at `eps = 1.8`
and 0.0000 at `eps = 2.1`, where nothing about the identity has changed:
`exp(-dH)` is dominated by rare proposals with large negative `dH` that a mean of
4000 draws does not contain. The estimator is biased low by an amount that grows
exactly as the integrator degrades, so the sharpest check fails in the regime it
exists to detect, and it fails toward "looks fine".

Day 4 has the same shape from the other side. `Var(x_i) = exp(4.5) = 90.02` comes
back 69.9, 70.3, 61.8, 61.1, 135.4 from the *correct* non-centred sampler.
`E[x^2] = E[exp(v)]` is a heavy-tailed expectation estimated by a mean, and 18k
samples do not contain its tail. That one is not the sampler's fault, and a
project that had only run the centred version would have blamed it.

## Key design choices

**Closed-form targets only.** No posteriors, no data. Every error figure is
against an exact number, which is what makes "the diagnostic is wrong" a
statement that can be made at all.

**Finite state spaces where possible.** Discretising a Gaussian onto 241 points
turns detailed balance into `max |pi_i P_ij - pi_j P_ji|` (1.7e-18), stationarity
into `|pi P - pi|` (3.5e-18), and the mixing time into an eigenvalue. Days 1 and
2 use it to check the two things everything else depends on: the correctness
argument, and Geyer's IACT estimator, which lands within 6% of exact across a
range spanning IACT 10 to 1632 and errs *low* at worst - the direction that
overstates ESS.

**Identities as residuals.** Volume preservation and reversibility are exact
statements that hold at any step size, so they are printed as numbers rather than
cited. At `eps = 5.0` both residuals blow up together, and the write-up says so:
the trajectory has reached magnitude 1e9, the finite-difference Jacobian is
measuring its own instrument, and neither identity has stopped holding.

**Comparisons at matched ESS, not matched chain length.** Day 3's banana
comparison was initially read backwards by comparing covariance error at equal
chain length, which credits the cheaper-per-sample sampler for a longer run.

**Fixed seeds throughout.** Every number in every docstring is reproducible by
running the file.

## Where it breaks

**The reparameterisation is not a method.** Day 4's fix is a change of variables
a person worked out for this target. Nothing here automates it, and on a model
where the geometry is not a known analytic funnel there is no equivalent to
reach for. The honest summary of day 4 is that the diagnostic told the truth and
the fix came from outside the sampler.

**Divergence counting is HMC-only.** The one diagnostic that works here has no
analogue for random-walk Metropolis, which fails on the funnel just as badly and
silently. Every scoring statement about divergences is conditional on using a
gradient-based sampler.

**One target for the diagnostics.** R-hat is scored on the funnel and on three
synthetic controls. That is enough to show what R-hat is a test *of*, and not
enough to say how often it fails in practice - a funnel is chosen precisely
because it is the standard adversarial case.

**No adaptation.** Step size and trajectory length are swept by hand everywhere.
NUTS, dual averaging and adaptive mass matrices would change day 3's tuning
conclusions and probably day 4's divergence counts, and none of them is here.

**`d = 3` funnel, one chain length.** Day 4's failures are measured at 20k steps
in three dimensions. The direction of every conclusion is stable across the step
sizes swept, but no claim is made about how the numbers scale.

## Running

```bash
python day1_metropolis.py
python day2_mixing.py
python day3_hmc.py
python day4_diagnostics.py
```

NumPy only - no SciPy, no PyTorch, nothing downloaded. Measured on this machine:
55s, 31s, 35s, 50s. Day 1 is the slowest because of the three-seed scaling sweep
over `d` = 1 to 50, and day 4's time is mostly the step-size and `sigma_v` sweeps.
