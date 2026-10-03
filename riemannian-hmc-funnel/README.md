# Riemannian Manifold HMC: A Step Size That Depends on Where the Particle Is

Where the SMC project breaks. Its day 3 found that a kernel shaped like the
particle cloud beats any number of moves of one that is not, and its day 4
found the opposite on the funnel, 75209 divergences from a mass matrix read off
the cloud, because the neck needs steps about 400x smaller than the mouth and no
single shape fits both. The MCMC project's day 4 removed the dependence by
reparameterising, which only works when the target is known to be a funnel.
RMHMC puts the dependence in the kinetic energy instead: `H = U + log det G / 2
+ p' G^-1 p / 2` with a position-dependent metric `G(q)`, integrated by
Girolami and Calderhead's generalised leapfrog with its two implicit solves.
Built from scratch in NumPy on the same closed-form funnel, `sigma_v = 3`,
`d = 3`, and scored against the MCMC project's divergence counts and the share
of p's neck mass below `v = -4.16` that AIS and SMC reached, 43% and 63%.

## Idea

The question was whether a metric read off the local curvature reaches the
neck at a cost per step still worth paying. There are two metrics to read it
off. The Fisher metric, the Hessian of U averaged over x given v, is diagonal, a
function of v alone, and known in closed form, and its log determinant cancels
the `n v / 2` in U. The Hessian itself has the `x_i e^-v` cross terms and is
indefinite on most of the target, so it needs Betancourt's SoftAbs,
`lambda coth(alpha lambda)` on each eigenvalue, and the solves become genuinely
implicit. Day 1 measures what those solves cost on a metric made up for the
purpose, days 2 and 3 take each metric to the funnel, and day 4 puts the one
that survived inside the SMC project's tempered sampler.

**The Fisher metric reaches the neck at every step size with no divergences,
and inside SMC one move per temperature already fills the neck to p's share.
SoftAbs does not survive its own fixed-point solver. The cost is two gradients
a step and the rest of the 0.068 is the schedule, not the kernel.**

## Layout

- `day1_generalised_leapfrog.py` - the generalised leapfrog on VI day 3's
  AR(1) target with a diagonal metric `m_i (1 + c q_i^2)`, where `c = 0` has to
  reduce to plain leapfrog: iterations against tolerance, where the fixed point
  stops converging, and reversibility and `log |det J|` measured to rounding.
- `day2_fisher_funnel.py` - the funnel with its closed-form Fisher metric,
  against the MCMC project's centred HMC at matched gradients, and the same
  step solved by hand in two gradients.
- `day3_softabs_metric.py` - the SoftAbs metric from the Hessian, with
  `dH/dq` in closed form and checked against finite differences, alpha against
  the neck depth, and the log det term switched off.
- `day4_rmhmc_in_smc.py` - the Fisher step on the tempered path
  `q^(1 - beta) p^beta`, still explicit, as the move inside SMC day 4's
  sampler against its HMC move at matched gradients.

## What the days actually show

**Day 1: the energy error cannot see the tolerance, the roundtrip can.** At
`c = 0` the two integrators are 5.3e-14 apart after 100 steps and every solve
takes exactly 2 iterations, one to land and one to find it did not move, so
even the degenerate case costs two gradients per half step. At `c > 0` the
fixed point converges about 0.75 iterations a decade of tolerance at
`eps = 0.2` and fails at step sizes plain leapfrog still takes, 42 of 50
trajectories at its stability limit, with the converged steps carrying finite
`|dH|`. Mean `|dH|` is 1.3e-2 to 1.5e-2 whether tol is 1e-4 or 1e-12, while
the roundtrip misses the start by 4x to 7x tol, so the tolerance has to be set
from the roundtrip and not from the acceptance rate. `log |det J|` with K fixed
iterations falls from 1.1e-2 at K = 1 to the finite-difference floor by K = 8.
Three right, two half.

**Day 2: on the funnel there is nothing implicit to solve.** The Fisher metric's
two solves are triangular, 3 iterations each at every eps through day 1's
solver and one substitution by hand, 2 gradients a step and 2.2e-15 apart. min v
is -13.0 to -10.8 for eps 0.05 to 0.8 where centred HMC reaches -7.5 to -2.7,
divergences are 0 up to `eps = 0.4` against 14 to 2125, and over eight seeds
the neck share is 8.33% +- 0.19 against p's 8.28%. At 400k gradients the ESS in
v is 12x to 23x centred HMC's best, and 0.9x at `eps = 0.1` through the solver,
so the gain is all in the step size the metric allows. At `eps = 0.8` it
diverges 2959 times and 44% of those start in the neck, which the derivation
does not predict, and the rejections cost mixing and not the neck share. The
MCMC project's count had that as 5784, every infinite dH twice. Four right,
two half.

**Day 3: SoftAbs is a solver problem before it is a metric problem.** The
Hessian is indefinite on 89.1% of draws from p and equally so in the neck and
the mouth, since `e^-v |x|^2` is chi-squared whatever v is. The fixed point
stops converging and alpha sets where: 0 of 50 trajectories fail at
`alpha = 1` up to `eps = 0.2`, 47 of 50 at `alpha = 100` by `eps = 0.1`, and
it is the position solve that gives up while the converged steps carry `|dH|`
of 3e-5. Every divergence in the chains is a solver failure, so the best row,
`alpha = 10`, gives 86 effective samples per 100k gradients against the Fisher
metric's 427 at the same eps and centred HMC's 157. With the log det switched
off the Fisher chain targets `exp(-U) sqrt(det G)`, v ~ N(-9, 9), and gives
mean v -9.01 and 94.8% in the neck at the same acceptance, 0.994, and the same
IACT, 3.04, as the chain on the right target. No diagnostic in this project's
kit sees the difference. Four right, one half, two wrong.

**Day 4: one RMHMC move per temperature fills the neck.** The tempered target's
Fisher metric is `beta G_p(v) + (1 - beta) Prec_q`, diagonal because the q is
mean-field, so the step stays explicit and matches day 2's at `beta = 1` to
1.6e-15. Inside SMC at `rho = 0.9` the cloud's share below `v = -4.16` is
8.54%, 8.54% and 8.46% at 1, 5 and 20 moves per temperature against p's 8.25%,
where the HMC move gives 4.68% at 5 moves and 5.33% at 40, and `P(v < -7)` is
1.1% against p's 0.98% where HMC has none. Acceptance is 0.995 at every
temperature with no divergences at `eps = 0.2` and 0.4, and the neck fills in
the last tenth of the path, 2.7% at `beta = 0.90` to 8.4% at 0.994. `log Z` is
0.042, 0.036 and 0.033 low at 1, 5 and 20 moves and 0.020 at `eps = 0.4`, so
the moves past the first buy almost nothing and what is left of SMC's 0.068 is
the schedule. At `eps = 0.8` the sampler diverges 17888 times in a run and the
neck share is still 8.3%, day 2's finding again. The neck share costs 87
gradients per effective sample at one move against 445 for the best HMC row,
which has 3.2% of the neck. Four right, one half, two wrong.

## The funnel, beside the projects before it

`sigma_v = 3`, `d = 3`, `log Z = 0`, p has 8.25% of its mass below `v = -4.16`.
SMC rows are N = 2000 and 8 runs, `rho = 0.9`, cost in gradients per particle;
chain rows are from day 2 and the MCMC project at 400k gradients.

| method | gradients | log Z error | P(v < -4.16) | P(v < -7) | divergences |
|---|---|---|---|---|---|
| AIS, HMC 0.2, T = 300 (AIS day 4) | 2990 / particle | -0.060 +- 0.015 | 3.5% | - | 208 |
| SMC, HMC 0.2, 20 moves (SMC day 4) | 1750 / particle | -0.051 +- 0.030 | 6.0% | 0.00% | 2339 |
| SMC, HMC 0.2, 40 moves | 3400 / particle | -0.064 +- 0.014 | 5.3% | 0.00% | 3281 |
| SMC, RMHMC 0.2, 1 move | 172 / particle | -0.042 +- 0.018 | 8.5% | - | 0 |
| SMC, RMHMC 0.2, 20 moves | 3500 / particle | -0.033 +- 0.012 | 8.5% | 1.10% | 0 |
| SMC, RMHMC 0.4, 20 moves | 3600 / particle | -0.024 +- 0.011 | 8.2% | 1.00% | 0 |
| SMC, RMHMC 0.8, 20 moves | 3450 / particle | -0.039 +- 0.010 | 8.3% | - | 17888 |
| centred HMC chain, best eps (MCMC day 4) | 400k | - | - | - | 14 to 2125 |
| Fisher RMHMC chain, eps 0.4, L 20 (day 2) | 400k | - | 8.33% +- 0.19 | - | 0 |
| p | - | 0 | 8.25% | 0.98% | - |

The HMC rows are the SMC project's sampler rerun from this file's seed, which
is why the 20-move row differs from that project's -0.068. The divergence
count is the one column that is large where the HMC kernel fails and zero
where the metric is right, and at `eps = 0.8` it is large while the neck share
is not wrong, so it reports the integrator and not the target.

## Key design choices

**Reduce to the known case first.** Day 1 at `c = 0` has to be plain
leapfrog, day 2's step by hand has to match the solver, day 4's step at
`beta = 1` has to be day 2's. Each of those is checked to rounding before the
sampler is read.

**The roundtrip as the tolerance check.** Flip p and run back; the miss tracks
tol and the energy error does not.

**One sampler, two kernels.** Day 4 registers the RMHMC move in the SMC
project's kernel table and runs that project's `smc` unchanged, so the only
difference between the rows is the move.

**Cost in gradient calls.** Two per leapfrog step here, one for plain HMC, and
every row says which.

**Predictions written into the files before running them.** Across the four
days, fifteen right, six half, four wrong, and the docstrings say which.

**Fixed seeds throughout.**

## Where it breaks

**No diagnostic sees a wrong target.** Day 3's chain on `exp(-U) sqrt(det G)`
had the same acceptance and IACT as the chain on p, and the divergence count
was zero on both. Every diagnostic this project and the three before it used
is a statistic of the chain, and a chain that mixes well on the wrong density
passes all of them. Geweke's joint-distribution test and simulation-based
calibration are built to see exactly this and neither has been run here.
They are the next project, [sampler-correctness-sbc](../sampler-correctness-sbc/),
whose day 4 puts this chain to them.

**SoftAbs through a fixed-point solve.** Day 3 did not settle whether the fault
is the metric or the solver. A damped iteration or a Newton step on the
position solve is the untried thing, and until it is tried SoftAbs has no
measured place on this target.

**The metric is known because the target is.** The Fisher metric was derived
by hand from the funnel, and day 4's tempered metric from a mean-field q. A
posterior from data has neither in closed form.

**The schedule's bias is still there.** Day 4's rows are 0.02 to 0.04 low with
the neck at p's share, which is the SMC project's day 2 adaptivity bias, and a
frozen pilot schedule was not run.

**Small and closed-form.** A three-dimensional funnel and an eight-dimensional
Gaussian. No posterior from data.

## Running

```bash
python day1_generalised_leapfrog.py
python day2_fisher_funnel.py
python day3_softabs_metric.py
python day4_rmhmc_in_smc.py
```

NumPy only. Days 2 to 4 import the MCMC, AIS and SMC projects' files from
their folders, and days 3 and 4 import earlier days here. About 35 seconds,
65 seconds, 7.5 minutes and 2 minutes on this machine.
