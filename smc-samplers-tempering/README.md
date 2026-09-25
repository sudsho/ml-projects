# Sequential Monte Carlo Samplers: Tempering with Resampling

Where the AIS project breaks. Every AIS run carried its own weight from q to p,
and on the funnel the log of the mean weight stopped about 0.09 nats low at
every T from 10 to 1000, because no run reached the neck and a mean of weights
cannot put back mass that was never visited. SMC samplers resample the cloud
between temperatures and choose the next temperature from the effective sample
size, which is the particle filter project's machinery pointed at a path
instead of at time. Built from scratch in NumPy on the same closed-form targets,
so every estimate of `log Z` is still scored against an exact one.

## Idea

AIS here is the SMC sampler with resampling switched off. Both share the path,
the kernel and every line of the sampler except the resample, so any difference
between them is the resample and nothing else. The question the project was
picked to answer is whether resampling moves particles into the region AIS did
not reach, or only piles copies onto the runs that were already doing well.

On the Gaussian targets of days 1 to 3 the per-step second moment of each
incremental weight is a closed form, so the first-order bias of both estimators
is known before the sampler runs, and the N = inf limit of the adaptive rule can
be built and frozen. Day 4 is the funnel, with closed forms only at the ends.

**Resampling moves the cloud, but only as far as the kernel can carry the
copies. On a Gaussian that is all the way. On the funnel it is about a quarter
of the plateau AIS left, and the adaptive rule and the cloud-shaped kernel each
cost more than they bring.**

## Layout

- `day1_tempered_smc.py` - reweight, resample, move on AIS day 1's AR(1)
  target, with AIS as the same function at `resample = 'never'`, the
  first-order bias of both as closed forms, and ESS-triggered resampling.
- `day2_adaptive_tempering.py` - the next beta by bisection on the ESS of the
  increments, the temperature count against `L / sqrt(log 1/rho)`, and the
  adaptive schedule against its own N = inf limit frozen.
- `day3_move_step.py` - random-walk Metropolis and MALA shaped by the resampled
  cloud, against day 1's MALA, with every resampling's parent indices kept.
- `day4_funnel_smc.py` - Neal's funnel from VI day 4's reverse-KL q with AIS
  day 4's HMC move, SMC against AIS at matched moves per particle, the cloud's
  share of the neck, and an HMC mass matrix read off the cloud.

## What the days actually show

**Day 1: the resample fixes the kernel, not the estimator.** With exact moves
the product of means is off by `sum (r_t - 1) / 2N` and AIS's mean of products
by `(prod r_t - 1) / 2N`, 0.0021 against 0.0025 at T = 100 from the CAVI q, and
the simulation cannot separate them. With one MALA step per temperature at
N = 1000, AIS is 0.37 low at T = 100 and SMC 0.19, and the weighted final cloud's
long-axis variance is 1.94 against 2.75 of p's 6.20, from 358 and 420 surviving
starting ancestors. Resampling only when the ESS falls below N/2 gives the same
bias at up to 4x the spread. One right, one half, three wrong.

**Day 2: the adaptivity is most of the bias.** From N(0, I) the rule's limit
temperature count is `L / sqrt(log 1/rho)` to within a temperature, and from the
CAVI q it is more, because `beta* = 1.019` caps the last step at 0.019 whatever
rho asks. At N = 100 and rho = 0.5 the adaptive sampler is 0.402 low and the
same rule's limit schedule, frozen, is 0.022 low and unbiased in Z. A cloud that
has not drawn the tail of w reports a higher ESS than the population has, steps
further, and then estimates that step's increment from the same cloud. Under
MALA the frozen schedule beats the adaptive one, 0.019 against 0.288 at rho = 0.9.
None right, two half, four wrong.

**Day 3: shape the kernel before adding moves.** At beta = 1 day 1's MALA steps
1/227 of the long-axis variance. One MALA move preconditioned by the cloud's
covariance is 0.027 low from the CAVI q, where day 1's MALA is 0.60 low at one
move and needs thirty to reach the target's width. Random-walk Metropolis at 27%
acceptance beats that MALA at 95%, and the high acceptance rate was the symptom
nothing on days 1 or 2 printed. Distinct starting ancestors depend on the kernel,
109 to 354 of 1000 for one against thirty random-walk moves, and whether more
moves still raise the count picks out the unmixed rows with no closed form
needed. Three right, two wrong.

**Day 4: on the funnel resampling moves the plateau by about a quarter.** At
rho = 0.9 with 20 HMC moves per temperature, 168 moves per particle, SMC is
0.068 +- 0.008 low, where AIS at T = 300 and 299 moves is 0.094 +- 0.006. The
best SMC cloud has 5.2% of its mass below v = -4.16 against p's 8.2% and AIS's
3.5%, so 63% of the neck where AIS had 43%. At a budget of 3 moves it is worse
than AIS at 9. The adaptive rule takes under ten temperatures, so half the
starting ancestors survive and there is no genealogy to read. The cloud-shaped
mass matrix, day 3's winner, diverges 75209 times in one row, because the
cloud's x variance is the mouth's. Two right, one half, two wrong.

## The funnel, beside the projects before it

`sigma_v = 3`, `d = 3`, `log Z = 0`, from VI day 4's reverse-KL q. Errors are
`estimate - log Z`.

| method | cost | error | P(v < -4.16) | what reports the failure |
|---|---|---|---|---|
| plain IS, 4000 draws (VI day 4) | 0 moves | -0.330 | - | k-hat 0.880 |
| AIS `log mean w`, T = 100 | 99 moves | -0.110 | - | divergences past beta 0.9 |
| AIS `log mean w`, T = 300 | 299 moves | -0.094 | 3.5% | divergences past beta 0.9 |
| SMC, rho = 0.5, 1 move | 3 moves | -0.264 | - | nothing separate |
| SMC, rho = 0.9, 5 moves | 41 moves | -0.090 | - | nothing separate |
| SMC, rho = 0.9, 20 moves | 168 moves | -0.068 | 5.2% | 1390 of 1461 divergences past beta 0.9 |
| SMC, cloud-shaped HMC, rho = 0.9, 20 moves | 168 moves | -0.112 | - | 75209 divergences |
| p | - | 0 | 8.2% | - |

A move is one HMC trajectory of ten leapfrog steps. The divergence count is
still the one number that is large where the sampler fails, and SMC's is 12x
AIS's per move, because more of its particles are standing in the neck when the
path opens it.

## Key design choices

**AIS as a flag, not a second sampler.** `tempered(..., resample='never')`
telescopes to AIS's `log mean w` exactly, so the comparison has no second code
path to differ in.

**Closed forms first.** On days 1 to 3 the exact-transition rows sit inside 2 se
of the first-order deltas before any MALA row is read, and the adaptive rule is
scored against its own N = inf limit rather than against another sample.

**Cost counted in moves per particle.** Each funnel row reports it, and AIS is
rerun at N = 2000 with eight repeats so its `log mean w` has a standard error,
which the AIS project's day 4 did not give it.

**Predictions written into the files before running them.** Across the four
days, six right, four half, eleven wrong, and the docstrings say which.

**Fixed seeds throughout.**

## Where it breaks

**One kernel shape for the whole cloud.** Day 3 found that a kernel shaped like
the cloud beats any number of moves of one that is not, and day 4 found the
opposite, because on the funnel no single shape fits: the neck needs steps about
400x smaller than the mouth. A metric that depends on where the particle is,
Riemannian manifold HMC with the funnel's Fisher information, or a
reparameterisation that removes the dependence, is the thing neither project
tried.

**Too few temperatures for the genealogy to mean anything.** The adaptive rule
reaches beta = 1 in under ten steps on the funnel, and day 3's ancestor count
diagnostic needs depth to work with.

**The adaptivity bias was never removed.** Day 2 measured it and day 4 used the
adaptive rule anyway. A pilot run to fix the schedule, then a fresh run on it,
is the standard fix and is not here.

**Small and closed-form.** An eight-dimensional Gaussian and a
three-dimensional funnel. No posterior from data.

## Running

```bash
python day1_tempered_smc.py
python day2_adaptive_tempering.py
python day3_move_step.py
python day4_funnel_smc.py
```

NumPy only. Every day imports AIS project files from
`../annealed-importance-sampling`, and days 2 to 4 import earlier days here.
About 3.5 minutes, 5, 5 and 30 seconds on this machine.
