# Getting It Right: Geweke's Joint Test and Simulation-Based Calibration

Where the RMHMC project breaks. Its day 3 switched off the log det term, the
chain moved to `v ~ N(-9, 9)` with 94.8% of its mass in the neck, and it did so
at the same acceptance, 0.994, and the same IACT, 3.04, as the chain on the
right target, with zero divergences on both. Every diagnostic in four projects
is a statistic of the chain, and a chain that mixes well on the wrong density
passes all of them. Geweke's joint-distribution test and simulation-based
calibration ask from the other side: draw the truth from the prior and data from
the model, run the sampler, and check that the moments agree or that the rank of
the truth among the draws is uniform. Built from scratch in NumPy, with the bugs
planted on purpose so that what each test should catch is known before it runs.

## Idea

Both tests use the same fact. If a sampler leaves `p(theta | y)` invariant,
then a truth drawn from the prior is one more draw from the posterior given the
data it generated. Geweke (2004) turns that into a second chain on the joint,
one sampler sweep given y then a fresh y given theta, whose moments must match
independent joint draws. SBC (Talts et al. 2018) holds each y still, runs the
sampler on it, and counts how many of L draws fall below the truth, which must
be uniform on `0..L`. Days 1 and 2 build each test on a conjugate normal model
with unknown mean and precision, where the Gibbs conditionals are closed form
and a bug is one wrong symbol. Day 3 plants the bugs in the samplers they
belong to, and day 4 takes SBC to the funnel.

**Every planted bug that moves the target is caught by one of the two tests,
but which one is cheaper depends on the bug, by up to a hundred times either
way. The funnel's log det bug, invisible to every chain diagnostic, is rejected
by SBC on every study at 20 simulations. Neither test can tell a correct sampler
that is slow from a wrong one, and a bug that kills the chain passes Geweke with
no rejections at all.**

## Layout

- `day1_geweke_joint_test.py` - the successive-conditional simulator against
  the marginal-conditional one on `mu ~ N(0, 1)`, `tau ~ Gamma(3, 2)`, five
  data points: the z-score with iid and batch-means standard errors, its size
  on a correct Gibbs sampler, and three one-symbol bugs and a slow sampler.
- `day2_sbc_ranks.py` - SBC on the same model, L = 99 in 20 bins, the
  chi-squared critical value simulated from exact uniform ranks, thinning, and
  100 independent studies per row so a rejection rate is a rate.
- `day3_planted_bugs.py` - the plan's bugs in their own samplers on the model
  in `(mu, log tau)`: HMC without its last half kick, RMHMC without its log
  det, a random walk without its Hastings factor, and the slow sampler, each
  put to both tests at matched cost against three correct samplers.
- `day4_funnel_sbc.py` - the RMHMC project's sampler with and without the log
  det, at `eps = 0.2` and 0.8, and centred HMC, through SBC on the funnel,
  all as arrays of chains.

## What the days actually show

**Day 1: the size depends on the standard error more than on the test.** The
joint chain is autocorrelated, lag 1 at 0.85 for mu, so an iid standard error
rejects a correct sampler 46% of the time on mu. Batch means with 50 batches
fix that at `M = 10000`, 4.5% to 7.0% per function, but not at 1000, where it
is 29% for any of five. The shape bug and the sd bug are caught on every
replication at `M = 1000`, the sd bug only by second moments since `E[mu]`
does not move. Dropping the prior's rate sends all 200 chains to inf by sweep
195, every z is nan, and a harness counting `|z| > 1.96` counts zero. A correct
sampler that is slow, lag 1 at 0.995, is rejected on 86% at `M = 1000` and
still 20% at 50000. Two right, one half, three wrong.

**Day 2: SBC had nothing to thin on the Gibbs chain.** Day 1's 0.85 belonged to
the joint chain, which draws a fresh y every sweep. With y held still the lag is
0.03 and the unthinned sampler rejects at 2% on mu and 5% for either. The
chi-squared critical value is 30.0 at both 50 and 1000 simulations against the
table's 30.1, so two and a half ranks a bin was not too few. The sd bug is a U,
3.05 in each end tenth, and is caught by 50 simulations. The shape bug is a
tilt from 0.62 to 1.43 and needs 500 simulations for 95%, 99,500 sweeps a study,
where day 1 caught it at 1000. Three right, one half, two wrong.

**Day 3: the bug decides which test is cheap.** Three correct samplers set the
size: Geweke 28% to 31% at `M = 1000` and 16% to 19% at 10000, SBC 3% to 10%
at 100 simulations. The missing log det moves `E[tau]` 8% at the right chain's
acceptance and Geweke catches it on every replication at 1000 transitions; SBC
sees the tilt and needs 500 simulations for 75%, 545,000 transitions a study.
The missing Hastings factor moves `E[tau]` 19% and both catch it. The dropped
half kick moves `E[tau]` under 1%, gets 24% from Geweke at 1000, inside the
size, and 17% from SBC at 500, and its histogram is the table's only hump: the
chain is too wide, not shifted. The slow sampler is all ends, 1.81 and 1.74, at
burn 100 or 2000, and only thin 100 brings it near the size. One right, two
half, three wrong.

**Day 4: SBC sees the funnel's bug at 20 simulations.** The chain diagnostics
still cannot tell it apart, acceptance 0.993 against 0.994 and IACT(v) 2.89
against 2.99. SBC rejects it on every study at 20 simulations, about 4,000
transitions, with 95% of v's ranks in the top tenth because the truth sits
above nearly every draw. Centred HMC diverges on 4.9% of proposals and SBC
agrees, 100% at 500, but its v histogram is a U and not the pile at the bottom
I predicted: chains start from prior draws, and one that starts in the neck
stays there as surely as one in the mouth stays out. The correct sampler needs
thin 5 to pass, 15% at thin 1 and 3% at thin 5, and at `eps = 0.8` it diverges
on 14.7% of proposals while SBC finds it close to correct. Three right, two
half.

## Which test caught which bug

Geweke rows are 100 replications, SBC rows 40 or 100 studies. A Geweke row at
`M = 1000` is read against the 28% to 31% the correct samplers get there.

| sampler | moves the target | Geweke, M = 1000 | Geweke, M = 10000 | SBC, 100 sims | SBC, 500 sims |
|---|---|---|---|---|---|
| correct samplers (day 3) | no | 28% to 31% | 16% to 19% | 3% to 10% | 3% to 7% |
| Gibbs, mu sd 1 / prec (day 1, 2) | spread of mu | 100% | - | 100% by 50 | 100% |
| Gibbs, tau shape n - 1 (day 1, 2) | `E[tau]` 0.29 sd | 100% | - | 52% at 200 | 95% |
| Gibbs, prior rate dropped (day 1) | chain overflows | 0%, every z nan | 0% | - | - |
| RMHMC, log det dropped (day 3) | `E[tau]` +8% | 100% | - | 23% | 75% |
| walk, Hastings factor dropped (day 3) | `E[tau]` -19% | 100% | - | 80% | 100% |
| HMC, last half kick dropped (day 3) | `E[tau]` -0.8% | 24% | 52% | - | 17% |
| slow random walk, correct (day 1, 3) | no | 86% | 44% | 88% at thin 10 | 100% at thin 10 |
| funnel RMHMC, log det dropped (day 4) | v to N(-9, 9) | not run | not run | 100% by 20 | 100% |
| funnel centred HMC (day 4) | no, but stuck | not run | not run | 90% | 100% |
| funnel RMHMC, correct, thin 5 (day 4) | no | not run | not run | - | 3% |

Day 2 did not run the shape bug at 100 simulations, so its entry there is the
200 it did run. Geweke was not run on the funnel because the funnel has no
data: the posterior is the prior, and the joint test's second chain would be the
sampler itself.

## Key design choices

**Plant the bug, then test.** Each bug is one symbol or one term, its effect on
the target is measured directly first, by `E[tau]` against quadrature at one y
or by the funnel's known neck share, and only then is the test asked.

**Calibrate the size on correct samplers, not on the table.** Geweke at
`M = 1000` rejects a correct sampler 30% of the time, and that row is what a
bug's row is read against.

**Simulate the critical value.** SBC's chi-squared cut is taken from exactly
uniform ranks at the same number of simulations, and it came out at the
table's value anyway.

**Rates, not single runs.** Every cell is the share of 40 to 100 independent
studies, and day 2 found a 100-study rate should not be read to better than 3
points either way.

**Predictions written into the files before running them.** Across the four
days, nine right, six half, eight wrong, and the docstrings say which.

**Fixed seeds throughout.**

## Where it breaks

**Slow is read as wrong.** The slow random walk leaves the posterior invariant
and fails both tests on nearly every study, and centred HMC on the funnel fails
SBC for the same reason, a chain that never crosses the truth in a run. Both
tests ask for independent draws and treat autocorrelation as error. Thinning
fixes it only once the thin is longer than the chain's memory, 100 for the slow
walk, and neither test reports whether it is looking at a bias or at a slow
chain. A rank statistic that charges a sampler for its effective sample size
and not for its raw draw count is the untried thing.

**The binned chi-squared is the weak statistic.** The shape bug took 500
simulations and the dropped half kick 17% at 500, and both had a histogram with
a visible shape well before that, a tilt and a hump. Twenty bins and Pearson's
statistic throw the order of the bins away. A test on the empirical CDF of the
ranks, or one aimed at a tilt or a hump, should spend fewer simulations on the
bugs that only shift or widen.

**A harness that cannot fail on nan.** The dropped prior rate killed every
chain and Geweke counted zero rejections. Any test whose rejection is a
comparison with a number passes a sampler that returns no numbers, and none of
the four days guards for it.

**No data on the funnel.** Day 4's SBC is a rank test against the prior,
because the funnel has no likelihood. A funnel posterior from data, the
eight-schools hierarchy, is where both tests would be asked the question with
y in it.

**Small and closed-form.** A two-parameter normal model and a three-dimensional
funnel.

## Running

```bash
python day1_geweke_joint_test.py
python day2_sbc_ranks.py
python day3_planted_bugs.py
python day4_funnel_sbc.py
```

NumPy only. Day 4 imports `integrated_act` from the MCMC project's folder.
About 45 seconds, 110 seconds, 165 seconds and 175 seconds on this machine.
