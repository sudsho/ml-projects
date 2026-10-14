# Sharper Rank Tests: ECDF Bands, Shaped Alternatives and Ranks That Know About Autocorrelation

Where the SBC project breaks, the first two items on its list. A slow sampler
that leaves the posterior invariant failed both of its tests on nearly every
study and needed a thin of 100 before it passed, because both tests read
autocorrelation as bias. And the binned chi-squared spent 500 simulations on
the shape bug and caught the dropped half kick 17% of the time at 500, when
both histograms had a visible tilt or hump long before that, since twenty bins
and Pearson's statistic throw the order of the bins away. Built from scratch in
NumPy on the SBC project's own planted bugs, its model in `(mu, log tau)` and
its funnel: the ECDF of the ranks under a simultaneous band, tests aimed at a
shift or a change of width, and ranks taken with the chain's measured
autocorrelation in hand.

## Idea

SBC's ranks are uniform on `0..99` for a correct sampler and every test here
is a function of their counts. The band keeps the order the chi-squared threw
away: the ECDF at the 99 inner rank values, each point scaled by its binomial
sd, and the 95% point of the largest scaled excursion on 20,000 sets of uniform
ranks, so the band and the test are one object. The shaped tests are one number
each, `|mean - L / 2|` for a tilt and `|var - V|` for a hump or a U, with their
95% points simulated the same way. Days 1 and 2 run all of them on the SBC
project's seven samplers at thin 10. Day 3 measures each chain's IACT from a
990-draw pilot and takes the ranks against an ESS-sized subsample or from the
chain run on at thin `ceil(IACT)`. Day 4 takes the four statistics and five
thins to the funnel, where the SBC project's chi-squared rejected a wrong
sampler and a slow correct one on every study alike.

**The band is sharper than the chi-squared on every bug that tilts, and
sharper on the slow correct sampler too, which is the wrong way. Thinning by
the measured IACT does not fix that at any price tried, 8 times the
transitions on the model and 16 times on the funnel, because a pilot reads a
lag it cannot hold and a subsample of it holds the same lag. What passes the
slow samplers is the rank mean, at every size on both the model and the
funnel, and it catches both shift bugs sooner than the band does. It does so
because a chain that never crosses the truth piles ranks at both ends evenly
and a wrong target piles them at one, and it is blind to the one bug that only
widens.**

## Layout

- `day1_ecdf_bands.py` - the standardised ECDF of the ranks, the simultaneous
  band by simulation and its exact coverage against 1.96 pointwise intervals,
  and the band, the chi-squared and a plain KS distance on the seven samplers.
- `day2_shape_tests.py` - the rank mean, the rank variance and the larger of
  the two in null sds, calibrated the same way, on the same samplers and on a
  bug made at the level of the ranks: gamma draws with the right mean and
  variance and the wrong skew.
- `day3_lag_aware_ranks.py` - each chain's IACT by Sokal's window from a
  990-draw pilot, with the nan guard the SBC project did not have, and the
  ranks taken against an ESS-sized subsample or from the chain run on at thin
  `ceil(IACT)`, so a slow chain pays for its lag in transitions.
- `day4_funnel_rank_tests.py` - the RMHMC project's Fisher sampler with and
  without the log det and centred HMC on the funnel, as arrays of chains, under
  all four statistics at thin 1, 5 and 10, the ESS subsample and run longer.

## What the days actually show

**Day 1: the band covers what it says and reads lag as bias more readily than
the test it replaces.** Coverage 0.949 to 0.955 at every n; pointwise 1.96
intervals cover all 99 points together 43% to 51% of the time. The log det
tilt reaches 45% at 100 and 100% at 500 under the band against 65% at 500 for
the chi-squared, 2.2 to 2.6 times its rate in between. The Hastings shift was
already cheap and plain KS gets it at 50 with 90%, where the band has 62%:
scaling the ends up spends the width where nothing is happening. Nothing finds
the half kick. The slow walk is rejected on 78% at 20 simulations and the
correct walk at step 0.5 on 12% to 20%, a small U from its lag that the
chi-squared read at 3% to 5%. Two right, one half, three wrong.

**Day 2: the rank mean is the cheapest test of a shift and does not see the
slow walk at all.** Hastings 97% at 50 under the mean against 72% for the band;
log det 78% at 200 against 62%. The slow walk at 10% to 17% from 50 to 500
under the mean, inside the size, where everything else passes 80% by 100: its
U moves the variance by 9.4 null sds and the mean by 0.4. The variance is the
right test for the half kick and still not enough, 38% at 500 against 25% for
the band, and gives the tilt up at 23%. The skewed draws, right in both
moments, are found by the band at 85% at 100 and not by either aimed test
until 500. One right, one half, four wrong.

**Day 3: the measured lag does not buy what thinning by 100 bought.** The slow
walk's pilot IACT is 77 at the median; thinning by it costs 8,997 transitions a
simulation against 1,090 and leaves the band at 42% at 500, because 990 draws
read a lag they cannot hold. The ESS subsample delays the band from 20 to 200
simulations and rejects every study by 500. The log det and Hastings rows do not
move under either. The new failure is HMC: its lag 1 is -0.66, the IACT floor
of 1 thins it by 1, and the band rejects the correct sampler on 17% to 28%.
Two right, one half, three wrong.

**Day 4: on the funnel the two failures are told apart by the statistic and
not by the thin.** The log det bug is a pile in v's top tenth, 9.52 times
uniform, and every statistic rejects it on every study at 20 simulations,
3,980 transitions a study. Centred HMC is a U at both ends, 1.72 and 1.81, the
band rejects it on 100% from 50 and the rank mean on 3% to 20% at every size
under every scheme. Thin `ceil(IACT)` costs 16 times thin 1 and leaves the band
at 100%; 3.4% of HMC's chains never moved in 990 transitions and the nan guard
caught them. The correct RMHMC sits at 15% to 20% under the band and 3% to 10%
under the mean. Three right, two half, one wrong.

## What each statistic costs, in simulations to 80% rejections

Model rows are thin 10, 40 studies, either parameter, read against a size of
about 10% with a standard error near 5. Funnel rows are thin 1. A dash is not
reached by 500.

| sampler | moves the target | chi-squared | band | rank mean | rank variance |
|---|---|---|---|---|---|
| three correct samplers (model) | no | - | - | - | - |
| RMHMC, log det dropped (model) | `E[tau]` +8%, a tilt | 500 | 500 | 500 | - |
| walk, Hastings factor dropped (model) | `E[tau]` -19%, a tilt | 100 | 100 | 50 | 500 |
| HMC, last half kick dropped (model) | `E[tau]` -0.8%, a hump | - | - | - | - |
| slow random walk, correct (model) | no, a U | 200 | 50 | - | 100 |
| gamma draws, right moments, wrong skew | skew 1 | 200 | 100 | - | - |
| funnel RMHMC, log det dropped | v to N(-9, 9), a pile | 20 | 20 | 20 | 20 |
| funnel centred HMC, correct | no, a U | 100 | 20 | - | 100 |
| funnel RMHMC, correct | no | - | - | - | - |

The one column with a dash on both slow samplers and a number on both shift
bugs is the rank mean. The one bug with a dash in every column is the half
kick, and the rank variance's 38% at 500 is the most anything reached on it.

## Key design choices

**Calibrate by simulation and read a row against the correct samplers.** Every
critical value is the 95% point of 20,000 sets of uniform ranks at that n, and
the correct samplers sit at 3% to 20% on either parameter, so 10% is the size
and a cell under 20% is not a finding.

**The band and the test are one object.** The band's half-width is the
statistic's own critical value, so a plot read against it and the test agree
by construction, and pointwise intervals, which they do not, were measured to
cover all points under half the time.

**Aim a statistic at the shape the bug has.** The SBC project said which way
each bug moved the histogram before any statistic here was chosen, and the
one-number tests were built to those shapes.

**Charge the chain for its lag, two ways, and measure the charge.** The IACT
is read from the same pilot thin 10 already pays for, and both schemes report
transitions a simulation alongside rejections, so a thin that passes a slow
chain is priced.

**A guard for a chain that returns no numbers.** A pilot with no variance gets
an IACT of its own length instead of nan, and day 4 is where it fired.

**Predictions written into the files before running them.** Across the four
days, eight right, five half, eleven wrong, and the docstrings say which.

**Fixed seeds throughout**, with every imported module's generator reseeded so
no day's rows share draws with another's.

## Where it breaks

**The rank mean passes slow chains for a reason that is not general.** It is
blind to a symmetric U, and both slow samplers here are symmetric because they
start from a prior draw and go nowhere. A slow chain started from a point, or
one whose lag is in a skewed direction, would tilt, and the mean would read it
as a shift. And the mean sees nothing of a width bug by construction. Slow
against wrong is still not a question these tests answer; the mean answers
slow-and-symmetric against shifted.

**Thinning by a measured IACT never passed a slow chain.** Not at 8 times the
cost on the model and not at 16 on the funnel. The pilot's length bounds the
lag it can read, a subsample of it holds its lag, and an antithetic chain gets
a floor that thins it by 1 and fails it. The thin that worked, 100 on the slow
walk, was chosen knowing the answer.

**The half kick is caught by nothing in four days.** 38% at 500 under the
variance is the best number, and the bug moves `E[tau]` under 1%. A test on the
draws themselves rather than on the ranks of the truth, the Geweke moment
comparison the SBC project's day 1 ran, had 52% on it at 10,000 transitions,
and nothing here has a route to a hump that is cheaper than that.

**No data on the funnel, still.** Every funnel row in two projects is a rank
test against the prior, because the funnel has no likelihood. The eight-schools
hierarchy is the funnel with data in it, where centred HMC's failure is the
textbook one and the non-centred parameterisation is the textbook fix, and
where SBC would be run with y in it for the first time in this series.

**Two parameters, 40 studies.** A rate from 40 studies has a standard error
near 5 points, and most of the cells that decided a prediction were 20 points
or more from the size. The ones under 20 were not read.

## Running

```bash
python day1_ecdf_bands.py
python day2_shape_tests.py
python day3_lag_aware_ranks.py
python day4_funnel_rank_tests.py
```

NumPy only. Days 1 to 3 import the SBC project's `day3_planted_bugs`, day 4
its `day4_funnel_sbc`, which imports the MCMC project's `day1_metropolis`.
About 90, 90, 160 and 245 seconds on this machine.
