# The Noise of a Rejection Share: How Far Forty Studies Move the Band

Where the mass matrix project breaks, its third item, with the second as the
first thing the answer is used on. Every verdict in three projects was the
share of 40 studies the band rejects at 500 simulations, read as a pass near
0.25 and a fail near 0.40 against a noise put at 0.07 and never measured.
Then one sampler read 0.60 on one set of studies and 0.42 on the next, and
the mass matrix day 1's only pass, 0.25 at two units, came back 0.47 and
0.38. Built from scratch in NumPy on the same non-centred eight schools
potentials, samplers and rank tests: the share when the ranks are uniform by
construction, the same share over many fresh sets of studies for a passing
and a failing sampler, how the spread falls with the number of studies, and
the mass pooled over a study's chains or fixed at the prior's variances
against the per-chain one.

## Idea

A share of 40 studies is a count out of 40. If the studies of a set are
independent and each is rejected with one chance p, the count is binomial
and its sd is sqrt(p (1 - p) / 40), 0.068 at 0.25 and 0.077 at 0.40. Two
things could make it wider: something shared by the studies of one set, or
studies whose chances differ in a way that correlates within a set. Day 1
measures the share where nothing can be wrong, ranks drawn uniform. Day 2
runs the old pass (unit mass, four units) and the old fail (NUTS under the
100 window mass) on many fresh sets. Day 3 pools sets into 80 and 160
studies, works out how many it takes to tell two shares apart, and reads
the old one-set verdicts against that. Day 4 uses the answer on the question
the mass matrix project left: whether NUTS's failure on `log tau` is the
per-chain mass's noise or the form of a diagonal mass.

**The binomial is the whole story at 40 studies for a sampler that is near
right, and every "on the line" verdict of the last two projects was a coin.
A correct sampler's joint share over four tests is 0.143 to 0.185 with sd
0.055, and reads above 0.25 on 2% of sets. Unit mass at four units, the old
pass, is 0.223 over 1280 studies, 3.3 sd above nominal: it passes the line
and is not a null sampler. 0.25 against 0.40 needs 103 studies; nominal
against the line needs 433. Of 14 old one-set verdicts 5 survive, all NUTS
fails at 0.42 or more, and no pass. On 120 studies NUTS fails on `log tau`
under every diagonal mass tried, per-chain, pooled over 500 chains, or the
prior's; pooling takes a third to a half off `log tau`'s band and nothing
off its lag-1.**

## Layout

- `day1_share_when_nothing_is_wrong.py` - uniform ranks by construction,
  2000 sets of 40 studies, three parameters tied through a Gaussian copula
  at correlation 0, 0.5 and 0.9, the band's share against the binomial.
- `day2_share_of_a_real_sampler.py` - `unit_set` and `nuts_set`, the two
  samplers the last project leaned on, 24 and 6 fresh sets, the between-set
  spread with a simulated binomial tail in `dispersion`.
- `day3_how_many_studies.py` - 32 and 4 more sets pooled into 80 and 160,
  `studies_needed` for a cut that misreads either side at most 5%, the
  gradients it costs, and the 14 old verdicts.
- `day4_noise_or_form.py` - three sets of 40 warmed once to the 400 window,
  NUTS from the same states under six masses: per-chain after 100 and 400,
  `pooled` over a study's chains after 25, 100 and 400, and the prior's.

## What the days actually show

**Day 1: the null share is binomial and nothing more.** Each parameter alone
rejects 0.048 to 0.053 at every size, sd 0.034 to 0.035 against the
binomial's 0.034 to 0.035. The joint at correlation 0 is 0.139 to 0.148
against 0.143, sd 0.054 to 0.056 against 0.055. Tying the parameters makes
the joint smaller, 0.116 at 0.9, not noisier. A correct sampler reads above
0.25 on 2.0% of sets, and the shares of one set at neighbouring sizes
correlate +0.18 to +0.31, so a row that fails at 200 and 500 is less than
two looks.

**Day 2: the old pass was a coin.** Unit mass at four units reads 0.241 over
24 sets, 9 of them above 0.25, range 0.125 to 0.375, with a spread at the
binomial (variance ratio 0.74). NUTS fails on all six, 0.275 to 0.500, joint
near binomial, but `log tau` alone is 0.075 to 0.400, ratio 2.48, a tail of
3%. At 100 simulations the two samplers cannot be told apart.

**Day 3: how many studies.** Unit mass holds the binomial at 40, 80 and 160
studies, sd x sqrt(studies) 0.41, and over 1280 studies sits 3.3 sd above
the 0.185 four tests at 5% give. NUTS's `log tau` ratio is 3.32 on four
more sets, so the extra spread is two readings of one thing. 0.25 against
0.40 needs 103 studies at a cut of 0.320, 0.35G gradients under unit mass;
the measured 0.223 against 0.481 needs only 34, so 40 did tell those two
apart. A pass needs 0.14 or under on 40 studies to be 5% clear of the line,
and the best old reading was 0.17. The prediction, the binomial's, was right
for unit mass and half right for NUTS.

**Day 4: form, mostly.** Pooling a study's 500 chains takes `log tau`'s
inverse mass from 0.33-1.65 over chains to 0.83-0.98 over studies after the
100 window, with the same median form. On 120 studies nothing passes: joint
0.400 and 0.408 per-chain after 100 and 400, 0.367, 0.325 and 0.358 pooled
after 100, 400 and 25, 0.333 under the prior's variances, every one 1.9 sd
or more above the line. `log tau` alone falls from 0.325 to 0.18-0.23 under
every pooled mass and the prior, and its lag-1 stays at +0.32 to +0.37
against +0.41. The 400 window buys nothing under the tree. The prior mass,
with only the fast window before it and 31 steps a draw, is as good as any,
though `mu` at 25 against its posterior 10.8 goes antithetic, lag-1 -0.30.
Running every mass from the same warmed state did not pair the readings: a
set's gaps swing -0.20 to +0.125, because the chains part after one
transition. One right, one wrong.

## What a share is worth

The joint band share at 500 simulations, four tests, NUTS and unit mass on
non-centred eight schools. The cut misreads either side at most 5%.

| comparison | studies needed | cut | gradients, unit mass |
|---|---|---|---|
| 0.25 against 0.40, binomial | 103 | 0.320 | 0.35G |
| measured 0.223 against 0.481 | 34 | 0.34 | 0.11G |
| 0.185 (nominal) against 0.25 | 433 | 0.216 | 1.45G |

| sampler | mass | studies | joint | `log tau` alone | verdict |
|---|---|---|---|---|---|
| null ranks | none | 2000 sets of 40 | 0.143 | 0.050 | nominal |
| fixed four units | unit | 1280 | 0.223 | 0.08 | passes the line, fails nominal |
| NUTS | per-chain, 100 window | 120 | 0.400 | 0.325 | fails |
| NUTS | per-chain, 400 window | 120 | 0.408 | 0.292 | fails |
| NUTS | pooled, 25 window | 120 | 0.358 | 0.183 | fails |
| NUTS | pooled, 100 window | 120 | 0.367 | 0.225 | fails |
| NUTS | pooled, 400 window | 120 | 0.325 | 0.233 | fails, P 0.029 |
| NUTS | prior variances | 120 | 0.333 | 0.233 | fails |

## Key design choices

**The null first.** Before any sampler, the share where nothing can be wrong,
so that every later spread has a floor that was measured, not assumed.

**Many sets, not many studies in one set.** The question was the spread
between sets, so the unit sampler ran 56 sets of 40 and NUTS 10, and pooling
into 80 and 160 came after.

**Simulated tails.** `dispersion` gets its chance from 20,000 binomial sets of
the same size, and `_tail` uses the standard library's `erfc`, so the
project stays NumPy only.

**The same warmed state under every mass on day 4.** One warmup with a
snapshot after every window, so a difference between masses is not a
difference in warmup. It did not make the readings paired, and the table
reads them on their pooled 120.

**Old verdicts re-read, not re-run.** Day 3 takes the 14 one-set shares as
printed by their own projects and asks only what a share that far from the
line is worth.

**Predictions written into the files before running them.** Two right, one
half, one wrong; days 1 and 2 wrote none for their numbers.

**Fixed seeds throughout**, with each imported module's generator reseeded.

## Where it breaks

**The lag on `log tau` is the tree's, under any diagonal mass.** Per-chain,
pooled or the prior's, `log tau`'s lag-1 stays above +0.32 and the band on
it alone above 0.18. The mass matrix project's untested guess was that the
tree is too long on `log tau`'s side, the reverse of the trajectory project,
where it was too short for `mu`. Nothing here tested it either, and it is
now the only explanation left standing.

**`log tau` is wider than binomial under NUTS and nothing says why.** Ratios
2.48 and 3.32 on ten sets in all. Something about a set of studies moves
NUTS's `log tau` together, and the obvious suspect is the share of studies
whose true tau is small, which was never recorded per set.

**Unit mass at four units is not a null sampler.** 0.223 against 0.185 over
1280 studies. Which parameter carries the excess, and whether more burn or
a jitter takes it to nominal, is not known.

**The prior mass is too light on `mu`.** 25 against 10.8 makes `mu`
antithetic at -0.30 without hurting its band, and a mass from the prior
times a fixed shrinkage was not tried.

**Paired runs part at once.** The same state under two masses does not give
a paired comparison, because the momenta differ from the first draw. Common
random numbers for the momenta would, and would cut the studies needed.

## Running

```bash
python day1_share_when_nothing_is_wrong.py
python day2_share_of_a_real_sampler.py
python day3_how_many_studies.py
python day4_noise_or_form.py
```

NumPy only. Day 1 imports the rank tests project's day 1; day 2 imports the
mass matrix project's days 1 and 2, which put the eight schools, rank tests
and SBC modules on the path; day 3 imports day 2, and day 4 imports days 2
and 3 here and the mass matrix day 4 for the snapshots. About 90, 2830,
2470 and 6100 seconds on this machine.
