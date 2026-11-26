# Too Long on Log Tau: Where the Tree Stops and Whether It Should

Where the noise of a rejection share breaks, its first item, with the last as
the tool it needs. On 120 studies NUTS failed on `log tau` under every
diagonal mass tried, per-chain, pooled over a study's 500 chains or fixed at
the prior's variances, with `log tau`'s lag-1 at +0.32 or more under all of
them. The mass matrix project guessed the tree runs too long on `log tau`'s
side, the reverse of the trajectory project where it was too short for `mu`,
and nothing had tested it. Built from scratch in NumPy on the same
non-centred eight schools potentials, tree and rank tests: common random
numbers so two samplers from one state are a paired reading, a traced tree
that records where `log tau` turns and where the draw lands, the tree capped
or its draw kept off the last doubling, and partial momentum refresh.

## Idea

If the tree overshoots `log tau`, the draw comes back toward where `log tau`
started and the lag-1 stays positive. Then three things follow: the draw
lands past `log tau`'s turn, the studies rejected on `log tau` are the ones
where it does so most, and a shorter tree fixes it. Day 1 builds the paired
design the last project lacked, so a gap between two samplers is read on the
same studies, states and numbers. Day 2 traces the tree. Day 3 cuts it. Day 4
tries the other way to change how far a draw moves: keep the momentum
between transitions instead of drawing it fresh.

**The guess was backwards. The tree does overshoot `log tau`, on 0.61 of
transitions and on rejected and passing studies alike, and every cut of the
tree makes `log tau` worse: depth 5 puts its share up 0.21, the first half
of the same tree up 0.61. The lag is the direction being forgotten. Plain
HMC at one unit has the tree's +0.42; the same unit with the momentum kept
at alpha 0.8 has +0.09 and fails `log tau` on 0.075 of 120 studies against
the full tree's 0.367, at the same 40 gradients a draw. Pairing saved
nothing on the joint.**

## Layout

- `day1_paired_readings.py` - `noise`, every random number of a NUTS
  transition from a stream fixed by (seed, transition), and `nuts_crn`, the
  mass matrix project's tree with those numbers handed in; per-chain against
  pooled mass on one seed and two.
- `day2_where_the_tree_stops.py` - `nuts_traced`, the same tree with every
  merged leaf's `mu` and `log tau` kept at its signed step, the turn and the
  draw's place against it, per-chain, pooled and unit mass.
- `day3_the_tree_cut_short.py` - `nuts_cut`, the tree capped at 3 to 6
  doublings or its draw taken from the half the start sits in, all on the
  same numbers as the full tree, three sets of 40.
- `day4_partial_refresh.py` - `ghmc`, generalised HMC with the momentum
  kept at weight alpha and flipped on a rejection, half a unit to two units
  at alpha 0 to 0.95, thinned to 40 gradients a draw, against the full tree.

## What the days actually show

**Day 1: pairing holds the chains apart, it does not join them.** On the
ten-dimensional normal not one chain in 4000 meets its partner. On eight
schools two masses on one seed sit 0.37 apart in `log tau` against 0.82 on
two seeds, but a study's joint verdict disagrees between them on 31.2%
paired and 30.0% unpaired, sd of a gap at 40 studies 0.088 against 0.086.
Ranks track at +0.98 either way, because where the truth sits sets most of a
rank. On `log tau` alone pairing saves an eighth of the studies, 94 against
108. The prediction, that pairing would cut the sd well under the last
project's, was wrong.

**Day 2: the tree overshoots, and it does not matter.** `log tau` turns
inside the tree on 0.81 of transitions, a median 10 steps out of 34, and
the draw lands past the turn on 0.61. The tree's length barely moves with
`log tau`, 37 steps in the neck and 33 at the top, while the turn comes at
11 and 5 or 6. Rejected and passing studies draw past the turn on 0.605 and
0.605, and the mass that fails least overshoots most. Right that it turns
early and the draw lands past it, wrong that this sorts the studies.

**Day 3: shorter is worse.** The full tree fails 0.467 joint and 0.375 on
`log tau`. Depth 5 caps 0.29 of transitions, saves a quarter of the
gradients and puts `log tau` up 0.208 (paired sd 0.042) against `mu` up
0.042. Depth 4 fails every study on `log tau`. The draw from the first half,
the same tree and gradients, fails 0.983 with lag-1 +0.72. The draws past
the turn are the part of the move `log tau` needs. Wrong.

**Day 4: keep the direction.** Every refresh keeps its target on the normal.
On 120 studies the full tree is 0.508 joint, 0.367 `log tau`, lag-1 +0.40;
one unit at alpha 0 is 0.375 / 0.267 / +0.42; one unit at 0.8 is 0.200 /
0.075 / +0.09, and half a unit at 0.95 the same 0.200 / 0.075. Every refresh
beats the tree on every set. More memory is not better: one unit at 0.95
has the lowest lag-1, +0.06, and fails `log tau` on 0.175. `mu` goes
antithetic under any refresh, -0.42 to -0.57. No prediction was written.

## The samplers on log tau

Non-centred eight schools, per-chain mass after the 100 window, band at 500
simulations. Gradients a study include the 2500 of warmup.

| sampler | studies | gradients | joint | `mu` | `log tau` | lag-1 `log tau` |
|---|---|---|---|---|---|---|
| full tree (day 3) | 120 | 8500 | 0.467 | 0.100 | 0.375 | +0.41 |
| depth 5 (day 3) | 120 | 7090 | - | 0.142 | 0.583 | +0.46 |
| first half (day 3) | 120 | 8500 | - | 0.442 | 0.983 | +0.72 |
| full tree (day 4) | 120 | 8500 | 0.508 | 0.100 | 0.367 | +0.40 |
| one unit, alpha 0 | 120 | 8460 | 0.375 | 0.067 | 0.267 | +0.42 |
| one unit, alpha 0.8 | 120 | 8460 | 0.200 | 0.050 | 0.075 | +0.09 |
| one unit, alpha 0.95 | 120 | 8460 | 0.333 | 0.058 | 0.175 | +0.06 |
| half a unit, alpha 0.95 | 120 | 8460 | 0.200 | 0.050 | 0.075 | +0.07 |
| two units, alpha 0.8 | 120 | 8460 | 0.225 | 0.058 | 0.125 | +0.08 |

For scale, from the last project: a correct sampler's joint is 0.143 to
0.185, unit mass at four units 0.223 over 1280 studies at 6710 gradients.

## Key design choices

**Numbers handed in.** Every sampler from day 1 on takes its random numbers
from `noise(seed, t, ...)`, so a variant and the full tree can be run on the
same studies, states and numbers and their gap read study by study.

**The full tree in every comparison.** Day 3 checks `nuts_cut` at full depth
against day 1's tree to the bit; day 4 reruns it on its own studies rather
than quoting day 3's.

**Equal gradients.** Day 4's fixed lengths are thinned so each kept draw
costs 40 gradients, the tree's 40.3, so a difference in share is not a
difference in work.

**Targets checked first.** Every new transition is run on a ten-dimensional
normal under the exact mass before eight schools, variance over truth within
1% for all of them.

**Predictions written into the files before running them.** None right, one
half, two wrong, and day 4 wrote none.

**Fixed seeds throughout**, with each imported module's generator reseeded.

## Where it breaks

**Alpha has an optimum and nothing says where.** 0.8 at one unit and 0.95
at half a unit read the same, 0.95 at one unit is worse. If what matters is
how far a direction is carried, about 1 / (1 - alpha) transitions of the
length, those three are 5, 10 and 20 units. But two units at 0.8 also
carries 10 and fails `log tau` on 0.125 against half a unit's 0.075, so the
carry alone does not decide it. A sweep of alpha at each length is the next
test.

**`mu` goes antithetic.** Lag-1 -0.42 to -0.57 under every refresh, with
`mu` passing at 0.05. The band does not mind; a summary of `mu`'s variance
from the chain would.

**0.200 is not shown nominal.** 1.5 binomial sd above a correct sampler on
120 studies. Whether partial refresh is right or only past the line needs
the 433 studies the last project put on that question.

**Lag-1 is not the autocorrelation.** At alpha 0.95 `log tau`'s lag-1 is the
lowest of all and the band fails it twice as often as at 0.8. The slow part
of the chain the lag-1 misses was never measured; an integrated
autocorrelation per variant would say what the band is reading.

**Refresh inside the tree was not tried.** NUTS draws a fresh momentum every
transition by construction. Keeping part of it requires the tree's draw to
return a momentum that keeps the joint, which this project did not build.

## Running

```bash
python day1_paired_readings.py
python day2_where_the_tree_stops.py
python day3_the_tree_cut_short.py
python day4_partial_refresh.py
```

NumPy only. Day 1 imports the mass matrix project's day 1, which puts the
eight schools, rank tests and SBC modules on the path; days 2 to 4 import
day 1, and day 4 imports day 3 for the full tree. About 2750, 1470, 3000 and
1560 seconds on this machine.
