# Trajectory Length Found Blind: The Period Guess, the U-Turn and a Length Tuned on the Chain

Where the eight schools project breaks, its second and third items. Every
non-centred pass there was found by a sweep: two units failed on `mu`, four
passed, six failed on `eta_1` with a deep U, eight broke it again, and the
six-unit failure was put down to `eta` turning every 2 pi units under unit
mass without anything testing it. A real model does not come with SBC run on
it first. Built from scratch in NumPy on the same eight schools potentials and
rank tests: a sweep of trajectory length through two periods, the same sweep
with jittered step counts, the No-U-Turn sampler's doubling tree with its
multinomial draw, and a length set during warmup from the lag of the widest
parameter.

## Idea

Everything is the non-centred eight schools model from the last project, SBC
with data drawn from the prior, chains started from an independent prior draw,
99 draws a simulation, 40 studies of 500 simulations, and the band,
chi-squared, rank mean and rank variance on `mu`, `log tau`, `theta_1` and
`eta_1`. Under unit mass a Gaussian coordinate's lag-1 autocorrelation after a
trajectory of length T is `cos(omega T)`, so each coordinate has a clock, and
a fixed length that lands on a multiple of half a period is a resonance. Day 1
sweeps the length at eps 0.1, where acceptance is 0.996 everywhere and nothing
is a rejection. Day 2 draws the step count fresh each transition. Day 3 lets
NUTS choose. Day 4 reads `mu`'s lag at one unit in warmup and sets the length
to its quarter period. Every configuration is priced in gradients a
simulation, mean steps x 99, burn-in left out.

**There are two clocks, not one. `eta` turns at about six units and fails at
every multiple of pi, a U where the chain returns and a W where it reflects,
and `mu` has a period near twenty units of its own, so fixed lengths pass at
four units and fourteen and nowhere else. Jitter of +/-50% removes every
resonance and turns four to ten units into a plateau. NUTS finds the four-unit
window with no sweep and fails in it on `mu`, 97% of studies, because the
U-turn sum is eight etas and one `mu`. A length read off `mu`'s own lag lands
at 3.8 units, fixes `mu`, fails on `eta` until it is jittered, and then passes
at 4500 gradients, NUTS's price plus 10%. The acceptance rate and the rank
mean saw none of it in any cell.**

## Layout

- `day1_period_sweep.py` - fixed lengths from one to fourteen units at eps
  0.1 with 2 pi and 4 pi added, lag-1 of each coordinate against `cos(T)`,
  and the four rank tests at each.
- `day2_jitter.py` - the step count drawn uniformly on `[c(1 - w), c(1 +
  w)]` per chain per transition, centres at three, four, six and ten units,
  `w` 0.5 and 1, integrated per chain with a mask.
- `day3_nuts.py` - NUTS on every chain at once, unit mass, depth cap 8, the
  generalised U-turn on every aligned block and biased progressive sampling,
  checked on a ten-dimensional normal, then SBC on non-centred at eps 0.1 and
  0.2 and centred at 0.1.
- `day4_length_tuned_on_the_chain.py` - 100 warmup transitions at one unit,
  the widest or the slowest coordinate's lag-1 turned into a quarter period,
  and that length fixed or jittered +/-50% per chain.

## What the days actually show

**Day 1: the six-unit break was `eta`'s period, counted at half its ticks.**
eps 0.1 x 60 reproduces eps 0.3 x 20's deep U on `eta_1`, ends 2.43 and 2.40
against 2.45 and 2.56, so it was the length and not the step. `eta_1`'s lag-1
follows `cos(T)` to a few hundredths, +0.50 at one unit, -0.95 at three, +0.85
at 2 pi. The failures sit at every multiple of pi, a U at six and 12.6 units
and a W at three and nine, band 100% at both. The U is deeper at 6.0 than 6.3,
so `eta`'s period is about six units, a 4% correction the ranks see and lag-1
does not. `mu` falls from +0.92 at one unit to -0.96 at ten, a period near
twenty, a U below two units and a W at ten and eleven. Two passes in fourteen,
four units at 0.20 and fourteen at 0.25. Two right, one half, one wrong.

**Day 2: jitter fills in every tick, including the one it was said not to.**
At three and six units the band on `eta_1` drops from 100% to 7% to 12%. At ten
units `mu`'s W goes too, 10% and 7%, though +/-5 units is only a quarter of its
clock; halving the antithetic lag, -0.96 to -0.63, was already below what the
band sees at 500. What is left is `mu` at three units, too short rather than
resonant, rho1 +0.58, joint 0.42. Four to ten units sit at 0.23 to 0.40,
within the 0.07 noise of 40 studies of each other. Two right, one wrong, one
not predicted.

**Day 3: NUTS finds the window's length and fails in it.** The sampler is
right on a normal with scales 0.5 to 5, variances within 0.5%. Non-centred
stops at 4.2 units at eps 0.1 and 4.4 at eps 0.2, so the posterior sets the
length and not the step, and it is day 1's window found without a sweep. The
band still rejects 97% of studies at 500, `mu` alone at 0.90 and 0.95, a U with
lag-1 +0.74. The etas turn at about pi and decide when the tree stops; `mu`
needs longer. That the multinomial draw from inside the tree makes it worse
than a fixed four units is a guess this project did not test. Centred goes to
depth 7 or 8, 1.87% divergent, and is rejected on 100% of studies by n = 100.
No prediction written.

**Day 4: a length that listens to `mu` fixes `mu`.** The widest and the
slowest coordinate are `mu` on 99% and 98% of chains, lag-1 +0.92 at one unit,
and the quarter period puts the trajectory at 3.8 units median, 2.6 to 6.8
across chains, flat in the study's tau. `mu`'s band is 0.07 to 0.17 against
NUTS's 0.90, but its lag-1 is +0.29 where the rule aimed for zero: `mu` is not
harmonic and arccos(0.92) gives a 15.6-unit period against day 1's near twenty.
Fixed per chain it fails on `eta_1`, 0.90, because a per-chain length is a
fixed length for each chain and the ones near three and six units sit on
`eta`'s ticks. Jittered +/-50% it passes, joint 0.25 and 0.40, nothing alone
over 0.17. One right, one wrong in its number, one not predicted.

## What a passing study costs, in gradients a simulation

Band rejections at 500 simulations, any parameter, 40 studies; a correct
sampler sits near 0.2 to 0.3 here and the error of one share is about 0.07.

| sampler | length | gradients | band at 500 | verdict |
|---|---|---|---|---|
| fixed, eps 0.1 | 3 units | 2970 | 100%, `eta_1` W | fails |
| fixed, eps 0.1 | 4 units | 3960 | 0.20 | passes |
| fixed, eps 0.1 | 6 units | 5940 | 100%, `eta_1` U | fails |
| fixed, eps 0.1 | 10 units | 9900 | 100%, `mu` W | fails |
| fixed, eps 0.1 | 14 units | 13860 | 0.25 | on the line |
| jittered +/-50% | 3 units | 2970 | 0.42, `mu` too short | fails |
| jittered +/-100% | 4 units | 3960 | 0.40 | on the line |
| jittered +/-50% | 6 units | 5940 | 0.23 | passes |
| jittered +/-100% | 10 units | 9900 | 0.23 | passes |
| NUTS, eps 0.2 | 4.4 units | 2194 | 97%, `mu` | fails |
| NUTS, eps 0.1 | 4.2 units | 4121 | 97%, `mu` | fails |
| NUTS centred, eps 0.1 | 14.8 units | 14645 | 100%, `log tau` | fails |
| tuned on `mu`, fixed | 3.8 units | about 4500 | 0.90, `eta_1` | fails |
| tuned on `mu`, +/-50% | 3.8 units | 4498 to 4538 | 0.25 to 0.40 | passes |

The cheapest pass is still the fixed four units the sweep found. The cheapest
pass found blind is the tuned and jittered length at 4500, plus 1000 gradients
of warmup at one unit that the other rows did not count either.

## Key design choices

**eps 0.1 throughout the sweeps.** Acceptance is 0.996 at every length, so
every failure in days 1 and 2 is the length's and none is a rejection's.

**Lag-1 read against `cos(T)`.** The harmonic prediction is free under unit
mass and was right for `eta` to a few hundredths, which is what made the
second clock, where it was wrong, visible.

**Jitter integrated per chain.** The vectorised version integrated every chain
to the longest draw and paid 1.5x to 2x in wall time; the gradient column
counts the mean, which is what a per-chain sampler pays.

**NUTS written for every chain at once and checked before it was trusted.**
Ten dimensions, scales 0.5 to 5, 4000 chains, before any SBC run.

**Both warmup picks kept.** Widest and slowest choose the same coordinate on
98% of chains, so their 0.25 against 0.40 is the size of the noise, measured
and not assumed.

**Predictions written into the files before running them**, except day 3,
which had none. Five right, one half, three wrong, two not predicted.

**Fixed seeds throughout**, with each imported module's generator reseeded.

## Where it breaks

**Unit mass is what gives the model two clocks.** `mu`'s posterior sd is near
three and the etas' near one, so under unit mass their periods differ by about
three times, and every failure in four days is one coordinate's clock against
another's. A diagonal mass matrix set from warmup variances, which is what
production samplers do, should put both on one clock, and then NUTS's U-turn
and a fixed length might serve every coordinate at once. Nothing here used
anything but unit mass.

**The multinomial draw is a guess.** NUTS's lag on `mu`, +0.74, is worse than
fixed four units' at nearly the same length, and the guess is that drawing
from inside the tree moves `mu` less than four units of integration would.
Drawing from the tree's far end, or comparing slice and multinomial
selection, would test it.

**The quarter period is read from one point.** arccos of the lag at one unit
put `mu`'s period at 15.6 units and day 1 read near twenty, because `mu` is not
harmonic. Lag at two or three warmup lengths would fit the clock instead of
assuming it.

**Forty studies cannot rank the plateau.** Four to ten units jittered, and
both warmup rules, are within 0.07 of each other, so whether any blind rule is
better than another on this model was not measured, only that each passes.

**Centred was left behind.** No length rule touched it; NUTS spent 14.8 units
in the funnel and failed. Whether a mass matrix adapted in warmup helps a
model whose scale changes with position, which a single matrix cannot follow,
is the funnel's old question again.

## Running

```bash
python day1_period_sweep.py
python day2_jitter.py
python day3_nuts.py
python day4_length_tuned_on_the_chain.py
```

NumPy only. Day 1 imports the eight schools project's day 1 to 4 modules and
the sharper rank tests project's band and shape tests, which import the SBC
project's; days 2 to 4 import day 1 for all of it. About 1400, 870, 1690 and 500 seconds on this machine.
