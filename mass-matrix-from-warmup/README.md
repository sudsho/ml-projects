# A Mass Matrix From Warmup: One Clock for Every Coordinate

Where the trajectory length project breaks, its first two items. Every failure
there was one coordinate's clock against another's: under unit mass the etas,
posterior sd near one, turn every six units, and `mu`, sd near three, every
twenty, so fixed lengths passed at four units and fourteen only, NUTS stopped
when the etas turned and failed on `mu` at 97%, and a length tuned on `mu`
failed on `eta` until it was jittered. Production samplers do not run unit
mass; they set a diagonal mass from warmup variances so every coordinate has
roughly the same clock. Built from scratch in NumPy on the same non-centred
and centred eight schools potentials and rank tests: a diagonal mass from
windowed warmup, the period sweep and NUTS again under it, a dense mass from
the warmup covariance, and the multinomial draw against a draw from the
tree's end.

## Idea

Everything is the eight schools model from the last two projects, SBC with
data drawn from the prior, chains started from an independent prior draw, 99
draws a simulation, 40 studies of 500 simulations, and the band, chi-squared,
rank mean and rank variance on `mu`, `log tau`, `theta_1` and `eta_1`. The
warmup is Stan's schedule run on each chain: a fast window of 75 transitions
at unit mass, then slow windows of 25, 50 and 100, the variance re-estimated
and shrunk toward 1e-3 at the end of each, every transition one unit long at
eps 0.1. If the inverse mass is each coordinate's posterior variance, every
Gaussian coordinate turns once every 2 pi units, and one length serves all of
them. Day 1 sweeps the fixed length under that mass. Day 2 runs the tree
under it and tests the last project's guess about the multinomial draw. Day 3
estimates the whole covariance and takes both masses to the centred model.
Day 4 keeps the mass at the end of every window, with two more doublings to
200 and 400, and asks how long the warmup has to be. Every configuration is
priced in gradients a simulation, and from day 1 on the warmup is counted.

**The mass does what it was built for and that was not the whole problem. It
finds the ratio of the two clocks, 3.35 against the 3.3 the periods implied,
and under it NUTS's failure on `mu` is gone, band 0.93 to 0.03, with the tree
no longer than before. What fails then is `log tau`, which has no clock at
any length, and whose inverse mass is still climbing after a 400-transition
window. The one pass under the mass, fixed two units after that window, costs
11480 gradients a simulation against 6710 for unit mass at four units. The
tree fails on `log tau` after either warmup. The draw from the tree's end was
not what held `mu` back, and no constant matrix, diagonal or dense, moves the
centred model's verdict.**

## Layout

- `day1_diagonal_mass_sweep.py` - `hmc_mass` with a per-chain diagonal mass,
  the windowed `warmup`, and fixed lengths from one to fourteen units from
  the warmed state, lag-1 of each coordinate against `cos(T)`.
- `day2_nuts_under_the_mass.py` - `nuts_mass`, the last project's tree with
  the U-turn on velocities and either the multinomial or the far-end draw,
  checked on a ten-dimensional normal, then four runs on the same studies:
  unit or adapted mass, either draw.
- `day3_dense_mass_and_centred.py` - `nuts_dense` under a full matrix from
  the window's covariance, checked on a correlated normal, then non-centred
  and centred, diagonal and dense, each from its own warmup.
- `day4_what_the_warmup_costs.py` - the state and the mass kept after windows
  of 25, 50, 100, 200 and 400, fixed two and four units from each on two
  independent sets of studies, and the tree after the 100 and the 400.

## What the days actually show

**Day 1: one clock for half a turn, and a coordinate with none.** The final
inverse mass is 9.35 on `mu`, 0.72 on `log tau` and 0.83 on `eta_1`, so the sd
ratio is 3.35. `mu`'s estimate is 2.07, 5.77 and 9.35 over the three slow
windows and has not stopped. `mu` and `eta_1` share a clock to 3.1 units,
lag-1 -0.76 and -0.87 where `cos` is -1.00, then `mu` dephases because each
chain has its own mass and so its own frequency. `log tau`'s lag-1 never goes
negative, +0.68 at one unit and +0.06 at its lowest, and it fails the quarter
turn at 1.6 units where the other two are near zero lag, 0.60 on it alone.
One pass in nineteen lengths, two units at 0.25, where unit mass had two.
One right, one half, one wrong, two not predicted.

**Day 2: the end draw was not it, and the mass fixes `mu` without moving the
tree.** Under unit mass the draw from the tree's far end leaves `mu`'s lag-1
at +0.77 against +0.74. The band on `mu` does fall, 0.93 to 0.40, but the
end draw is biased when the depth depends on the start, variances 0.95 to
1.06 on a normal, and draws that are too wide cancel the U of a slow chain.
Under the adapted mass the tree is 40.3 steps against 41.6 and `mu` is done,
lag-1 +0.08, band 0.03. `log tau` is 0.47 alone, a pile in its lowest tenth
and not a U, with the rank mean passing. Three right, one of them for the
wrong reason, one wrong.

**Day 3: nothing to find on non-centred, and no matrix has the neck in it.**
The non-centred posterior's median correlations are 0.00, and a matrix built
from a hundred draws in ten dimensions still has condition number 10.6, so
the dense tree is 55.4 steps for 40.3 and `log tau` is 0.38 against 0.28. The
same diagonal sampler that read 0.60 on day 2's studies reads 0.42 here, so
40 studies move a share by 0.2. On centred the diagonal mass shortens the
tree from 148 steps to 61.5 and the dense one cuts divergences five times,
1.08% to 0.20%, and both leave `log tau`'s lag-1 at +0.80 and the band at
100% from 100 simulations. Where the true tau is below 1, `log tau`'s rank is
in the lowest tenth on 0.83 and 0.78 of simulations against 0.62 on the
non-centred runs. Three right, one half.

**Day 4: the warmup's length is set by the coordinate with no clock.** `mu`'s
inverse mass goes 2.05, 5.73, 9.36, 10.43, 10.76 over five windows and levels
near 10.8. `log tau`'s goes 0.50, 0.55, 0.72, 0.90, 1.02 and is still rising,
chains from 0.65 to 1.67. Day 1's pass at two units was the draw: from the
same 100 window it is 0.47 and 0.38 on two fresh sets. It passes only after
the 400 window, 0.23 on both, and what moved is `log tau` alone, 0.35 and
0.20 after 200 against 0.12 and 0.05 after 400. Four units after a 25 window
is 1.00 and 0.97 on `eta_1`, worse than no mass at all. The tree is 0.47
after the 100 window and 0.42 after the 400. One right, one half, one wrong,
one not predicted.

## What a passing study costs, in gradients a simulation

Band rejections at 500 simulations, any parameter, 40 studies, warmup and
burn included. Two numbers in a cell are two sets of studies. A correct
sampler sits near 0.2 to 0.3; day 3 put the noise of one share at 0.2
between sets, not the 0.07 the last project assumed.

| sampler | mass | gradients | band at 500 | verdict |
|---|---|---|---|---|
| fixed four units | unit | 6710 | 0.25, 0.17 | passes |
| fixed two units | diagonal, 100 window | 5480 | 0.25, then 0.47, 0.38 | fails, `log tau` |
| fixed two units | diagonal, 200 window | 7480 | 0.42, 0.42 | fails, `log tau` |
| fixed two units | diagonal, 400 window | 11480 | 0.23, 0.23 | passes |
| fixed four units | diagonal, 25 window | 6960 | 1.00, 0.97 | fails, `eta_1` |
| fixed four units | diagonal, 400 window | 14460 | 0.30, 0.28 | on the line |
| NUTS | diagonal, 100 window | 8501 | 0.60, 0.42, 0.47 | fails, `log tau` |
| NUTS | diagonal, 400 window | 13711 | 0.42 | fails, `log tau` |
| NUTS | dense, 100 window | 10748 | 0.50 | fails, `log tau` |
| NUTS centred | diagonal | 11662 | 100% | fails, `log tau` |
| NUTS centred | dense | 24711 | 100% | fails, `log tau` |

The last project's tuned and jittered length passed at 4500, with 1000 of
warmup on top. Nothing under an adapted mass is cheaper than that or than
unit mass at four units. At 20 steps a draw against 40 the mass is ahead
once a chain keeps more than 338 draws; at 1000 draws it is 2920 gradients
per 99 against 4232. At SBC's 99 draws a chain it never pays.

## Key design choices

**Stan's windows, not a long run.** The question was what a blind warmup
finds, so the mass comes from the schedule a production sampler would run,
and day 4 lengthens it only by doubling.

**A mass per chain.** Each chain estimates its own, as it would in practice.
That is what dephases `mu` after half a turn on day 1, and it is why the
spread over chains is reported next to every median.

**The U-turn on velocities.** Under a mass the turn is measured along
`minv p`, and each tree was checked on a normal with known variances before
any SBC run. That check is what showed the end draw is biased under unit
mass.

**The same warmed state for every arm of a comparison.** Day 2's four runs
and day 4's lengths start from one warmup on one set of studies, so a
difference between arms is not a difference between draws.

**Two sets of studies on day 4.** Day 3 found one set could not separate
0.42 from 0.60. Reading every cell twice is what turned day 1's pass into a
fail.

**Warmup counted in the price.** The last two projects left burn-in out. A
mass that needs 8500 gradients before the first kept draw cannot be priced
that way.

**Predictions written into the files before running them.** Eight right,
three half, three wrong, three not predicted.

**Fixed seeds throughout**, with each imported module's generator reseeded.

## Where it breaks

**`log tau` has no clock and a variance is the wrong summary of it.** Its
lag-1 never goes negative at any length, its inverse mass is still climbing
at 400 transitions, and its spread over chains is more than three times
`mu`'s. The diagonal mass assumes each coordinate is a Gaussian with one
scale. `log tau` under the non-centred model has a prior-shaped tail on the
low side and a likelihood wall on the high side, and nothing here measured
which of the two the mass is still learning.

**The mass is estimated from one chain.** Every number is a per-chain mass
from at most 400 correlated draws. Production warmup pools nothing either,
but SBC has 500 chains a study on the same model, and a mass pooled over
them, or fixed at the prior's variances, would say whether the failures are
the mass's noise or the mass's form. Four units after a 25 window, 1.00 on
`eta_1`, is the noise; `log tau` after 400 may be the form.

**Forty studies move a share by 0.2.** Day 1's only pass was a draw, and the
day 2 and day 3 readings of one sampler are 0.60 and 0.42. Every "on the
line" in three projects was read against a noise of 0.07 that was never
measured. The between-set spread of the band's rejection share, and how many
studies it takes to tell 0.25 from 0.40, is owed to all of them.

**The tree fails where a shorter fixed length passes.** After the 400 window
two units passes at 0.23 and the tree, 35 steps, is 0.42 on `log tau`. The
guess in day 4 is that the tree is too long on `log tau`'s side. It was not
tested, and it is the reverse of the last project, where the tree was too
short for `mu`.

**The centred model is where it was.** Diagonal or dense, the chain sits too
high on `log tau` wherever the truth is low. A constant matrix cannot follow
a scale that changes with position, which the RMHMC project already said,
and this one only confirmed at a second price.

## Running

```bash
python day1_diagonal_mass_sweep.py
python day2_nuts_under_the_mass.py
python day3_dense_mass_and_centred.py
python day4_what_the_warmup_costs.py
```

NumPy only. Day 1 imports the trajectory length project's day 1, which puts
the eight schools, sharper rank tests and SBC modules on the path; days 2 to
4 import day 1 here, and day 4 imports day 2 for the tree. About 880, 2100,
2900 and 1300 seconds on this machine.
