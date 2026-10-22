# Eight Schools: SBC With Data in It, Centred Against Non-Centred

Where the sharper rank tests project breaks, its fourth item. Every funnel row
in two projects was a rank test against the prior, because the funnel has no
likelihood, and the funnel's two failures were told apart by the rank mean
only because one was a pile at one end and the other a symmetric U from chains
started at the prior. Eight schools is the funnel with data in it: eight
estimated effects under a shared mean and scale, a likelihood that pulls the
scale toward a neck, centred HMC's divergences the textbook failure and the
non-centred parameterisation the textbook fix. Built from scratch in NumPy on
the same unit-mass HMC and the RMHMC project's averaged metric, with SBC run
with y in it for the first time in this series.

## Idea

The model is `mu ~ N(0, 25)`, `tau` half-normal with sd 5 and sampled as
`u = log tau` with its Jacobian, `theta_j ~ N(mu, tau^2)` and `y_j ~ N(theta_j,
sigma_j^2)` with the real study's sigmas. The centred sampler carries the
thetas, the non-centred carries `eta_j` with `theta_j = mu + tau eta_j`. Day 1
checks both gradients, builds a long reference posterior on the real data and
reads both samplers against it. Days 2 to 4 run SBC: each simulation draws the
hierarchy and eight observations from the prior, starts a chain from an
independent prior draw, and ranks the truth among 99 draws of `mu`, `log tau`,
`theta_1` and, from day 3, `eta_1`, under the sharper rank tests project's
band, chi-squared, rank mean and rank variance, 40 studies of 500 simulations
each. Day 4 prices every configuration in gradient evaluations a simulation,
`thin x steps x 99`, so a pass is a cost and not only a verdict.

**Centred HMC with data in the model fails the way the funnel did, as a U on
every parameter at once, and the rank mean passes it. The non-centred fix is
correct and is not free: at thin 1 it fails the band on `mu` whenever the
trajectory is shorter than `mu`'s posterior, and passes once the trajectory is
four units long, at 1980 gradients a simulation. Past that there is no window,
six units break it on `eta` and eight on rejections. The divergence count
diagnoses less than its reputation on both sides: centred's pooled moments are
right at every step size while it diverges, one divergence costs a chain
nothing, and the funnel's metric takes the divergences to zero and still fails
at thin 1. The centred model passes once in four days, under RMHMC at eps 0.4
thin 5, for five times the non-centred price.**

## Layout

- `day1_centred_vs_noncentred.py` - both potentials and gradients against
  central differences, a 50 x 20,000 non-centred reference on the real data,
  and both samplers from prior starts at four step sizes with divergences,
  IACT and the spread of the chain means.
- `day2_sbc_with_data.py` - SBC with data drawn from the prior, both samplers
  at eps 0.1 and thin 1, 5 and 10, the four statistics on three parameters,
  and the divergence count over the prior studies.
- `day3_eta_ranks_and_divergences.py` - fresh studies, `eta_1` ranked beside
  the three parameters under both samplers, the non-centred sampler at eps 0.2
  and 0.4 unthinned, and centred's divergence count held against the band per
  study and per chain.
- `day4_step_size_and_the_neck.py` - the trajectory's length against its step
  count, centred HMC at eps 0.05 and 0.2, and RMHMC on the centred model with
  the funnel's metric averaged over the thetas, all priced in gradients.

## What the days actually show

**Day 1: the divergences are where the textbook puts them and the pooled
posterior is not.** Gradients agree to 3e-11 relative. The reference has
`log tau` mean 0.783, sd 1.106, and 19.1% of its mass below `tau = 1`.
Centred HMC diverges on 9.9%, 1.07%, 0.38% and 0.06% of proposals at eps 0.2,
0.1, 0.05 and 0.02, the non-centred on none, and centred's pooled mean `log
tau` is within 0.09 of the reference at every one of them. What it costs is
mixing, IACT 30 to 239 against 1.9 to 40 and a chain-mean sd of 0.15 to 0.78,
hidden because 50 chains started from the prior pool to the right answer.
Three right, one half, two wrong.

**Day 2: with data in it centred HMC is a U on everything, and the fix fails
too.** The band rejects every centred study at 20 simulations at thin 1, the
end tenths of `log tau` 2.18 and 2.19 and of `mu` and `theta_1` the same, and
the rank mean sits at 7% to 23% on any one parameter. Thin 10 shrinks the ends
to 1.25 and still fails at 500, IACT 40 at the median. The non-centred sampler
fails the band at thin 1 as well, 85% at 500 on `mu`'s small U, and passes at
thin 5. Divergences on 2.62% of proposals, on four chains in five. Three
right, one half, two wrong.

**Day 3: the fix fails on `mu` alone and a longer trajectory passes it.** The
-0.4 on `log tau` from day 2 was a draw, 0.00 on fresh studies. At eps 0.1
thin 1 the band rejects 97% on `mu` and 0% on `eta_1`; the 20-step trajectory
is two units and `mu` is wider. eps 0.2 passes everything unthinned, eps 0.4
diverges for the first time and fails `log tau` 97% on a U from rejections.
Under centred the derived `eta_1` is a shallow U, ends 1.40 against
`theta_1`'s 2.13, so rank both what the sampler carries and what it derives.
The count doubles between rejected and passed studies, 3.35% against 1.63%,
at a rank correlation of 0.13; one divergence costs a chain nothing on `log
tau`, 0.26 against 0.27 in the end tenths, and only the top tenth by count is
stuck, on `theta_1` first. Two right, one half, three wrong.

**Day 4: the trajectory, not the step count, and the metric removes the
divergences and not the lag.** eps 0.2 x 10, two units, fails on `mu` at 93%;
eps 0.1 x 40, four units, passes at 20% for twice the gradients of eps 0.2 x
20. eps 0.3 x 20 fails 100% at 20 on `eta_1`, ends 2.45 and 2.56, and six
units being near `eta`'s period of 2 pi is a guess. Centred HMC passes at no
step size and its U is deeper where it diverges less. RMHMC on the centred
model has no divergences at eps 0.1, 0.2 or 0.4, fails at thin 1 on `log tau`
with a U tilted low, and passes at eps 0.4 thin 5, 12% at 500. Three right,
two half, one wrong.

## What a passing study costs, in gradients a simulation

Band rejections at 500 simulations, any parameter, 40 studies, read against
the correct non-centred sampler's 12% to 28%. Gradients are `thin x steps x
99` and leave out burn-in.

| sampler | eps x steps | thin | gradients | band at 500 | verdict |
|---|---|---|---|---|---|
| non-centred | 0.2 x 10 | 1 | 990 | 93%, `mu` | fails |
| non-centred | 0.1 x 20 | 1 | 1980 | 85% to 97%, `mu` | fails |
| non-centred | 0.2 x 20 | 1 | 1980 | 12% to 23% | passes |
| non-centred | 0.3 x 20 | 1 | 1980 | 100% at 20, `eta_1` | fails |
| non-centred | 0.4 x 20 | 1 | 1980 | 97%, `log tau` | fails |
| non-centred | 0.1 x 40 | 1 | 3960 | 20% | passes |
| non-centred | 0.1 x 20 | 5 | 9900 | 5% to 28% | passes |
| centred HMC | 0.05, 0.1, 0.2 x 20 | 1 and 5 | 1980 to 9900 | 100% | fails |
| centred HMC | 0.1 x 20 | 10 | 19800 | 100% | fails |
| centred RMHMC | 0.1, 0.2 x 20 | 1 | 1980 | 100%, `log tau` | fails |
| centred RMHMC | 0.4 x 20 | 1 | 1980 | 62%, `log tau` | fails |
| centred RMHMC | 0.4 x 20 | 5 | 9900 | 12% | passes |

The cheapest pass is the non-centred sampler at four units of trajectory and
eps 0.2. The only centred pass is five times that, and thin 2 to 4 were not
run, so 9900 is an upper bound on its price and not a measurement of it.

## Key design choices

**Data drawn from the prior, chains started from a different prior draw.**
SBC with y in it is the point of the project, and an independent start is
what makes a stuck chain show as a U at both ends instead of hiding at the
truth.

**Rank what the sampler carries and what it derives.** `eta_1` beside
`theta_1` under both samplers, because day 3 found each parameterisation's
derived coordinate wrong in the opposite direction: carrying a lag its parent
did not have under non-centred, cancelling a failure its parents had under
centred.

**Price by gradients, not transitions.** A longer trajectory and a heavier
thin both buy independence, and only a count of gradients compares them.

**The metric kept explicit.** The RMHMC metric is the negative Hessian
averaged over the thetas, a function of `log tau` alone, with the one
non-vanishing cross term and the prior's term in `u`'s mass dropped so the
generalised leapfrog needs no fixed-point iteration. Both drops are named in
day 4 and the second is where it breaks.

**Predictions written into the files before running them.** Across the four
days, eleven right, five half, eight wrong, and the docstrings say which.
Day 4 was rerun before it was committed and its docstring rewritten to the
measurement.

**Fixed seeds throughout**, with each imported module's generator reseeded
so no day's rows share draws with another's.

## Where it breaks

**`u`'s mass is too heavy in the neck.** RMHMC's lag sat on `log tau`, not on
`mu` as predicted, and `u`'s mass was held at `2J`, the hierarchy's Fisher
information, with the prior's `2 tau^2 / 25` dropped to keep the step
explicit. In the neck that mass is right for the thetas and wrong for the
scale. The full metric with an implicit step, or `u`'s mass tuned on its own,
is the next thing to try, and whether either passes centred at thin 1.

**The break at six units is a guess.** eps 0.3 x 20 failed on `eta_1` with a
deep U, and `eta`'s near unit-normal posterior turns every 2 pi units under
unit mass. That predicts a failure at every multiple of the period and a pass
in between. Nothing here tested it; a sweep of trajectory length at fixed eps,
or jittered steps, would.

**Every pass was found by a sweep, not chosen.** The window at four units was
found by trying two, four, six and eight, and a real model does not come with
SBC run on it first. A rule that sets the trajectory from the chain itself,
NUTS's U-turn or a length tuned on the measured IACT of the widest parameter,
would be the honest test of whether the window can be found blind.

**Divergences mark the stuck chain and not the wrong study.** A rate that
doubles between rejected and passed studies at a rank correlation of 0.13 is
not a stand-in for the test, and a chain with one divergence was as good as a
chain with none. Whether the count is worth more on a model where the
pooled moments are actually biased, which centred eight schools never was,
was not asked.

**Eight schools and one set of sigmas.** The neck share, 19.1%, comes from
eight noisy schools under a half-normal of sd 5. More schools or smaller
sigmas move the posterior out of the neck, and every number here is at one
point on that line.

## Running

```bash
python day1_centred_vs_noncentred.py
python day2_sbc_with_data.py
python day3_eta_ranks_and_divergences.py
python day4_step_size_and_the_neck.py
```

NumPy only. Day 1 imports the MCMC project's `day1_metropolis`; days 2 to 4
import day 1 and the sharper rank tests project's `day1_ecdf_bands` and
`day2_shape_tests`, which import the SBC project's modules. About 45, 260, 300
and 650 seconds on this machine.
