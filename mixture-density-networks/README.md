# Mixture Density Networks for Multimodal Regression

A network that emits the parameters of a Gaussian mixture instead of a number,
trained by negative log-likelihood, on an inverse problem where the conditional
distribution genuinely has three modes and a squared-error regressor averages
them into an answer that is not one of them.

## Idea

Sample `t ~ U(0, 1)`, push it through `g(t) = t + 0.3 sin(2 pi t)`, add Gaussian
noise, and then use the pair backwards: the network sees `y = g(t) + eps` and is
asked for `t`.

`g` is not monotone. `g'(t) = 1 + 0.6 pi cos(2 pi t)` goes negative wherever
`cos(2 pi t) < -0.53`, so `g` folds, and a `y` inside the folded band has three
preimages. Forward, this is an ordinary regression problem and a small MLP learns
it without difficulty. Inverted, the folds become real ambiguity — the three
preimages are all equally consistent with the observation and no amount of data
distinguishes them.

The point is not that the network is inaccurate. It is that the *output object*
is wrong. A point estimate has one slot and the answer needs three, so the
question is what a point regressor does when asked for something it cannot
represent, and the answer turns out not to be the one usually given.

Synthetic on purpose. The claim being measured is about the gap between the
conditional mean and the conditional modes, and to measure it the true
conditional has to be known rather than estimated — the roots of `g(x) = y` are
computable, so every number here is checked against ground truth instead of
against another fit.

## Layout

- `day1_inverse_problem.py` — the forward map and its folds, the root finder that
  serves as ground truth, the MSE and MAE point regressors, and the accounting
  for where their predictions land relative to the true preimages.
- `day2_mixture_head.py` — the mixture head with softmax weights and floored
  softplus scales, the log-sum-exp NLL checked against a brute-force integration
  of its own density, and the four numerical traps in that loss pinned with
  measurements rather than asserted.
- `day3_training_calibration.py` — training, the component-death study across
  seeds, where in the input range a dead component actually costs anything,
  scale collapse, modes against true preimages, PIT calibration and its limits,
  and ancestral sampling from the predictive mixture.

## What the days actually show

**Day 1: the point regressor is not visibly broken, which is worse.** The usual
telling is that squared error lands in the valley between the modes, at a point of
near-minimal density. It does not happen here. Measured across the ambiguous band
the fitted regressor sits a median of 0.012 from a genuine preimage, and always
the same one — at any preimage the density is `exp(0)`, so all three peaks have
identical height and what separates them is *width*. The bump at a root has scale
`sigma / |g'|`, so the flattest branch carries the most mass, and that is the
folded middle one at every `y` in the band. The conditional mean is pulled onto
the most probable branch and stays there.

So the estimate is a legitimate preimage and looks correct under every point
metric, which means no point metric will catch it. What it cannot do is say the
other two preimages exist, and they hold 56% of the conditional mass. Ruled out
rather than asserted: not the optimizer or capacity (the fit agrees with a
nonparametric binned estimate of `E[t | x]` in units of each bin's own standard
error), not the loss (L1 has a different minimizer and lands on the same branch),
and not the fit in general (three sigma clear of the fold the same network is
accurate to 0.02).

**Day 2: `logsumexp` fixes outliers, not dead components.** The standard argument
for the stable mixture NLL is that it keeps far components alive. The first half —
that the naive form underflows — is true, and it starts returning `inf` at
`sigma = 0.03`, which is *inside* the range of scales a correct fit on this data
wants, so it survives initialization and breaks partway through training. The
second half is false. With a component 0.48 from the target at `sigma = 0.02`, its
log-responsibility is -288 and `exp(-288)` is exactly zero in float32 under both
forms; the stable version hands that component a gradient of `-0.0`, not a small
one. What has underflowed is the softmax responsibility, and the responsibility
*is* the gradient. What `logsumexp` actually rescues is the different event where
*every* component is far from the target — an outlier point, which under the naive
form produces `nan` gradients on all three means and destroys the batch.

Two more, both measured rather than asserted: the scale floor is not an
epsilon-for-stability but the fix for a genuinely unbounded objective, and the
descent into it is a constant-gradient ramp of slope 1 in the unconstrained
parameter rather than a cliff, so a smaller learning rate only slows the walk down
it. And `log(softmax(z))` versus `log_softmax(z)` agree in the forward pass right
up to where one is `-inf`, which `logsumexp` absorbs without complaint, so the
loss stays finite and correct and only the *backward* pass is `nan` — a clean loss
curve followed by `nan` parameters one step later.

**Day 3: component death is decided in the first 25 epochs.** Across six seeds
with default initialization, every run loses at least one component — 7 of 18
dead, where dead means the component is never the most probable explanation of any
point. Spreading the mean head's bias across the target range, one line, loses
none across the same seeds and buys 0.35 nats. The timing is the content: all
three components start with peak responsibility near 0.4 and by epoch 25 the loser
is at 0.000. This is not slow starvation that a longer schedule would fix. At
initialization the components sit within about `1/sqrt(hidden)` of zero and
overlap almost exactly, the first few noisy updates settle which one claims which
branch, and after that the loser has no gradient.

**Day 3: the damage is not where the missing mode is.** The obvious guess is that
a dead component costs nats inside the folded band and nothing outside it.
Measured, it is the other way round — 0.41 nats outside against 0.30 inside, on a
total of 0.35. A surviving component gets stretched to cover the dead one's work
and is therefore too wide everywhere, and NLL punishes a wrong width in proportion
to how peaked the truth is: outside the band the conditional is a single sharp
bump where nats are expensive, inside it is diffuse where they are cheap. The
region that does not need three modes is where the loss notices they are missing,
which makes a per-region loss breakdown a bad instrument for locating a structural
failure. The responsibilities are the good one.

**Day 3: pooled calibration hides the miscalibration.** The probability integral
transform over all 3000 points has mean 0.495, standard deviation 0.288 against a
uniform's 0.2887, and a decile chi-square of 6.7 on 9 degrees of freedom — a clean
pass. Split on the folded band, the same values give 22.0 inside and 23.1 outside,
both past the 1% point. The halves are biased in opposite directions, mean 0.506
inside against 0.484 outside, and they cancel. Nothing about the pooled number is
wrong; it is a correct answer about the marginal distribution, and marginal
calibration is much weaker than it sounds. Central coverage, which is what an
interval check reports, is weaker still — 0.796, 0.894, 0.955 against nominal
0.80, 0.90, 0.95, collapsing ten deciles into one number.

**Day 3: the noiseless roots undercount the modes.** Day 1 flagged that the
noiseless preimage count is the wrong boundary once the problem has noise in it.
Today it bit. The first mode check compared learned modes against the exact roots
of `g(x) = y` and flagged a learned mode at 0.35 for `y = 0.62` as spurious — but
`g(0.35) = 0.593` is 0.027 from the observation, well inside a noise scale of
0.05, so that branch carries real mass and the root finder cannot see it because
it answers a zero-noise question. Against the analytic noisy conditional the model
matches mode for mode to within 0.019 at every probe, and three of the five probes
lie outside the folded band while being genuinely bimodal. The band is a fact
about `g`, not about how many answers the data admits.

Sampling closes the loop. Ancestral draws at the centre of the band put 21%, 31%
and 22% of their mass within 0.06 of the three preimages, and the sample mean
lands at 0.496 — day 1's answer, reproduced exactly, by a model that is no longer
obliged to report it.

## Where it breaks

The dataset is one-dimensional in both directions and the mixture is diagonal by
construction. In higher dimensions the component count needed to cover a
conditional grows with the dimension, and the parameter count of a full covariance
grows quadratically on top of that, which is where the practical versions of this
switch to diagonal or low-rank covariances and pay for it in fidelity.

The spread-bias fix works here because the target range is known and small.
Initializing means across the output range is not available when the output range
is unknown or unbounded, and the more general answers — a weight-entropy penalty,
EM-style responsibility warm-up, or simply restarting and keeping the best
likelihood — are not tested here. What is established is narrower: death is an
initialization event on this problem, so an intervention has to act at
initialization to be relevant at all.

Everything here is aleatoric uncertainty. The mixture reports the spread of the
data, and it reports it with exactly the same confidence in a region where there
was no training data at all. Nothing in the NLL objective distinguishes "the
answer is genuinely ambiguous" from "I have never seen an input here", and the
PIT cannot either, since it only scores points that exist. That is the next
project rather than a caveat on this one.

And `K = 3` is chosen knowing the answer. Choosing it from data means either a
model-selection pass or a mixture with more components than needed and a penalty
that prunes them, and the second one interacts directly with day 3's finding —
extra components make death more likely, not less.

## Running

```bash
python day1_inverse_problem.py
python day2_mixture_head.py
python day3_training_calibration.py
```

Requires PyTorch and NumPy. Days 1 and 2 run in under a minute. Day 3 takes about
fifteen — the death study trains twelve networks to convergence, which is the
measurement rather than an accident of the schedule, since the whole question is
what varies across initializations.

Every file asserts its own claims at the bottom, so a silent exit is the pass
condition. The numbers are printed as well, because most of what is worth
reporting here is a magnitude rather than a pass or a fail.
