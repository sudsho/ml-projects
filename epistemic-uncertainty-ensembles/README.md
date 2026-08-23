# Epistemic Uncertainty: Deep Ensembles, MC Dropout and Laplace

The half of predictive uncertainty a density head cannot represent, built from
scratch in PyTorch: the aleatoric/epistemic decomposition on a regression problem
with a hole punched in its input range, deep ensembles, MC dropout with its
lengthscale correspondence, a last-layer Laplace approximation via the
Gauss-Newton Hessian, and a final day of scoring all four predictive
distributions with proper scoring rules on held-out data drawn from inside the
hole.

## Idea

The MDN project ended with a model that reports the shape of the conditional
distribution correctly and says nothing about whether it has seen data at all. At
an input that never appeared in training it still emits a perfectly ordinary set
of parameters, produced by whatever the hidden layers extrapolate to, with no
marker on them saying they were made up.

The two quantities have names. Split the predictive variance with the law of
total variance over a posterior on weights:

    Var[y | x]  =  E_theta[ sigma^2(x, theta) ]  +  Var_theta[ mu(x, theta) ]
                   \-------- aleatoric --------/    \------- epistemic ------/

The first term is noise in the data-generating process. It is irreducible, and it
is the only one a heteroscedastic head or an MDN models. The second is
disagreement between the models the data has not ruled out - reducible, and the
one that is supposed to grow where there is no data.

**A single network sets the second term to zero identically.** Not approximately:
one point estimate of `theta` makes `Var_theta` a variance over a point mass. So
the problem is the wrong output object rather than a badly fitted number, and the
three methods here are three ways of manufacturing something to take a variance
over.

The experiment is a one-dimensional regression on `[-4, 4]` with known
heteroscedastic noise and no training data in `(-1.5, 1.5)`. Because the gap is
cut rather than found, the generative process can still be sampled inside it, so
every method can be scored on held-out points in a region with zero training
support. That is the one thing here that could not be done on a real dataset and
it is what the whole design is for.

Every method is measured against a **control**: the same method trained on data
with the gap filled in. Whatever it reports in the gap there is that method's
floor, because the missing data is no longer missing. A method whose in-gap
reading does not clear its own floor is reporting a property of itself.

## Layout

- `day1_the_gap.py` - the dataset and its deliberate hole, a heteroscedastic
  Gaussian head trained by NLL, the decomposition, and the demonstration that a
  single network's reported sigma in the gap sits inside the range it takes on the
  training data.
- `day2_deep_ensembles.py` - ten independent fits, the mixture their predictions
  form, mixture NLL by log-sum-exp, mode counting, how the estimate moves with
  ensemble size, and the control that says whether the disagreement is the missing
  data or the optimizer.
- `day3_dropout_and_laplace.py` - MC dropout as approximate variational inference,
  the dropout-rate/lengthscale correspondence and a sweep testing it, and a
  last-layer Laplace approximation with the Gauss-Newton Hessian, which is exact
  for this loss rather than an approximation.
- `day4_scoring.py` - NLL, CRPS and PIT calibration on held-out data inside and
  outside the gap, closed-form CRPS for Gaussian mixtures, calibration curves, and
  what each method costs in training runs.

## What the days actually show

**Day 1: the gap has to be wide enough to not be interpolable, and the first one
was not.** A hole a network can smoothly bridge is not a region without
information - the function's own regularity carries the answer across it, and the
model is right there for a reason that has nothing to do with uncertainty. The
gap was widened until the target stopped being recoverable from its edges, and
only then was the measurement about anything.

The aleatoric half works: the head recovers the noise scale where there is data.
That is what makes the rest attributable. In the gap it reports the same sigma it
reports on data, the mean is wrong, and the standardized residuals blow up - so
the head is reporting the aleatoric term correctly in the one place where
reporting it is the entire error.

**Day 2: the ensemble's disagreement is real and too small.** Ten members from
different initializations disagree in the gap at 18x what the filled-data control
produces, so it is the missing data talking and not SGD noise. It also points the
right way, peaking where the aleatoric term bottoms out.

And the worst mean error in the gap sits at about 2.5 reported standard
deviations. The signal is genuine and the scale is wrong, which are separate
claims, and the control is what separates them.

**Day 3: MC dropout detects nothing, and both predictions were backwards.** The
in-gap reading moves 1.8x across a sweep of dropout rates, the control's floor
moves by the same factor, and the ratio of the two never leaves 1.0. At `p = 0.20`
and `p = 0.35` the model trained on data with no hole reports *more* in-gap
disagreement than the one trained on data with a hole. What MC dropout measures is
the variance the masks inject, which is present everywhere in equal measure.

The number that would have fooled me is `0.1465` at `p = 0.10` - half the
ensemble's reading and the same order, exactly what a slightly-worse-but-working
method looks like. The control returns `0.1381` for the same setting. Day 2 built
that control and day 3 is the first time it killed something, and what it killed
was an ordinary-looking number, not an outlier.

Last-layer Laplace was predicted to under-report because the feature map is
frozen. It separates the regions nearly twice as sharply as the ensemble - a ratio
of 19.4 against 10.3, a signal of 48x against its own control - for one training
run and one 65x65 solve. The frozen-feature objection is an argument about
high-dimensional inputs where a point far away in input space can still land on
covered features. This input is one-dimensional, so the feature curve genuinely
goes somewhere new in the gap, and one dimension is the regime where the last
layer is enough.

**Day 4: the two proper scoring rules rank the methods in opposite orders.**

    in-gap ranking by NLL : laplace  < dropout < ensemble < single
    in-gap ranking by CRPS: ensemble < dropout < single   < laplace

Laplace wins NLL by 0.49 nats and loses CRPS to the ensemble by 54%. Both scores
are proper. Neither is a proxy for the other, and three days of this project used
"proper scoring rule" as though it named one thing.

They split because NLL is unbounded below and is dominated by points where a
method put almost no mass on what happened - the single head scores 13.27 in the
gap with a 99th percentile of 56.07, and that tail is the whole number. CRPS is an
integrated squared CDF error in the units of the target and is bounded by roughly
the size of the miss. So NLL asks whether the outcome was admitted to be possible
and CRPS asks how far off the prediction was, and in a region with no data those
are different questions.

## The honest method is the inaccurate one

Mean absolute coverage gap in the gap region: Laplace 0.062, ensemble 0.287,
dropout 0.347, single 0.417. Laplace's central intervals in the gap land at 0.50
nominal / 0.50 empirical and 0.90 / 0.81. The ensemble's 90% interval covers 57%.
The single head's 90% interval covers 19% of the points it was asked about.

Mean absolute error in the gap: ensemble 0.4365, single 0.5443, dropout 0.5826,
Laplace 0.6730.

So the only method that is calibrated in the gap is the least accurate in it, and
it is last by the score that measures accuracy. Those are not in tension; they are
the same fact seen twice. Laplace is honest about a worse mean, and a scoring rule
in the units of the target charges it for the mean.

**Most of the CRPS ordering is the mean ordering.** The two agree except that
dropout and the single head swap - dropout has the worse mean and the better CRPS,
because it has some spread and the single head has essentially none. That is CRPS
working as advertised: it rewards calibrated spread, and much less than it rewards
a good mean.

Which means the ensemble wins CRPS largely because averaging ten independently
initialized means is variance reduction on the mean itself. That is a real benefit
of ensembling with nothing to do with uncertainty quantification, and three days
of treating the ensemble as an epistemic-uncertainty method ended with its best
score earned by its point prediction.

## The diagnostic was one of two available rankings

Days 1 to 3 ranked methods by the in-gap/on-data ratio of reported epistemic
standard deviation, against the filled-data control. That diagnostic is mine.
Nobody optimizes it and it has no decision-theoretic standing.

It put Laplace first, the ensemble second, dropout nowhere - which is the NLL
ordering and the calibration ordering, and the reverse of CRPS on its top and
bottom entries. The diagnostic was never method-neutral: it measures whether
reported spread grows where data stops, which is what NLL rewards and what CRPS
mostly ignores. It looked like a general-purpose ranking for three days and it was
one of two, chosen before I knew there were two.

## Cost

| method | training runs | wall time | wins |
| --- | --- | --- | --- |
| single Gaussian head | 1 | 5s | nothing |
| deep ensemble (M=10) | 10 | 49s | CRPS, mean error |
| MC dropout (p=0.10) | 1 | 6s | second on both, detects nothing |
| last-layer Laplace | 1 | 5s | NLL, calibration |

MC dropout's row is the combination that would have got it adopted: it looks
reasonable by every number in the table and the number that tests it is the one
nobody computes.

## Key design choices

**The control is the whole instrument.** Every in-gap reading is compared against
the same method trained with the gap filled in. Without it, MC dropout's `0.1465`
is a working method.

**Scoring the mixture, not its Gaussian summary.** The ensemble's predictive
distribution is a ten-component mixture, and the Gaussian sharing its mean and
variance is a different object that scores better exactly where the members
disagree - which is the region under study. Day 2 computes mixture NLL by
log-sum-exp and day 4 uses the closed-form mixture CRPS, so the summary is for
reporting and never for scoring.

Laplace is the exception and legitimately so: the mean is linear in the last-layer
weights, so the pushforward of the weight posterior is exactly Gaussian and adding
the aleatoric variance keeps it Gaussian. Its one-component representation is the
predictive rather than a summary of it.

**Calibration through the PIT.** Coverage is read off `F(y)` rather than from
quantiles, which works unchanged for a mixture - a mixture's quantile function has
no closed form and its CDF does.

**`sigma_i` is fixed in the Laplace Hessian.** Putting a posterior on the variance
head as well would mix the two terms back together in the one place the design is
trying to keep them apart.

## Where it breaks

**One confound survives.** The single head trains without weight decay and the
Laplace model trains with it, because the Gauss-Newton Hessian needs a prior
precision to be a posterior at all. So the mean-error difference between those two
is a regularization difference and not anything Laplace did - Laplace does not
touch the mean, it adds an epistemic term to a fit that already exists. The clean
comparison is Laplace against the `rate=0.0` dropout model, which is the same fit.

**One dimension.** The whole reason last-layer Laplace works here is that
`phi(x)` traces a curve through `R^64` and the gap is an uncovered arc of it. In
high dimensions the data manifold is thin, a point far away in input space can
land on covered features, and the frozen-feature objection stops being about the
wrong regime. Nothing here tests that.

**One gap, one seed per method.** The ensemble is ten runs, but every other row is
a single fit at one seed. The day 3 sweep varies the dropout rate and the Laplace
prior precision and nothing varies the initialization, so the spreads reported for
the one-run methods have no error bars on them.

**Laplace has a free parameter too.** Two decades of prior precision move the
in-gap standard deviation by 6x and the ratio by 5x. The method that works also
has a knob setting how much it works, and the hyperparameter argument that killed
MC dropout moves house rather than disappearing.

## Running

```bash
python day1_the_gap.py
python day2_deep_ensembles.py
python day3_dropout_and_laplace.py
python day4_scoring.py
```

Requires PyTorch and NumPy. No scipy - the normal CDF comes from `torch.erf`.
Day 1 takes about 15 seconds, day 2 about a minute for the ten members, day 3
about three minutes for the rate sweep and its controls, day 4 about 70 seconds
for twelve training runs plus scoring.

Every file asserts its own claims at the bottom rather than printing numbers for
inspection, so a silent exit is the pass condition. The asserts are written to
fail if a finding reverses, including the uncomfortable ones - day 3 asserts an
upper bound on MC dropout's signal, and day 4 asserts that the best-calibrated
method is still the least accurate.
