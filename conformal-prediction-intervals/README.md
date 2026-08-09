# Conformal Prediction for Distribution-Free Uncertainty

Split conformal prediction written from scratch in NumPy. Nonconformity scores
on a held-out calibration set, the `(n+1)`-corrected quantile that turns them
into a finite-sample coverage guarantee, conformalized quantile regression for
intervals that adapt their width, prediction sets for classification, and a
final experiment that breaks the one assumption the whole thing rests on.

## Idea

Most uncertainty estimates in machine learning are claims about the model rather
than about the world. A Gaussian process posterior variance is right if the
kernel and noise model are right. A bootstrap interval is right if resampling
reproduces the sampling distribution. Both degrade quietly when the assumption
behind them is wrong, and neither reports that it happened.

Conformal prediction makes a much narrower promise and keeps it. Given a fitted
model `f` and a calibration set the fit never saw:

    1. score each calibration point by how badly f missed it:  s_i
    2. take q = the k-th smallest score, k = ceil((n + 1) * (1 - alpha))
    3. return the set of labels whose score would be at most q

Then `P(y_test in C(x_test)) >= 1 - alpha`, in finite samples, for any base
model, any data distribution, and any `n`. No asymptotics, no well-specified
likelihood, no assumption that the model is any good.

The whole guarantee lives in the `+1`. Under exchangeability the test score is
equally likely to occupy any of the `n+1` positions in the sorted combined
sample, so `P(s_test <= k-th smallest) >= k / (n + 1)`, and the ceiling is the
smallest `k` reaching `1 - alpha`. Using the ordinary `k / n` quantile is
asymptotically fine and wrong by `O(1/n)` — at `n = 50, alpha = 0.1` that is the
difference between 90% and about 88%.

Everything else in the project is a choice of score `s`. The conformal step is
imported unchanged from day 1 into every subsequent file, three times, on scores
that are a residual, a signed distance, and a cumulative probability mass. That
is the actual content of "any nonconformity score works": the base method
supplies the *shape* of the prediction, conformal supplies the *validity*, and
the two never negotiate.

## Layout

- `day1_split_conformal.py` — ridge regression on heteroscedastic data,
  absolute-residual scores, `conformal_quantile` and where the `+1` comes from,
  coverage across alpha, the guarantee holding under deliberately broken base
  models, and the conditional-coverage failure that motivates day 2.
- `day2_conformalized_quantile_regression.py` — quantile regression by
  subgradient descent on the pinball loss, the CQR score
  `max(q_lo - y, y - q_hi)`, and the bin-by-bin comparison against day 1's
  constant-width band.
- `day3_conformal_classification.py` — softmax regression on overlapping
  classes, prediction sets from LAC and APS scores, randomized APS, coverage vs
  nominal swept across alpha, and the covariate-shift experiment that breaks
  exchangeability on purpose.

## What the days actually show

**Day 1: marginal validity is real, and it is not what you wanted.** Coverage
lands on nominal at every alpha (0.925 at target 0.90, 0.975 at 0.95, 0.995 at
0.99), and stays there when the base model is replaced by a constant predictor
or a degree-20 polynomial — only the width moves. Broken down by `|x|`, the same
intervals cover 100% of the low-noise bin and 80.1% of the high-noise bin
against a nominal 90%. Nothing is broken. Marginal coverage is an average over
the draw of the test point, and a procedure can satisfy it while being wrong
everywhere in a way that cancels.

**Day 2: adapting the width costs nothing in validity.** CQR conformalizes an
interval instead of a point, so the band is already wider where the noise is.
Coverage spread across the `|x|` bins drops from 0.199 to 0.044 while marginal
coverage is unchanged (0.9018 vs 0.9078 over 100 independent splits). The CQR
score can also go *negative* — when the quantile models are over-wide, `Q < 0`
and the conformal step tightens the band rather than inflating it (here
`Q = -0.010`). An absolute residual has no way to express "this was too
conservative".

**Day 3: the label does not have to be a number.** The same quantile applied to
a classification score returns a *set* of labels, and the interesting quantity
becomes cardinality rather than width. Set size is a per-point difficulty
readout that a softmax argmax discards. At `alpha = 0.1` on a 4-class problem
where the base model gets 84%:

    method      threshold   coverage  mean size    empty  singleton
    lac             0.718      0.887      1.142    0.000      0.858
    aps             0.996      0.902      1.776    0.042      0.363
    aps-rand        0.901      0.891      1.322    0.029      0.658

All three are valid. They differ entirely in what they charge for it.

## Key design choices

- **`conformal_quantile` returns `inf`, not the maximum score.** When
  `k > n` — which happens whenever `n < 1/alpha - 1` — the recipe asks for an
  order statistic that does not exist. 19 calibration points cannot certify 95%
  coverage. Clamping to the largest observed score is a quiet way to break the
  guarantee and is the failure mode to watch for whenever alpha is small or the
  calibration split is.
- **Three splits, never two.** Scoring on training residuals makes them
  optimistically small by an amount that depends on how much the model overfits,
  so it is not a fixed bias that could be corrected afterwards. The calibration
  set exists to be unseen by the fit, which is the only reason its scores are
  exchangeable with the test score.
- **The APS set rule is matched to the APS score term for term.** The true label
  is in the set exactly when its own score is at or below the threshold, so no
  step of the coverage argument is an approximation. Written the other common
  way — "smallest prefix whose mass *reaches* q" — the set is one class larger
  than the score justifies and coverage comes out at 1.000 against a nominal
  0.90: valid, and useless.
- **Randomized APS is about size, not coverage.** The deterministic score moves
  in jumps the size of a class probability, which drags the quantile up to 0.996
  and pulls a trailing low-probability class into nearly every set. Discounting
  the label's own mass by a uniform draw fills in the gaps between the jumps and
  recovers about half a class per prediction at identical coverage. The price is
  a non-deterministic output for a fixed input.
- **Empty prediction sets are left alone.** An empty set is the method reporting
  that no label is plausible for this point — a genuine signal about the point.
  Substituting the argmax hides it and repairs nothing, since the guarantee
  already accounts for empty sets never containing the truth.

## Where it breaks

Every guarantee above is conditional on the calibration scores and the test
score being exchangeable, and that is an assumption about the world rather than
a property of the method. The last experiment in day 3 breaks it deliberately:
calibration data is unshifted, test points are drawn with the covariate
distribution translated, everything else identical.

    shift      lac cov      lac size aps-rand cov aps-rand size  accuracy
      0.0        0.900          1.15        0.899          1.32     0.844
      0.5        0.879          1.14        0.885          1.31     0.826
      1.0        0.844          1.15        0.852          1.32     0.783
      1.5        0.794          1.16        0.805          1.33     0.732
      2.0        0.733          1.15        0.762          1.34     0.672

The sets do not widen. They cannot — the threshold was fixed by calibration data
that never saw the shift, so the only thing free to move is how often the truth
falls outside it. Coverage decays smoothly from 90% to 73% while the procedure
reports nothing at all, and no quantity computable from the calibration set
detects it.

This is the honest boundary of the method rather than a caveat attached to it.
Conformal prediction converts exchangeability into a finite-sample guarantee; it
cannot manufacture one when the input assumption is false. Weighted and online
variants exist for shift that is known or estimable, and none of them remove the
need to know something about it. Knowing the size of the gap is arguably more
useful than the guarantee itself, because in deployment the shift is the default
case and exchangeability is the special one.

## Running

```bash
python day1_split_conformal.py
python day2_conformalized_quantile_regression.py
python day3_conformal_classification.py
```

Requires NumPy and nothing else. The multi-split studies at the bottom of each
file refit the base model a few dozen times and take under a minute each; the
quantile regression in day 2 is explicit subgradient descent rather than a
linear-programming solver, which is slower but keeps the pinball loss visible.
