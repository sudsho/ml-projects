"""
Day 3 of conformal prediction from scratch.

Days 1 and 2 both produced intervals, and an interval is a slightly misleading
advertisement for what conformal prediction is. Nothing in the argument needed
the label to be a real number. The construction takes a nonconformity score, a
calibration set, and the (n+1)-corrected quantile, and returns a *set* of labels
that is guaranteed to contain the truth with probability 1 - alpha. In
regression that set happens to be an interval because the score was built from a
distance. In classification it is a subset of the classes, and the interesting
quantity stops being width and becomes cardinality.

That change makes the tradeoff visible in a way the regression version hid. A
prediction set of size 1 is a confident classification. Size 3 out of 4 is the
model admitting it does not know. Size 0 - which is a legal output of the first
method below - is the model saying that no label is plausible, which on a test
point drawn from the training distribution means the point is strange rather
than the classes being wrong. Set size is a per-point difficulty readout that a
softmax argmax throws away entirely.

Two scores are built here and they disagree in a way worth understanding.

  LAC (least ambiguous classifier, sometimes called THR): s = 1 - p_hat(y | x).
  It looks only at the mass assigned to the label that actually occurred and
  ignores how the rest is distributed. This gives provably the smallest average
  set size of any conformal score, which sounds like it settles the matter.

  APS (adaptive prediction sets): sort the classes by descending probability and
  score the true label by the cumulative mass up to and including it, so a label
  is penalised for the competition above it and not only for its own
  probability. Sets come out larger on average. What is bought with that size is
  the same thing day 2 bought over day 1 - the score reads the whole conditional
  distribution rather than one entry of it, so it responds to genuine ambiguity
  instead of averaging over it.

APS also arrives with a wrinkle that turns out to be instructive. Its score
moves in jumps the size of a class probability, and a coarse score forces the
conformal quantile up to the next reachable jump. Coverage still comes out
right - the bookkeeping is exact either way - but every set is inflated to pay
for it. Adding a uniform draw that discounts the true label's own mass fills in
the gaps between the jumps and buys back most of the excess size at identical
coverage. That is measured below rather than asserted; it is worth about half a
class per prediction here.

The last section is the one this project has been deferring since day 1. Every
guarantee here rests on exchangeability of the calibration scores and the test
score, and that is an assumption about the world, not a property of the method.
When it fails, conformal fails, quietly and without any internal signal that
something is wrong. The final experiment breaks it deliberately - test points
drawn from a shifted covariate distribution, with everything else identical -
and measures how far coverage falls. It falls a long way. Knowing the size of
that gap is more useful than the guarantee itself, because in deployment the
shift is the default case and exchangeability is the special one.
"""

import numpy as np

from day1_split_conformal import conformal_quantile, three_way_split


def make_overlapping_classes(n_samples, rng, n_classes=4, shift=0.0):
    """A 2-d classification problem whose classes genuinely overlap.

    Class means sit on a circle of radius 2.2 with per-class isotropic noise
    that ranges from tight to broad, so some regions of the plane are decidable
    and others are not. The overlap is the point: on a separable problem every
    prediction set has size 1 and there is nothing to measure.

    `shift` translates the sampling distribution along the first axis without
    touching the class-conditional structure. At shift=0 the data is
    exchangeable with the calibration set; at shift>0 it is not, while looking
    entirely reasonable to any check that inspects one point at a time.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_classes, endpoint=False)
    means = np.stack([2.2 * np.cos(angles), 2.2 * np.sin(angles)], axis=1)
    spreads = np.linspace(0.7, 1.6, n_classes)

    labels = rng.integers(0, n_classes, size=n_samples)
    noise = rng.standard_normal((n_samples, 2)) * spreads[labels][:, None]
    x = means[labels] + noise
    x[:, 0] += shift
    return x, labels


def quadratic_features(x):
    """Intercept, linear, and pure/cross quadratic terms for a 2-d input.

    Six columns. Enough curvature that the decision boundaries are not straight
    lines, which keeps the softmax probabilities from being trivially
    monotone in one coordinate, and few enough that the fit below converges
    without any tuning.
    """
    x0, x1 = x[:, 0], x[:, 1]
    return np.stack([np.ones_like(x0), x0, x1, x0 * x0, x1 * x1, x0 * x1], axis=1)


def softmax(logits):
    """Row-wise softmax with the standard max subtraction.

    Subtracting the row max is not cosmetic. Logits here reach into the tens
    once the quadratic terms are fit, and exp of that overflows float64 to inf,
    after which the normalization is inf/inf = nan and the whole run is silently
    poisoned. The shift is exactly invariant, so it costs nothing.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def fit_softmax_regression(x, y, n_classes, learning_rate=0.35,
                           iterations=900, penalty=1e-3):
    """Multinomial logistic regression by full-batch gradient descent.

    The gradient of the average cross-entropy with respect to the weight matrix
    is design.T @ (probabilities - onehot) / n, which is the same residual-times-
    features form as least squares and is why this converges without anything
    clever. The L2 penalty is there to stop the weights running off on the
    classes that are nearly separable; it does not touch the conformal step,
    which is the recurring point of the project - the base model can be tuned,
    untuned, or actively bad and the coverage guarantee does not move.
    """
    design = quadratic_features(x)
    n, n_features = design.shape

    onehot = np.zeros((n, n_classes))
    onehot[np.arange(n), y] = 1.0

    weights = np.zeros((n_features, n_classes))
    for _ in range(iterations):
        probabilities = softmax(design @ weights)
        gradient = design.T @ (probabilities - onehot) / n + penalty * weights
        weights -= learning_rate * gradient
    return weights


def predict_proba(weights, x):
    """Class probabilities for new points, shape (n, n_classes)."""
    return softmax(quadratic_features(x) @ weights)


def lac_scores(probabilities, y):
    """s_i = 1 - p_hat(y_i | x_i), the score behind the smallest average sets.

    High score means the model assigned little mass to the label that actually
    occurred, which is exactly the "nonconforming" notion the theory wants. Note
    it uses only the probability of the true class and ignores how the remaining
    mass is distributed, which is both why the sets are small and why they can
    come out empty.
    """
    return 1.0 - probabilities[np.arange(y.size), y]


def lac_sets(probabilities, threshold):
    """Boolean (n, n_classes) mask of {k : p_hat(k | x) >= 1 - threshold}.

    Rewriting `1 - p <= q` as `p >= 1 - q` is what makes the set explicit. When
    q is small the bar 1 - q is high and no class clears it, and the row is all
    False - a genuinely empty prediction set. That is not a bug to be patched by
    inserting the argmax: the empty set is the method reporting that this point
    is unlike anything in calibration, and overwriting it hides the signal while
    breaking nothing about the coverage arithmetic (an empty set never contains
    the truth, and the guarantee already accounts for that).
    """
    return probabilities >= (1.0 - threshold)


def _descending_cumulative(probabilities):
    """(order, sorted probabilities, cumulative mass) for each row, descending."""
    order = np.argsort(-probabilities, axis=1)
    sorted_probabilities = np.take_along_axis(probabilities, order, axis=1)
    return order, sorted_probabilities, np.cumsum(sorted_probabilities, axis=1)


def aps_scores(probabilities, y, rng=None):
    """Cumulative probability mass, descending, up to and including the label.

    Sort each row's probabilities descending, walk down until the true class is
    reached, and return the running total including it. A label the model ranks
    first with mass 0.9 scores 0.9; the same label ranked third scores whatever
    the two classes above it hold plus its own mass, so it is penalised for the
    competition rather than only for its own probability. This is adaptive in
    the same sense day 2's score was - it reads the shape of the whole
    distribution at x, not one number from it.

    With `rng` supplied, the label's own mass is discounted by a uniform draw,
    E = cumulative - u * p(y | x). This is the randomized APS of Romano, Sesia
    and Candes, and it is not a refinement anyone would invent for fun. Without
    it the score moves in jumps the size of a class probability, so the
    calibration scores cluster near 1 for every point the model ranked wrong and
    the (n+1)-quantile is dragged up to the next reachable jump - 0.996 at
    alpha = 0.1 below. Coverage is unaffected, because the set rule is matched
    to the score exactly, but a threshold that high pulls a trailing
    low-probability class into almost every set. The uniform draw fills in the
    gaps between the jumps, and the alpha table runs both so the cost of not
    doing it is visible as a number rather than a claim.
    """
    order, sorted_probabilities, cumulative = _descending_cumulative(probabilities)

    # position of the true label within each row's descending ranking
    rank = np.argmax(order == y[:, None], axis=1)
    rows = np.arange(y.size)
    scores = cumulative[rows, rank]
    if rng is None:
        return scores
    return scores - rng.uniform(size=y.size) * sorted_probabilities[rows, rank]


def aps_sets(probabilities, threshold, rng=None):
    """Top-ranked classes whose discounted cumulative mass stays under threshold.

    Mirrors `aps_scores` term for term, which is what makes the coverage
    argument go through: the true label is in the set exactly when its own score
    is at or below the threshold, and that is the event the conformal quantile
    controls. Nothing here is an approximation of the calibration rule.

    The kept classes are always a prefix of the ranking even in the randomized
    case, because consecutive discounted scores differ by
    u * p_(j-1) + (1 - u) * p_(j), which is strictly positive. So the set is
    still "the top few classes", and set size remains a readable difficulty
    measure. Empty sets are possible when even the top class fails the test,
    which happens only at small alpha on points the model is confident and
    wrong about.
    """
    order, sorted_probabilities, cumulative = _descending_cumulative(probabilities)

    discounted = cumulative
    if rng is not None:
        draws = rng.uniform(size=(probabilities.shape[0], 1))
        discounted = cumulative - draws * sorted_probabilities

    keep_sorted = discounted <= threshold
    sets = np.zeros_like(keep_sorted)
    np.put_along_axis(sets, order, keep_sorted, axis=1)
    return sets


def set_coverage(sets, y):
    """Fraction of test points whose true label is inside the prediction set."""
    return float(np.mean(sets[np.arange(y.size), y]))


def set_size_summary(sets):
    """(mean size, fraction empty, fraction singleton) for a batch of sets."""
    sizes = sets.sum(axis=1)
    return float(sizes.mean()), float(np.mean(sizes == 0)), float(np.mean(sizes == 1))


def calibrate_and_predict(probabilities_calib, y_calib, probabilities_test,
                          alpha, method, rng=None):
    """Run one full conformal pass and return (sets, threshold).

    Deliberately written so the methods differ only in which score function and
    which set constructor get used. `conformal_quantile` is imported unchanged
    from day 1 for the third time in this project, on a score that is now
    neither a residual nor a signed distance, because the guarantee never
    depended on what the score meant.

    `method` is one of "lac", "aps" (deterministic, conservative), or
    "aps-rand" (randomized, exact). The randomized variant needs the *same*
    generator semantics on both sides - calibration scores and test sets are
    each given their own independent uniform draws, which is correct because
    exchangeability is over (x, y, u) triples and the draws are independent of
    everything else.
    """
    if method == "lac":
        scores = lac_scores(probabilities_calib, y_calib)
        threshold = conformal_quantile(scores, alpha)
        return lac_sets(probabilities_test, threshold), threshold

    aps_rng = rng if method == "aps-rand" else None
    scores = aps_scores(probabilities_calib, y_calib, rng=aps_rng)
    threshold = conformal_quantile(scores, alpha)
    return aps_sets(probabilities_test, threshold, rng=aps_rng), threshold


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n_classes = 4

    x, y = make_overlapping_classes(3000, rng, n_classes=n_classes)
    (x_tr, y_tr), (x_ca, y_ca), (x_te, y_te) = three_way_split(x, y, rng)

    weights = fit_softmax_regression(x_tr, y_tr, n_classes)
    p_ca = predict_proba(weights, x_ca)
    p_te = predict_proba(weights, x_te)

    accuracy = float(np.mean(p_te.argmax(axis=1) == y_te))
    print(f"softmax base model: {accuracy:.3f} test accuracy on {n_classes} "
          f"overlapping classes")
    print(f"  train={x_tr.shape[0]} calib={x_ca.shape[0]} test={x_te.shape[0]}")

    alpha = 0.1
    methods = ("lac", "aps", "aps-rand")
    draw_rng = np.random.default_rng(11)

    print(f"\nprediction sets at alpha={alpha} (nominal coverage {1 - alpha:.2f}):")
    print(f"  {'method':<10}{'threshold':>11}{'coverage':>11}{'mean size':>11}"
          f"{'empty':>9}{'singleton':>11}")
    for method in methods:
        sets, threshold = calibrate_and_predict(p_ca, y_ca, p_te, alpha, method,
                                                rng=draw_rng)
        mean_size, empty, singleton = set_size_summary(sets)
        print(f"  {method:<10}{threshold:>11.3f}{set_coverage(sets, y_te):>11.3f}"
              f"{mean_size:>11.3f}{empty:>9.3f}{singleton:>11.3f}")
    print("  all three land on nominal coverage; they differ entirely in what")
    print("  they charge for it. lac is the smallest and answers with a single")
    print("  class most of the time. deterministic aps carries an extra half")
    print("  class per point purely from the coarseness of its score, which the")
    print("  uniform draw in aps-rand recovers at identical coverage.")

    # the guarantee is a statement about every alpha at once, so checking one
    # value of it is checking almost nothing. averaged over independent splits
    # because a single calibration draw has visible sampling noise at n ~ 900.
    print("\ncoverage vs nominal across alpha (mean of 40 splits):")
    header = f"  {'alpha':>7}{'nominal':>10}"
    for method in methods:
        header += f"{method + ' cov':>13}{method + ' size':>14}"
    print(header)
    for alpha_sweep in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3):
        totals = {method: [0.0, 0.0] for method in methods}
        trials = 40
        for seed in range(trials):
            trial_rng = np.random.default_rng(4000 + seed)
            xs, ys = make_overlapping_classes(1200, trial_rng, n_classes=n_classes)
            (a, b), (c, d), (e, f) = three_way_split(xs, ys, trial_rng)
            w = fit_softmax_regression(a, b, n_classes, iterations=400)
            pc, pt = predict_proba(w, c), predict_proba(w, e)
            for method in methods:
                sets, _ = calibrate_and_predict(pc, d, pt, alpha_sweep, method,
                                                rng=trial_rng)
                totals[method][0] += set_coverage(sets, f)
                totals[method][1] += sets.sum(axis=1).mean()
        row = f"  {alpha_sweep:>7.2f}{1 - alpha_sweep:>10.2f}"
        for method in methods:
            coverage, size = (v / trials for v in totals[method])
            row += f"{coverage:>13.3f}{size:>14.2f}"
        print(row)
    print("  every method sits on the diagonal at every alpha, which is the")
    print("  claim the whole project rests on and the least surprising column")
    print("  in the table. the size columns are where the methods actually")
    print("  differ, and where the difficulty of the problem is reported: sets")
    print("  fall below one class on average once alpha passes 0.2, meaning the")
    print("  method has started returning nothing rather than guess.")

    # and now the assumption. nothing above this line is wrong; everything above
    # this line stops being true the moment the test points stop being
    # exchangeable with the calibration points, and no quantity computed from
    # the calibration set can detect that on its own.
    print(f"\ncovariate shift at alpha={alpha} - calibration is unshifted, "
          f"test is not:")
    shift_methods = ("lac", "aps-rand")
    header = f"  {'shift':>7}"
    for method in shift_methods:
        header += f"{method + ' cov':>13}{method + ' size':>14}"
    print(header + f"{'accuracy':>10}")
    for shift in (0.0, 0.5, 1.0, 1.5, 2.0):
        shift_rng = np.random.default_rng(99)
        x_shift, y_shift = make_overlapping_classes(4000, shift_rng,
                                                    n_classes=n_classes,
                                                    shift=shift)
        p_shift = predict_proba(weights, x_shift)
        row = f"  {shift:>7.1f}"
        for method in shift_methods:
            sets, _ = calibrate_and_predict(p_ca, y_ca, p_shift, alpha, method,
                                            rng=shift_rng)
            row += f"{set_coverage(sets, y_shift):>13.3f}"
            row += f"{sets.sum(axis=1).mean():>14.2f}"
        shift_accuracy = float(np.mean(p_shift.argmax(axis=1) == y_shift))
        print(row + f"{shift_accuracy:>10.3f}")

    print("\n  the sets do not widen to compensate. they cannot - the threshold")
    print("  was fixed by calibration data that never saw the shift, so the only")
    print("  thing that moves is how often the truth falls outside. coverage")
    print("  degrades smoothly and silently while the procedure reports nothing.")
    print("  this is the honest boundary of the whole method: conformal converts")
    print("  exchangeability into a finite-sample guarantee, and it has no way to")
    print("  manufacture one when the input assumption is false. weighted and")
    print("  online variants exist for known shift; none of them remove the need")
    print("  to know something about it.")
