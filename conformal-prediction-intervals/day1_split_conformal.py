"""
Day 1 of conformal prediction from scratch.

Almost every uncertainty estimate in machine learning is a claim about the model
rather than about the world. A Gaussian process hands back a posterior variance
that is exactly right *if* the kernel and the noise model are right. A bootstrap
interval is right if resampling reproduces the sampling distribution. Dropout at
test time approximates a posterior over weights that nobody chose deliberately.
All three degrade quietly when the assumption behind them is wrong, and none of
them tell you that it happened.

Split conformal prediction makes a much narrower promise and actually keeps it.
Given a fitted model and a held-out calibration set, it returns intervals whose
marginal coverage is at least 1 - alpha in finite samples, for any base model,
any data distribution, and any sample size. No asymptotics, no well-specified
likelihood, no assumption that the model is even remotely good. The one thing it
does need is exchangeability of the calibration and test points.

The mechanism is almost embarrassingly simple, which is the interesting part:

  1. fit the model on a training split, and never look at it again;
  2. score every calibration point by how badly the model missed it, here
     s_i = |y_i - f(x_i)|, the "nonconformity" score;
  3. take a specific empirical quantile of those scores, q;
  4. predict f(x) +/- q.

Everything nontrivial lives in step 3, in the +1 that turns an ordinary
empirical quantile into a finite-sample guarantee. That correction and where it
comes from is most of what this file is about.

What is *not* promised is equally important and is what day 2 exists for. The
guarantee is marginal: it holds on average over the draw of the calibration set
and the test point together. It says nothing about coverage conditional on x,
and the constant-width band built here is visibly wrong on heteroscedastic data
- too wide where the noise is small, too narrow where it is large - while still
being perfectly valid on average. The data below is generated with input-
dependent noise on purpose so that failure is measurable rather than asserted.
"""

import numpy as np


def make_heteroscedastic_data(n_samples, rng, noise_floor=0.15):
    """A 1-d regression problem whose noise scale grows with x.

    The mean function is smooth and easy; the point of the dataset is entirely
    in the noise. sigma(x) rises roughly fivefold across the domain, so a
    constant-width interval cannot be simultaneously tight on the left and
    honest on the right. Marginal coverage will still come out right, which is
    precisely the gap between marginal and conditional validity.
    """
    x = rng.uniform(-3.0, 3.0, size=n_samples)
    mean = np.sin(1.5 * x) + 0.25 * x
    sigma = noise_floor + 0.45 * np.abs(x)
    y = mean + sigma * rng.standard_normal(n_samples)
    return x[:, None], y


def polynomial_features(x, degree):
    """Vandermonde design matrix with an intercept column, shape (n, degree+1)."""
    x = np.asarray(x).reshape(-1)
    return np.vander(x, degree + 1, increasing=True)


def fit_ridge(x, y, degree=6, penalty=1e-3):
    """Closed-form ridge fit on polynomial features.

    Deliberately a plain least-squares model rather than anything clever. The
    conformal machinery is indifferent to what produced the predictions, and
    using a mediocre model makes that indifference easier to see: the coverage
    below does not move when the base model gets worse, only the width does.

    The intercept is left unpenalized. Shrinking it would tie the fit to the
    arbitrary location of y and is a standard bug rather than a modelling
    choice.
    """
    design = polynomial_features(x, degree)
    ridge = penalty * np.eye(design.shape[1])
    ridge[0, 0] = 0.0
    gram = design.T @ design + ridge
    return np.linalg.solve(gram, design.T @ y)


def predict_ridge(coefficients, x, degree=6):
    """Apply a fit from `fit_ridge` to new inputs."""
    return polynomial_features(x, degree) @ coefficients


def absolute_residual_scores(model_fn, x, y):
    """Nonconformity scores s_i = |y_i - f(x_i)|.

    "Nonconformity" is just a name for any function that gets larger when a
    point is more surprising under the fitted model. Absolute residual is the
    obvious choice for regression and gives constant-width intervals. Nothing
    in the theory requires it - dividing by an estimate of sigma(x) gives
    varying width and is one of the two routes into tomorrow's material.
    """
    return np.abs(y - model_fn(x))


def conformal_quantile(scores, alpha):
    """The (n+1)-corrected empirical quantile of the calibration scores.

    Take k = ceil((n + 1) * (1 - alpha)) and return the k-th smallest score.

    The correction is the entire finite-sample guarantee, so it is worth being
    explicit about where it comes from. Under exchangeability the test score
    s_{n+1} is equally likely to land in any of the n+1 positions of the sorted
    combined sample, calibration scores included. So

        P(s_{n+1} <= k-th smallest calibration score) >= k / (n + 1)

    and choosing the smallest k that makes k / (n + 1) reach 1 - alpha is
    exactly the ceiling above. Using the ordinary empirical quantile - k / n -
    would be asymptotically fine and wrong by O(1/n) here, which at n = 50 and
    alpha = 0.1 is the difference between 90% and roughly 88%.

    When k > n the recipe asks for an order statistic that does not exist,
    which happens whenever n < 1/alpha - 1, and the honest return value is
    infinity: 19 calibration points cannot certify 95% coverage. Returning the
    maximum score instead is a quiet way to break the guarantee, and it is the
    failure mode to watch for when alpha is small or the calibration split is.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return np.inf
    return np.partition(scores, k - 1)[k - 1]


def split_conformal(x_train, y_train, x_calib, y_calib, x_test, alpha=0.1,
                    degree=6, penalty=1e-3):
    """Fit, calibrate, and return (lower, upper, half_width) for the test points.

    The three splits do genuinely different jobs and mixing any two of them
    breaks the guarantee rather than merely weakening it. Training data shapes
    f. Calibration data is never seen by the fit, which is what leaves the
    calibration residuals exchangeable with the test residual. Scoring on the
    training residuals instead would make the scores optimistically small and
    the intervals too narrow, and the amount too narrow depends on how much the
    model overfits - so the error is not a fixed bias that could be corrected.
    """
    coefficients = fit_ridge(x_train, y_train, degree=degree, penalty=penalty)
    model_fn = lambda x: predict_ridge(coefficients, x, degree=degree)

    scores = absolute_residual_scores(model_fn, x_calib, y_calib)
    half_width = conformal_quantile(scores, alpha)

    center = model_fn(x_test)
    return center - half_width, center + half_width, half_width


def empirical_coverage(lower, upper, y_true):
    """Fraction of true values that land inside their interval."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def average_width(lower, upper):
    """Mean interval width, the thing being traded against coverage."""
    return float(np.mean(upper - lower))


def three_way_split(x, y, rng, train_frac=0.4, calib_frac=0.3):
    """Shuffle once, then cut into train / calibration / test."""
    order = rng.permutation(x.shape[0])
    n_train = int(train_frac * x.shape[0])
    n_calib = int(calib_frac * x.shape[0])
    train, calib, test = np.split(order, [n_train, n_train + n_calib])
    return (x[train], y[train]), (x[calib], y[calib]), (x[test], y[test])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x, y = make_heteroscedastic_data(2000, rng)
    (x_tr, y_tr), (x_ca, y_ca), (x_te, y_te) = three_way_split(x, y, rng)
    print(f"split sizes: train={x_tr.shape[0]} calib={x_ca.shape[0]} test={x_te.shape[0]}")

    print("\ncoverage vs nominal, single split:")
    for alpha in (0.20, 0.10, 0.05, 0.01):
        lo, hi, q = split_conformal(x_tr, y_tr, x_ca, y_ca, x_te, alpha=alpha)
        print(f"  alpha={alpha:<5} target={1 - alpha:.2f} "
              f"empirical={empirical_coverage(lo, hi, y_te):.3f} "
              f"width={average_width(lo, hi):.3f} q={q:.3f}")

    # the guarantee is about the average over calibration draws, so one split
    # proving nothing is expected. across many splits the mean should sit just
    # above nominal, and the spread is real - a single interval is a random
    # object even though the procedure is not.
    print("\n200 independent splits at alpha=0.1:")
    coverages = []
    for seed in range(200):
        trial_rng = np.random.default_rng(1000 + seed)
        xs, ys = make_heteroscedastic_data(600, trial_rng)
        (a, b), (c, d), (e, f) = three_way_split(xs, ys, trial_rng)
        lo, hi, _ = split_conformal(a, b, c, d, e, alpha=0.1)
        coverages.append(empirical_coverage(lo, hi, f))
    coverages = np.array(coverages)
    print(f"  mean={coverages.mean():.4f} sd={coverages.std():.4f} "
          f"min={coverages.min():.3f} max={coverages.max():.3f}")

    # the claim that the base model does not matter is easy to state and easy
    # to disbelieve, so here it is with a model that has been sabotaged. degree
    # 0 is a constant predictor; it knows nothing about x at all.
    print("\nsame procedure, deliberately bad base models (alpha=0.1):")
    for degree in (0, 1, 6, 20):
        lo, hi, q = split_conformal(x_tr, y_tr, x_ca, y_ca, x_te,
                                    alpha=0.1, degree=degree)
        print(f"  degree={degree:<3} coverage={empirical_coverage(lo, hi, y_te):.3f} "
              f"width={average_width(lo, hi):.3f}")

    # and the part the guarantee does not cover. coverage is marginal, so it
    # can be exactly right overall while being badly wrong in every region.
    lo, hi, _ = split_conformal(x_tr, y_tr, x_ca, y_ca, x_te, alpha=0.1)
    print("\nconditional coverage by |x| bin (nominal 0.90):")
    magnitude = np.abs(x_te).reshape(-1)
    edges = [0.0, 0.75, 1.5, 2.25, 3.0]
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (magnitude >= left) & (magnitude < right)
        if mask.sum() == 0:
            continue
        print(f"  |x| in [{left:.2f}, {right:.2f}): n={mask.sum():<4} "
              f"coverage={empirical_coverage(lo[mask], hi[mask], y_te[mask]):.3f}")

    # too few calibration points and the honest answer is an infinite interval
    # rather than a narrow one.
    print("\ncalibration too small for the requested alpha:")
    for n_calib in (10, 18, 19, 20, 40):
        q = conformal_quantile(np.linspace(0.1, 2.0, n_calib), alpha=0.05)
        print(f"  n={n_calib:<3} alpha=0.05 -> q={q}")
