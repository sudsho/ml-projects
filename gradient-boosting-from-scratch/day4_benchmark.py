"""
Day 4 of the gradient boosting regressor from scratch.

Days 1-3 built the whole machine: a CART regression tree, the boosting loop, and
then the general pseudo-residual view with robust losses, subsampling and early
stopping. Today is the "does it actually work" day. Three things:

  1. Benchmark against scikit-learn's GradientBoostingRegressor on a real-ish
     regression problem (Friedman #1), matching hyperparameters as closely as the
     two APIs allow, and report RMSE plus fit time side by side. The goal is not
     to beat sklearn - a pure-NumPy tree that rescans every threshold will always
     be slower - but to land at a comparable error, which is the evidence that
     the from-scratch math is right.

  2. Gain-weighted feature importance from the split gains recorded on each tree
     node (day 1 was refactored to keep `gain` and `n_samples` around for exactly
     this). Friedman #1 is the perfect test: the target only depends on the first
     five features, the last five are pure noise, so a correct importance should
     light up features 0-4 and leave 5-9 near zero.

  3. Learning curves - validation RMSE as a function of the number of boosting
     rounds, for both our model and sklearn, showing the underfit -> good-fit
     arc and where early stopping decided to halt. Saved to a PNG with --plot.

Everything is deterministic and uses only synthetic data from sklearn.datasets,
so the script runs with no downloads.
"""

import argparse
import time

import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBR
from sklearn.model_selection import train_test_split

from day1_cart_tree import _rmse
from day3_losses_stochastic import SquaredError, StochasticGradientBoostingRegressor


# ---------------------------------------------------------------------------
# Feature importance from the split gains stored on the trees.
# ---------------------------------------------------------------------------
def _accumulate_gains(node, importances):
    """Add gain * n_samples of every internal split under `node` into the array.

    Weighting the raw variance reduction by the number of samples that reached
    the node is the standard "impurity importance" recipe: a split high in the
    tree that separates many rows counts for more than an equal-gain split buried
    in a leaf-heavy corner. Leaves contribute nothing - they make no split.
    """
    if node is None or node.is_leaf:
        return
    importances[node.feature] += node.gain * node.n_samples
    _accumulate_gains(node.left, importances)
    _accumulate_gains(node.right, importances)


def gain_feature_importances(model, n_features):
    """Normalized gain-weighted importance summed over every tree in the model."""
    importances = np.zeros(n_features)
    for tree in model.trees_:
        _accumulate_gains(tree.root, importances)
    total = importances.sum()
    return importances / total if total > 0 else importances


# ---------------------------------------------------------------------------
# Learning curves.
# ---------------------------------------------------------------------------
def staged_val_rmse(model, X_val, y_val):
    """Validation RMSE after each successive boosting round of our model.

    Rebuilds the additive prediction one tree at a time instead of calling
    predict() in a loop, so the whole curve costs a single pass over the trees.
    """
    f = np.full(X_val.shape[0], model.init_)
    curve = []
    for tree in model.trees_:
        f = f + model.learning_rate * tree.predict(X_val)
        curve.append(_rmse(y_val, f))
    return np.array(curve)


def sklearn_staged_val_rmse(model, X_val, y_val):
    """Same curve for the sklearn model via its staged_predict generator."""
    return np.array([_rmse(y_val, pred) for pred in model.staged_predict(X_val)])


def plot_learning_curves(ours, theirs, best_round, out_path):
    """Overlay both validation-RMSE curves and mark our early-stopping round."""
    import matplotlib

    matplotlib.use("Agg")  # headless - just write a file
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.arange(1, len(ours) + 1), ours, label="from scratch", lw=2)
    ax.plot(np.arange(1, len(theirs) + 1), theirs, label="sklearn", lw=2, alpha=0.8)
    ax.axvline(best_round, color="gray", ls="--", lw=1,
               label=f"our early stop (round {best_round})")
    ax.set_xlabel("boosting rounds")
    ax.set_ylabel("validation RMSE")
    ax.set_title("Gradient boosting learning curves - Friedman #1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved learning-curve plot to {out_path}")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true",
                        help="save the learning-curve figure to a PNG")
    args = parser.parse_args()

    # Friedman #1: y = 10 sin(pi x0 x1) + 20 (x2 - 0.5)^2 + 10 x3 + 5 x4 + noise.
    # Features 5-9 are irrelevant, which is what makes it a clean importance test.
    # The from-scratch tree rescans every candidate threshold in pure Python, so
    # this stays a deliberately small benchmark - a few hundred rows and under a
    # hundred rounds - which is enough to show comparable error without waiting
    # minutes on the O(features * n^2) split search.
    X, y = make_friedman1(n_samples=500, n_features=10, noise=1.0, random_state=0)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=0)

    common = dict(n_estimators=70, learning_rate=0.08, max_depth=3, subsample=0.8)

    # --- our model -------------------------------------------------------
    t0 = time.perf_counter()
    ours = StochasticGradientBoostingRegressor(
        loss=SquaredError(),
        n_estimators=common["n_estimators"],
        learning_rate=common["learning_rate"],
        max_depth=common["max_depth"],
        subsample=common["subsample"],
        n_iter_no_change=20,
        tol=1e-4,
        random_state=0,
    ).fit(X_train, y_train, X_val, y_val)
    ours_fit_time = time.perf_counter() - t0
    ours_test_rmse = _rmse(y_test, ours.predict(X_test))

    # --- sklearn baseline ------------------------------------------------
    t0 = time.perf_counter()
    skl = SklearnGBR(
        loss="squared_error",
        n_estimators=common["n_estimators"],
        learning_rate=common["learning_rate"],
        max_depth=common["max_depth"],
        subsample=common["subsample"],
        n_iter_no_change=20,
        tol=1e-4,
        validation_fraction=0.15,  # sklearn carves its own early-stop slice
        random_state=0,
    )
    # sklearn manages its own validation split out of the training rows for early
    # stopping, so it only sees X_train/y_train here.
    skl.fit(X_train, y_train)
    skl_fit_time = time.perf_counter() - t0
    skl_test_rmse = _rmse(y_test, skl.predict(X_test))

    print("=" * 60)
    print(f"Benchmark on Friedman #1 ({X.shape[0]} samples, {X.shape[1]} features)")
    print("=" * 60)
    print(f"{'model':16s} | {'trees':>6s} | {'test RMSE':>10s} | {'fit (s)':>8s}")
    print("-" * 60)
    print(f"{'from scratch':16s} | {len(ours.trees_):6d} | "
          f"{ours_test_rmse:10.4f} | {ours_fit_time:8.2f}")
    print(f"{'sklearn':16s} | {skl.n_estimators_:6d} | "
          f"{skl_test_rmse:10.4f} | {skl_fit_time:8.2f}")

    # --- feature importance ---------------------------------------------
    ours_imp = gain_feature_importances(ours, X.shape[1])
    skl_imp = skl.feature_importances_
    print()
    print("Gain-weighted feature importance (features 0-4 are signal, 5-9 noise)")
    print(f"{'feature':>8s} | {'from scratch':>13s} | {'sklearn':>9s}")
    print("-" * 38)
    for j in range(X.shape[1]):
        marker = "  <- signal" if j < 5 else ""
        print(f"{j:8d} | {ours_imp[j]:13.3f} | {skl_imp[j]:9.3f}{marker}")
    signal_mass = ours_imp[:5].sum()
    print(f"\nour importance mass on the 5 signal features: {signal_mass:.1%}")

    # --- learning curves -------------------------------------------------
    ours_curve = staged_val_rmse(ours, X_val, y_val)
    skl_curve = sklearn_staged_val_rmse(skl, X_val, y_val)
    print(f"\nour val RMSE: {ours_curve[0]:.3f} (round 1) -> "
          f"{ours_curve[-1]:.3f} (round {len(ours_curve)})")
    print(f"sklearn val RMSE: {skl_curve[0]:.3f} (round 1) -> "
          f"{skl_curve[-1]:.3f} (round {len(skl_curve)})")

    if args.plot:
        plot_learning_curves(ours_curve, skl_curve, ours.best_iteration_ + 1,
                             "learning_curves.png")


if __name__ == "__main__":
    main()
