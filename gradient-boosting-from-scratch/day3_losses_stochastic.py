"""
Day 3 of the gradient boosting regressor from scratch.

Day 2 fit trees to plain residuals y - F(x). That works only because for
squared-error loss the negative gradient of the loss w.r.t. the current
prediction happens to equal the residual. The real gradient-boosting recipe is
more general: whatever the differentiable loss L(y, F), each round fits a tree to
the *pseudo-residual* -dL/dF evaluated at the current model, then takes a shrunken
step along it. Swapping the loss is then just swapping how the initial constant
and the pseudo-residual are computed.

Three things get added today, all of which are what separates a toy from a
usable regressor:

  1. A small loss interface (squared error, absolute error, Huber) so the same
     boosting loop can be robust to outliers, not just least-squares.
  2. Stochastic gradient boosting: fit each tree on a random subsample of the
     rows. Besides being faster, the injected noise decorrelates the trees and
     usually improves generalization (Friedman, 2002).
  3. Early stopping: watch the validation loss and stop once it has not improved
     by more than `tol` for `n_iter_no_change` consecutive rounds, keeping only
     the trees up to the best round.
"""

import numpy as np

from day1_cart_tree import RegressionTree, _rmse


class SquaredError:
    """Least-squares loss. Pseudo-residual is the ordinary residual."""

    def init_estimate(self, y):
        return float(y.mean())  # constant minimizing mean squared error

    def negative_gradient(self, y, f):
        return y - f


class AbsoluteError:
    """L1 loss. Robust to outliers; pseudo-residual is the sign of the error."""

    def init_estimate(self, y):
        return float(np.median(y))  # constant minimizing mean absolute error

    def negative_gradient(self, y, f):
        return np.sign(y - f)


class HuberError:
    """Huber loss: squared for small errors, linear beyond `delta`.

    A compromise between the two above - smooth near zero, robust in the tails.
    The threshold is recomputed from the residual quantile each round, following
    Friedman's original stochastic-gradient-boosting paper.
    """

    def __init__(self, alpha=0.9):
        self.alpha = alpha  # quantile of |residual| used as the delta cutoff

    def init_estimate(self, y):
        return float(np.median(y))

    def negative_gradient(self, y, f):
        error = y - f
        delta = np.quantile(np.abs(error), self.alpha)
        small = np.abs(error) <= delta
        grad = np.where(small, error, delta * np.sign(error))
        return grad


class StochasticGradientBoostingRegressor:
    """Gradient boosting with a pluggable loss, row subsampling, early stopping.

    Parameters
    ----------
    loss : an object exposing `init_estimate(y)` and `negative_gradient(y, f)`.
    n_estimators : maximum number of boosting rounds.
    learning_rate : shrinkage applied to each tree's contribution.
    max_depth, min_samples_split : forwarded to each weak-learner tree.
    subsample : fraction of rows drawn (without replacement) to fit each tree.
        1.0 recovers plain gradient boosting; < 1.0 is stochastic GB.
    n_iter_no_change, tol : early-stopping patience and minimum improvement.
        Require validation data passed to `fit` to take effect.
    random_state : seed for the subsampling RNG (reproducible runs).
    """

    def __init__(
        self,
        loss=None,
        n_estimators=300,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        subsample=1.0,
        n_iter_no_change=None,
        tol=1e-4,
        random_state=0,
    ):
        self.loss = loss if loss is not None else SquaredError()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state

        self.init_ = 0.0
        self.trees_ = []
        self.train_loss_ = []
        self.val_loss_ = []
        self.best_iteration_ = None

    def _val_metric(self, y_val, f_val):
        # RMSE is a readable stand-in for the objective on the validation set;
        # it is monotonic with squared-error loss and fine as a stopping signal.
        return _rmse(y_val, f_val)

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        track_val = X_val is not None and y_val is not None
        if track_val:
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float)

        self.init_ = self.loss.init_estimate(y)
        self.trees_ = []
        self.train_loss_ = []
        self.val_loss_ = []

        f_train = np.full(len(y), self.init_)
        f_val = np.full(len(y_val), self.init_) if track_val else None

        n = len(y)
        sample_size = max(1, int(round(self.subsample * n)))
        best_val = np.inf
        rounds_since_best = 0

        for m in range(self.n_estimators):
            # Pseudo-residual: negative gradient of the loss at the current model.
            pseudo = self.loss.negative_gradient(y, f_train)

            if sample_size < n:
                idx = rng.choice(n, size=sample_size, replace=False)
            else:
                idx = np.arange(n)

            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            ).fit(X[idx], pseudo[idx])
            self.trees_.append(tree)

            # The tree is fit on a subsample but predicts (and updates) all rows.
            f_train += self.learning_rate * tree.predict(X)
            self.train_loss_.append(_rmse(y, f_train))

            if track_val:
                f_val += self.learning_rate * tree.predict(X_val)
                current = self._val_metric(y_val, f_val)
                self.val_loss_.append(current)

                if current < best_val - self.tol:
                    best_val = current
                    self.best_iteration_ = m
                    rounds_since_best = 0
                else:
                    rounds_since_best += 1

                if (
                    self.n_iter_no_change is not None
                    and rounds_since_best >= self.n_iter_no_change
                ):
                    # Drop the trees added after the best round - they only made
                    # validation worse - and stop.
                    self.trees_ = self.trees_[: self.best_iteration_ + 1]
                    break

        if self.best_iteration_ is None:
            self.best_iteration_ = len(self.trees_) - 1
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        f = np.full(X.shape[0], self.init_)
        for tree in self.trees_:
            f += self.learning_rate * tree.predict(X)
        return f


if __name__ == "__main__":
    # Reuse day 2's synthetic surface, then spike a handful of targets to make the
    # robust losses earn their keep.
    rng = np.random.default_rng(0)
    n = 500
    X = rng.uniform(-3, 3, size=(n, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + rng.normal(0, 0.1, size=n)
    outliers = rng.choice(n, size=15, replace=False)
    y[outliers] += rng.normal(0, 6, size=15)  # heavy-tailed contamination

    split = int(0.8 * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    losses = {
        "squared": SquaredError(),
        "absolute": AbsoluteError(),
        "huber": HuberError(alpha=0.9),
    }
    print("loss     | trees used | best round | val RMSE")
    for name, loss in losses.items():
        model = StochasticGradientBoostingRegressor(
            loss=loss,
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.7,
            n_iter_no_change=15,
            tol=1e-4,
            random_state=0,
        ).fit(X_train, y_train, X_val, y_val)
        val_rmse = _rmse(y_val, model.predict(X_val))
        print(
            f"{name:8s} | {len(model.trees_):10d} | {model.best_iteration_ + 1:10d} "
            f"| {val_rmse:8.4f}"
        )
    # On outlier-contaminated data the L1/Huber models should reach a lower
    # validation RMSE than plain squared error, and early stopping should keep
    # far fewer than the 400-round budget.
