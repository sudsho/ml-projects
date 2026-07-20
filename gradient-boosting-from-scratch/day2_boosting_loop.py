"""
Day 2 of the gradient boosting regressor from scratch.

Day 1 built the weak learner - a shallow CART regression tree. Today we wrap it
in the boosting loop that turns a pile of weak trees into one strong model.

The idea for squared-error loss is short. Start from a constant prediction (the
mean of the training targets, which is the constant that minimizes squared
error). Then repeat: look at what the current model still gets wrong - the
residuals y - F(x) - and fit a fresh shallow tree to those residuals. Add a
shrunken slice of that tree's prediction back into the model:

    F_{m}(x) = F_{m-1}(x) + learning_rate * tree_m(x)

For squared error the negative gradient of the loss with respect to the current
prediction is exactly the residual, so "fit a tree to the residuals" is
literally a gradient-descent step in function space - hence *gradient* boosting.
Day 3 generalizes that gradient view to other losses; today we lean on the fact
that for squared error the pseudo-residual and the plain residual coincide.

The learning rate (shrinkage) is the key regularizer. Each tree only gets to
nudge the model a little, so no single weak learner dominates and the ensemble
generalizes better - at the cost of needing more rounds. We track train and
validation RMSE at every round so the underfit -> good-fit -> overfit arc is
visible, which is what day 3's early stopping will key off of.
"""

import numpy as np

from day1_cart_tree import RegressionTree, _rmse


class GradientBoostingRegressor:
    """Least-squares gradient boosting with shallow regression trees.

    Parameters
    ----------
    n_estimators : number of boosting rounds (trees added to the ensemble).
    learning_rate : shrinkage applied to each tree's contribution. Smaller
        values regularize harder and usually want more estimators.
    max_depth : depth of each weak-learner tree. Depth 2-4 is the sweet spot -
        deep enough to model interactions, shallow enough to stay weak.
    min_samples_split : forwarded to each tree; a node below this size is a leaf.
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.init_ = 0.0        # the constant F_0 prediction
        self.trees_ = []        # the fitted weak learners, in order
        self.train_rmse_ = []   # train RMSE after each round
        self.val_rmse_ = []     # val RMSE after each round (if val data given)

    def fit(self, X, y, X_val=None, y_val=None):
        """Fit the ensemble, optionally tracking a held-out validation curve."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        track_val = X_val is not None and y_val is not None
        if track_val:
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float)

        # F_0: the mean is the constant minimizing squared error on the target.
        self.init_ = float(y.mean())
        self.trees_ = []
        self.train_rmse_ = []
        self.val_rmse_ = []

        # Running predictions for train (and val), updated in place each round so
        # we never re-run the whole ensemble just to compute residuals.
        f_train = np.full(len(y), self.init_)
        f_val = np.full(len(y_val), self.init_) if track_val else None

        for _ in range(self.n_estimators):
            # Negative gradient of 1/2 * (y - F)^2 w.r.t. F is the residual.
            residual = y - f_train

            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            ).fit(X, residual)
            self.trees_.append(tree)

            # Take a shrunken step along this tree's correction.
            f_train += self.learning_rate * tree.predict(X)
            self.train_rmse_.append(_rmse(y, f_train))

            if track_val:
                f_val += self.learning_rate * tree.predict(X_val)
                self.val_rmse_.append(_rmse(y_val, f_val))

        return self

    def staged_predict(self, X):
        """Yield the ensemble prediction after each additional tree.

        Handy for plotting learning curves or picking a round to stop at without
        refitting - each item is F_m(x) for m = 1 .. n_estimators.
        """
        X = np.asarray(X, dtype=float)
        f = np.full(X.shape[0], self.init_)
        for tree in self.trees_:
            f += self.learning_rate * tree.predict(X)
            yield f.copy()

    def predict(self, X):
        """Full-ensemble prediction: F_0 plus every shrunken tree."""
        X = np.asarray(X, dtype=float)
        f = np.full(X.shape[0], self.init_)
        for tree in self.trees_:
            f += self.learning_rate * tree.predict(X)
        return f


if __name__ == "__main__":
    # Same deterministic synthetic surface as day 1 so no downloads are needed
    # and the two scripts are directly comparable.
    rng = np.random.default_rng(0)
    n = 400
    X = rng.uniform(-3, 3, size=(n, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + rng.normal(0, 0.1, size=n)

    split = int(0.8 * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # A single depth-3 tree (day 1's best shallow learner) as the baseline to beat.
    baseline = RegressionTree(max_depth=3).fit(X_train, y_train)
    print(f"single depth-3 tree   val RMSE: {_rmse(y_val, baseline.predict(X_val)):.4f}")

    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=3
    ).fit(X_train, y_train, X_val, y_val)

    # Report the curve at a few checkpoints - RMSE should fall fast then flatten.
    print("\nround | train RMSE | val RMSE")
    for m in (1, 5, 20, 50, 100, 200):
        i = m - 1
        print(f"{m:5d} | {model.train_rmse_[i]:10.4f} | {model.val_rmse_[i]:8.4f}")

    best_round = int(np.argmin(model.val_rmse_)) + 1
    print(
        f"\nbest val RMSE {min(model.val_rmse_):.4f} at round {best_round} "
        f"(of {model.n_estimators})"
    )
    # The boosted ensemble should clear the single-tree baseline comfortably, and
    # the best round sitting short of the last one is the early-stopping signal
    # day 3 will act on.
