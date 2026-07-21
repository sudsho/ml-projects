"""
Day 1 of the gradient boosting regressor from scratch.

Gradient boosting is an additive model built out of many small regression trees,
so before any boosting can happen we need the weak learner itself: a CART
regression tree. This file builds one from scratch in NumPy.

A regression tree greedily partitions the feature space. At each node it searches
every feature and every candidate threshold for the split that most reduces the
squared error of the target, then recurses on the two halves until a stopping
rule fires (max depth, too few samples, or no split that helps). A leaf predicts
the mean of the targets that land in it, which is exactly the constant that
minimizes squared error over those samples.

The one design choice worth calling out for later: the boosting loop in day 2
fits trees to *residuals*, which can be negative, so this tree makes no
assumption that the target is positive. It also exposes `max_depth` and
`min_samples_split` because shallow trees (depth 2-4) are the usual weak learner
- a single deep tree would overfit and leave nothing for boosting to correct.
"""

import numpy as np


class _Node:
    """One node of the tree: either an internal split or a leaf.

    Internal nodes store the split feature index and threshold plus left/right
    children. Leaves store a constant prediction value. Keeping both in one class
    keeps the recursive predict simple - a node is a leaf iff `value` is set.

    Internal nodes also record `gain` (the variance reduction this split earned)
    and `n_samples` (how many training rows reached the node). Neither is needed
    for prediction, but day 4 sums `gain * n_samples` over every split of a
    feature to build a gain-weighted feature importance, so the tree has to keep
    the bookkeeping around instead of throwing it away after the split is chosen.
    """

    __slots__ = ("feature", "threshold", "left", "right", "value", "gain", "n_samples")

    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None, gain=0.0, n_samples=0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.gain = gain
        self.n_samples = n_samples

    @property
    def is_leaf(self):
        return self.value is not None


def _variance_reduction(y, left_mask):
    """Weighted variance reduction of a split, the CART regression criterion.

    Splitting a node is worthwhile only if the children are "purer" (lower
    variance) than the parent. We measure that as the parent variance minus the
    sample-weighted average of the two child variances. Because the leaf
    prediction is the mean, minimizing child variance is identical to minimizing
    the residual sum of squares, so this is the right thing to maximize.
    """
    n = len(y)
    n_left = int(left_mask.sum())
    n_right = n - n_left
    # A split that isolates zero samples on a side is not a real split.
    if n_left == 0 or n_right == 0:
        return 0.0

    parent_var = y.var()
    left_var = y[left_mask].var()
    right_var = y[~left_mask].var()
    child_var = (n_left * left_var + n_right * right_var) / n
    return parent_var - child_var


def _best_split(X, y, min_samples_split):
    """Find the (feature, threshold) with the largest variance reduction.

    We scan every feature. For a feature we sort its unique values and try
    midpoints between consecutive values as thresholds - midpoints keep the split
    off the exact data points and generalize slightly better than testing the
    values themselves. Returns (feature, threshold, gain) or (None, None, 0.0) if
    no split improves on the parent.
    """
    best_feature, best_threshold, best_gain = None, None, 0.0

    for feature in range(X.shape[1]):
        column = X[:, feature]
        values = np.unique(column)
        if values.size < 2:
            continue  # constant feature, nothing to split on

        thresholds = (values[:-1] + values[1:]) / 2.0
        for threshold in thresholds:
            left_mask = column <= threshold
            if left_mask.sum() < 1 or (~left_mask).sum() < 1:
                continue
            gain = _variance_reduction(y, left_mask)
            if gain > best_gain:
                best_feature, best_threshold, best_gain = feature, threshold, gain

    return best_feature, best_threshold, best_gain


class RegressionTree:
    """A CART regression tree fit by greedy variance reduction.

    Parameters
    ----------
    max_depth : maximum number of splits from root to leaf. Small values (2-4)
        give the shallow weak learners boosting relies on.
    min_samples_split : a node with fewer samples than this becomes a leaf.
    min_impurity_decrease : ignore splits whose variance reduction is below this,
        a light regularizer against splitting on noise.
    """

    def __init__(self, max_depth=3, min_samples_split=2, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        """Recursively grow the tree, returning the subtree rooted here."""
        # Stopping rules: depth budget, too few samples, or a pure node.
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or np.allclose(y, y[0])
        ):
            return _Node(value=float(y.mean()))

        feature, threshold, gain = _best_split(X, y, self.min_samples_split)
        # No split found, or the best split is not worth making.
        if feature is None or gain <= self.min_impurity_decrease:
            return _Node(value=float(y.mean()))

        left_mask = X[:, feature] <= threshold
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return _Node(feature=feature, threshold=threshold, left=left, right=right,
                     gain=gain, n_samples=len(y))

    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.value
        branch = node.left if x[node.feature] <= node.threshold else node.right
        return self._predict_one(x, branch)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.root) for x in X])


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


if __name__ == "__main__":
    # A deterministic synthetic regression problem so the script runs with no
    # downloads. y is a smooth nonlinear surface plus noise - a single tree can
    # only approximate it in axis-aligned steps, which is exactly why day 2 needs
    # a boosted ensemble to sharpen the fit.
    rng = np.random.default_rng(0)
    n = 400
    X = rng.uniform(-3, 3, size=(n, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + rng.normal(0, 0.1, size=n)

    split = int(0.8 * n)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print("depth | train RMSE | val RMSE")
    for depth in (1, 2, 3, 5, 8):
        tree = RegressionTree(max_depth=depth).fit(X_train, y_train)
        train_rmse = _rmse(y_train, tree.predict(X_train))
        val_rmse = _rmse(y_val, tree.predict(X_val))
        print(f"{depth:5d} | {train_rmse:10.4f} | {val_rmse:8.4f}")

    # A shallow tree underfits (high train + val error); a deep one overfits
    # (train error keeps dropping while val error flattens or rises). Boosting
    # keeps the tree shallow and corrects the underfit over many rounds.
