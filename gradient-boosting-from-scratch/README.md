# Gradient Boosting Regressor from Scratch

A gradient boosting machine built in pure NumPy, no scikit-learn in the model
itself. It grows a CART regression tree as the weak learner, then a boosting
loop that fits successive shallow trees to the negative gradient of a
differentiable loss and adds a shrunken slice of each back into the ensemble.
The result lands within noise of scikit-learn's `GradientBoostingRegressor` on a
Friedman #1 benchmark, which is the evidence that the from-scratch math is right.

## Idea

Boosting turns a crowd of weak learners into one strong one by fixing mistakes
sequentially. Start from a constant prediction, then at every round look at the
gradient of the loss with respect to the current model's output - the
pseudo-residual - and fit a shallow tree to it. Add a small step along that tree:

    F_m(x) = F_{m-1}(x) + learning_rate * tree_m(x)

For squared-error loss the pseudo-residual is exactly the ordinary residual
`y - F(x)`, so "fit a tree to the residuals" is literally gradient descent in
function space. Swapping the loss only changes how the initial constant and the
pseudo-residual are computed, which is what lets the same loop be robust to
outliers instead of only least-squares.

## Layout

- `day1_cart_tree.py` - the weak learner. A CART regression tree that splits by
  greedy variance reduction, with `max_depth` / `min_samples_split` stopping and
  a leaf that predicts the mean. Each internal node also records the split
  `gain` and `n_samples` so day 4 can build a gain-weighted importance.
- `day2_boosting_loop.py` - the squared-error boosting loop: constant init, fit
  trees to residuals, shrinkage, and per-round train/validation RMSE tracking so
  the underfit -> good-fit -> overfit arc is visible.
- `day3_losses_stochastic.py` - the general pseudo-residual view. A small loss
  interface (squared, absolute, Huber), stochastic gradient boosting (each tree
  on a random row subsample), and early stopping on the validation loss.
- `day4_benchmark.py` - benchmark against scikit-learn, gain-weighted feature
  importance from the recorded split gains, and learning curves (`--plot`).

## Key design choices

- **Shallow trees as the weak learner.** Depth 2-4 trees deliberately underfit;
  boosting relies on many small corrections rather than one deep tree that would
  leave nothing to correct.
- **Shrinkage is the main regularizer.** Each tree only nudges the model by
  `learning_rate`, so no single learner dominates and the ensemble generalizes
  better, at the cost of needing more rounds.
- **Pseudo-residuals, not residuals.** Fitting trees to `-dL/dF` rather than the
  raw residual is what makes the absolute-error and Huber losses work and stay
  robust to the heavy-tailed targets that wreck plain least-squares.
- **Stochastic subsampling.** Fitting each tree on a random fraction of the rows
  decorrelates the trees and usually improves generalization (Friedman, 2002).
- **Early stopping.** Training halts once the validation loss stops improving for
  `n_iter_no_change` rounds, keeping only the trees up to the best round.
- **Gain-weighted importance.** Summing `gain * n_samples` over every split of a
  feature reproduces the standard impurity-importance recipe, which is why the
  tree keeps that bookkeeping around after choosing each split.

## Results

On Friedman #1 (500 samples, 10 features, only the first 5 carry signal), 70
rounds at `learning_rate=0.08`, depth-3 trees, 0.8 subsampling:

| model        | trees | test RMSE |
|--------------|-------|-----------|
| from scratch |   70  |   ~1.64   |
| sklearn      |   70  |   ~1.71   |

The gain-weighted importance puts ~99% of its mass on the five true signal
features and leaves the five noise features near zero, matching sklearn's
`feature_importances_` closely. The pure-NumPy tree rescans every candidate
threshold, so it is far slower to fit than sklearn's histogram-free Cython
splitter - the benchmark is kept small for that reason, not because the model
needs it.

## Run

```bash
python day1_cart_tree.py          # weak learner, depth-vs-RMSE table
python day2_boosting_loop.py      # squared-error boosting, train/val curves
python day3_losses_stochastic.py  # robust losses + early stopping
python day4_benchmark.py --plot   # sklearn benchmark, importance, curves
```

Everything is deterministic and uses only synthetic data from
`sklearn.datasets`, so no downloads are needed.
