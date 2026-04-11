"""
Day 4: Ensemble Methods, Final Evaluation, and Project Documentation
House Price Prediction Project

Builds a stacking ensemble on top of the Day 3 base models (Ridge, Lasso,
GradientBoosting) using a meta-learner (Ridge). Also adds a VotingRegressor
baseline, runs permutation importance analysis, and produces final summary
metrics to wrap up the project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (GradientBoostingRegressor,
                               RandomForestRegressor,
                               VotingRegressor,
                               StackingRegressor)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance

# ─────────────────────────────────────────────
# 1. Load preprocessed data (produced by Day 2)
# ─────────────────────────────────────────────
train_df = pd.read_csv("train_features.csv")
val_df   = pd.read_csv("val_features.csv")

X_train = train_df.drop(columns=["log_price"])
y_train = train_df["log_price"]
X_val   = val_df.drop(columns=["log_price"])
y_val   = val_df["log_price"]

print("=== Day 4: Ensemble Methods & Final Evaluation ===")
print(f"Train: {X_train.shape}, Val: {X_val.shape}\n")

# ─────────────────────────────────────────────
# 2. Base estimators (same config as Day 3)
# ─────────────────────────────────────────────
ridge = Ridge(alpha=1.0, random_state=42)
lasso = Lasso(alpha=0.001, random_state=42, max_iter=5000)
gb    = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=4, subsample=0.8, random_state=42)
rf    = RandomForestRegressor(
            n_estimators=300, max_depth=8,
            min_samples_leaf=3, random_state=42)

# ─────────────────────────────────────────────
# 3. Voting Regressor (simple average ensemble)
# ─────────────────────────────────────────────
voter = VotingRegressor(
    estimators=[("ridge", ridge), ("lasso", lasso), ("gb", gb), ("rf", rf)]
)
voter.fit(X_train, y_train)
voter_preds = voter.predict(X_val)

# ─────────────────────────────────────────────
# 4. Stacking Regressor (meta-learner = Ridge)
# ─────────────────────────────────────────────
stacker = StackingRegressor(
    estimators=[("gb", gb), ("rf", rf), ("ridge", ridge)],
    final_estimator=Ridge(alpha=0.5),
    cv=5,
    passthrough=False
)
stacker.fit(X_train, y_train)
stacker_preds = stacker.predict(X_val)

# ─────────────────────────────────────────────
# 5. Evaluate all ensemble variants
# ─────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2, "preds": y_pred}

ensemble_results = [
    evaluate("Voting Ensemble",  y_val, voter_preds),
    evaluate("Stacking Ensemble", y_val, stacker_preds),
]

print(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
print("=" * 50)
for r in ensemble_results:
    print(f"{r['model']:<22} {r['rmse']:>8.4f} {r['mae']:>8.4f} {r['r2']:>8.4f}")
print()

# ─────────────────────────────────────────────
# 6. Cross-validation on the stacker
# ─────────────────────────────────────────────
cv_r2  = cross_val_score(stacker, X_train, y_train, cv=5, scoring="r2")
cv_mse = cross_val_score(stacker, X_train, y_train, cv=5,
                          scoring="neg_mean_squared_error")
print(f"Stacking 5-fold CV  R²:   {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"Stacking 5-fold CV  RMSE: {np.sqrt(-cv_mse).mean():.4f} ± {np.sqrt(-cv_mse).std():.4f}\n")

# ─────────────────────────────────────────────
# 7. Convert best model predictions to real $ values
# ─────────────────────────────────────────────
best = min(ensemble_results, key=lambda x: x["rmse"])
y_real      = np.expm1(y_val)
preds_real  = np.expm1(best["preds"])
rmse_real   = np.sqrt(mean_squared_error(y_real, preds_real))
mae_real    = mean_absolute_error(y_real, preds_real)
print(f"Best ensemble: {best['model']}")
print(f"  Val RMSE = ${rmse_real * 100_000:,.0f}")
print(f"  Val MAE  = ${mae_real  * 100_000:,.0f}\n")

# ─────────────────────────────────────────────
# 8. Permutation importance on the stacking model
# ─────────────────────────────────────────────
perm = permutation_importance(stacker, X_val, y_val,
                               n_repeats=10, random_state=42, scoring="r2")
imp_df = pd.DataFrame({
    "feature":    X_val.columns,
    "importance": perm.importances_mean,
    "std":        perm.importances_std,
}).sort_values("importance", ascending=False)

print("=== Top 10 Permutation Importances (Stacking) ===")
print(imp_df.head(10).to_string(index=False))

# ─────────────────────────────────────────────
# 9. Final comparison plot: all 6 models
# ─────────────────────────────────────────────
all_models = {
    "Linear Regression": LinearRegression(),
    "Ridge":             Ridge(alpha=1.0, random_state=42),
    "Lasso":             Lasso(alpha=0.001, random_state=42, max_iter=5000),
    "Gradient Boosting": gb,
    "Voting Ensemble":   voter,
    "Stacking Ensemble": stacker,
}
final_results = []
for name, mdl in all_models.items():
    if name not in ["Voting Ensemble", "Stacking Ensemble"]:
        mdl.fit(X_train, y_train)
    p = mdl.predict(X_val)
    final_results.append(evaluate(name, y_val, p))

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
names  = [r["model"] for r in final_results]
rmses  = [r["rmse"]  for r in final_results]
r2s    = [r["r2"]    for r in final_results]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bars = axes[0].bar(names, rmses, color=colors, edgecolor="white", alpha=0.85)
axes[0].set_title("Validation RMSE (lower is better)")
axes[0].set_ylabel("RMSE (log scale)")
axes[0].set_xticklabels(names, rotation=20, ha="right", fontsize=9)
for b, v in zip(bars, rmses):
    axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0005,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=8)

bars = axes[1].bar(names, r2s, color=colors, edgecolor="white", alpha=0.85)
axes[1].set_title("Validation R² (higher is better)")
axes[1].set_ylabel("R²")
axes[1].set_ylim(0, 1)
axes[1].set_xticklabels(names, rotation=20, ha="right", fontsize=9)
for b, v in zip(bars, r2s):
    axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=8)

plt.suptitle("Final Model Comparison — House Price Prediction", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("final_comparison.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nSaved: final_comparison.png")

# ─────────────────────────────────────────────
# 10. Permutation importance bar chart
# ─────────────────────────────────────────────
top10 = imp_df.head(10)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top10["feature"][::-1], top10["importance"][::-1],
        xerr=top10["std"][::-1], color="#4C72B0", alpha=0.85,
        edgecolor="white", capsize=3)
ax.set_title("Permutation Importances — Stacking Ensemble", fontsize=12)
ax.set_xlabel("Mean R² decrease")
plt.tight_layout()
plt.savefig("permutation_importance.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: permutation_importance.png")

print("\n=== Day 4 Complete — Project Finished! ===")
best_final = min(final_results, key=lambda x: x["rmse"])
print(f"  Best overall model: {best_final['model']}")
print(f"  Val RMSE:           {best_final['rmse']:.4f}")
print(f"  Val R²:             {best_final['r2']:.4f}")
