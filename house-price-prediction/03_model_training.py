"""
Day 3: Model Training - Linear Regression, Ridge, Lasso, Gradient Boosting
House Price Prediction Project

Loads the preprocessed features from Day 2 and trains four regression models.
Compares performance using RMSE and R² on the validation set. Plots residuals
and learning curves to diagnose underfitting vs overfitting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, learning_curve

# ─────────────────────────────────────────────
# 1. Load preprocessed data from Day 2
# ─────────────────────────────────────────────
train_df = pd.read_csv("train_features.csv")
val_df   = pd.read_csv("val_features.csv")

X_train = train_df.drop(columns=["log_price"])
y_train = train_df["log_price"]
X_val   = val_df.drop(columns=["log_price"])
y_val   = val_df["log_price"]

print("=== Day 3: Model Training ===")
print(f"Train: {X_train.shape}, Val: {X_val.shape}")
print(f"Features: {list(X_train.columns)}")

# ─────────────────────────────────────────────
# 2. Define models
# ─────────────────────────────────────────────
models = {
    "Linear Regression":     LinearRegression(),
    "Ridge (alpha=1)":       Ridge(alpha=1.0, random_state=42),
    "Lasso (alpha=0.001)":   Lasso(alpha=0.001, random_state=42, max_iter=5000),
    "Gradient Boosting":     GradientBoostingRegressor(
                                 n_estimators=300, learning_rate=0.05,
                                 max_depth=4, subsample=0.8,
                                 random_state=42
                             ),
}

# ─────────────────────────────────────────────
# 3. Train, predict, and evaluate each model
# ─────────────────────────────────────────────
results = []

print("\n" + "=" * 60)
print(f"{'Model':<24} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'CV-R²':>10}")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae  = mean_absolute_error(y_val, preds)
    r2   = r2_score(y_val, preds)

    # 5-fold cross-val on training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    cv_r2 = cv_scores.mean()

    results.append({
        "model": name, "rmse": rmse, "mae": mae,
        "r2": r2, "cv_r2": cv_r2, "preds": preds
    })
    print(f"{name:<24} {rmse:>8.4f} {mae:>8.4f} {r2:>8.4f} {cv_r2:>10.4f}")

print("=" * 60)

# ─────────────────────────────────────────────
# 4. Convert log predictions back to dollar values
# ─────────────────────────────────────────────
print("\n=== Val Set Error in Real Units ($100k) ===")
y_val_real = np.expm1(y_val)

for res in results:
    preds_real = np.expm1(res["preds"])
    rmse_real  = np.sqrt(mean_squared_error(y_val_real, preds_real))
    mae_real   = mean_absolute_error(y_val_real, preds_real)
    print(f"  {res['model']:<24}  RMSE=${rmse_real*100:.0f}k   MAE=${mae_real*100:.0f}k")

# ─────────────────────────────────────────────
# 5. Plot: Predicted vs Actual (all 4 models)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

for i, (res, ax) in enumerate(zip(results, axes)):
    ax.scatter(y_val, res["preds"], alpha=0.2, s=6, color=colors[i])
    lims = [min(y_val.min(), res["preds"].min()),
            max(y_val.max(), res["preds"].max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="perfect fit")
    ax.set_xlabel("Actual log(price)")
    ax.set_ylabel("Predicted log(price)")
    ax.set_title(f"{res['model']}\nR²={res['r2']:.4f}  RMSE={res['rmse']:.4f}", fontsize=10)
    ax.legend(fontsize=8)

plt.suptitle("Predicted vs Actual — All Models", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("pred_vs_actual.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nSaved: pred_vs_actual.png")

# ─────────────────────────────────────────────
# 6. Residual plots
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()

for i, (res, ax) in enumerate(zip(results, axes)):
    residuals = y_val.values - res["preds"]
    ax.scatter(res["preds"], residuals, alpha=0.2, s=6, color=colors[i])
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted log(price)")
    ax.set_ylabel("Residual")
    ax.set_title(f"{res['model']} — Residuals", fontsize=10)

plt.suptitle("Residual Plots", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("residuals.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: residuals.png")

# ─────────────────────────────────────────────
# 7. Model comparison bar chart
# ─────────────────────────────────────────────
model_names = [r["model"].replace(" (", "\n(") for r in results]
rmse_vals   = [r["rmse"] for r in results]
r2_vals     = [r["r2"]   for r in results]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

bars = axes[0].bar(model_names, rmse_vals, color=colors, edgecolor="white", alpha=0.85)
axes[0].set_title("Validation RMSE (lower is better)", fontsize=11)
axes[0].set_ylabel("RMSE")
for bar, val in zip(bars, rmse_vals):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

bars = axes[1].bar(model_names, r2_vals, color=colors, edgecolor="white", alpha=0.85)
axes[1].set_title("Validation R² (higher is better)", fontsize=11)
axes[1].set_ylabel("R²")
axes[1].set_ylim(0, 1)
for bar, val in zip(bars, r2_vals):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

plt.suptitle("Model Comparison — House Price Prediction", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: model_comparison.png")

# ─────────────────────────────────────────────
# 8. Gradient Boosting feature importances
# ─────────────────────────────────────────────
gb_model = models["Gradient Boosting"]
importances = pd.Series(gb_model.feature_importances_, index=X_train.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind="barh", ax=ax, color="#4C72B0", edgecolor="white", alpha=0.85)
ax.set_title("Gradient Boosting Feature Importances", fontsize=12)
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importances_gb.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: feature_importances_gb.png")

# ─────────────────────────────────────────────
# 9. Summary
# ─────────────────────────────────────────────
best = min(results, key=lambda x: x["rmse"])
print(f"\n=== Day 3 Complete: Model Training done ===")
print(f"  Best model: {best['model']}")
print(f"  Val RMSE:   {best['rmse']:.4f}  |  Val R²: {best['r2']:.4f}")
print(f"\nNext (Day 4): Ensemble methods, stacking, and final evaluation")
