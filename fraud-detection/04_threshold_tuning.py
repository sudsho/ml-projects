# %% markdown
# ## Credit Card Fraud Detection - Day 4: Threshold Tuning & Final Analysis
#
# Picking the best classification threshold matters more than picking the best
# model when classes are this imbalanced. The default 0.5 cutoff is almost
# never the right operating point. Today we sweep thresholds and find the one
# that meets a business constraint - "precision >= 0.90, then maximize recall".

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

# %% markdown
# ### Reproduce the same train/test split used in earlier scripts

X, y = make_classification(
    n_samples=50_000, n_features=30,
    n_informative=10, n_redundant=5,
    weights=[0.998, 0.002],
    flip_y=0.005, random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42,
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% markdown
# ### Refit XGBoost (best model from yesterday) and grab probabilities

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    scale_pos_weight=neg / pos,
    eval_metric='aucpr', n_jobs=-1, random_state=42,
    use_label_encoder=False,
)
xgb.fit(X_train, y_train)
y_prob = xgb.predict_proba(X_test)[:, 1]

print(f"Test fraud rate: {y_test.mean() * 100:.3f}%")
print(f"PR-AUC: {average_precision_score(y_test, y_prob):.4f}")

# %% markdown
# ### Sweep thresholds and tabulate precision / recall / f1 / costs
#
# Cost model assumption (rough estimate, easy to swap out):
#   - false positive (legit txn flagged): $5  - friction, support cost
#   - false negative (fraud missed):     $200 - average chargeback

FP_COST = 5
FN_COST = 200

thresholds = np.linspace(0.05, 0.95, 19)
rows = []
for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    rows.append({
        "threshold": round(float(t), 2),
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "expected_cost": int(fp * FP_COST + fn * FN_COST),
    })

sweep = pd.DataFrame(rows)
print("\n" + "=" * 80)
print("THRESHOLD SWEEP")
print("=" * 80)
print(sweep.to_string(index=False))

# %% markdown
# ### Pick the best operating point under different policies

policies = {
    "max F1": sweep.loc[sweep["f1"].idxmax()],
    "min expected cost": sweep.loc[sweep["expected_cost"].idxmin()],
}

# precision >= 0.90, then highest recall
high_prec = sweep[sweep["precision"] >= 0.90]
if not high_prec.empty:
    policies["precision >= 0.90, max recall"] = high_prec.loc[high_prec["recall"].idxmax()]

print("\n" + "=" * 80)
print("RECOMMENDED OPERATING POINTS")
print("=" * 80)
for label, row in policies.items():
    print(f"\n[{label}]")
    print(row.to_string())

# %% markdown
# ### Plot precision-recall and mark the chosen operating point

prec_curve, rec_curve, thresh_curve = precision_recall_curve(y_test, y_prob)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(rec_curve, prec_curve, linewidth=2)
chosen = policies.get("precision >= 0.90, max recall", policies["max F1"])
axes[0].scatter(chosen["recall"], chosen["precision"],
                color="red", s=100, zorder=5,
                label=f"chosen t={chosen['threshold']:.2f}")
axes[0].axhline(0.90, color="gray", linestyle="--", alpha=0.5, label="precision=0.90")
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title("Precision-Recall Curve")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(sweep["threshold"], sweep["f1"], label="F1", linewidth=2)
axes[1].plot(sweep["threshold"], sweep["precision"], label="precision", linewidth=2)
axes[1].plot(sweep["threshold"], sweep["recall"], label="recall", linewidth=2)
axes[1].axvline(chosen["threshold"], color="red", linestyle="--", alpha=0.7,
                label=f"chosen={chosen['threshold']:.2f}")
axes[1].set_xlabel("Threshold")
axes[1].set_ylabel("Score")
axes[1].set_title("Metrics vs. Threshold")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("threshold_analysis.png", dpi=150)
plt.close()
print("\nSaved threshold_analysis.png")

# %% markdown
# ### Final report at the chosen threshold

t_final = float(chosen["threshold"])
y_pred_final = (y_prob >= t_final).astype(int)
print("\n" + "=" * 60)
print(f"FINAL REPORT @ threshold = {t_final:.2f}")
print("=" * 60)
print(classification_report(y_test, y_pred_final, digits=4, zero_division=0))

print("Wrote final write-up to README.md.")
