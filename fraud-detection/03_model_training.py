# %% markdown
# ## Credit Card Fraud Detection - Day 3: Model Training & Comparison
#
# Today we train three different models suited for imbalanced classification:
#   1. Random Forest with class_weight='balanced'
#   2. XGBoost with scale_pos_weight
#   3. Isolation Forest (unsupervised anomaly detection)
#
# We evaluate each on the same test set using PR-AUC, F1, and recall@high-precision.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    average_precision_score, roc_auc_score, f1_score,
)
from xgboost import XGBClassifier

# %% markdown
# ### Load data (using the same synthetic distribution as Day 2)

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

print(f"Train: {X_train.shape}  |  fraud: {y_train.mean() * 100:.3f}%")
print(f"Test:  {X_test.shape}  |  fraud: {y_test.mean() * 100:.3f}%")

# %% markdown
# ### Helper: evaluate a model and collect metrics

def evaluate(name, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "model": name,
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "recall_at_thresh": y_pred[y_true == 1].mean() if (y_true == 1).any() else 0.0,
    }


results = []

# %% markdown
# ### Model 1: Random Forest (balanced class weights)

print("\n" + "=" * 50)
print("Random Forest (class_weight='balanced')")
print("=" * 50)
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=200, max_depth=12,
    class_weight='balanced',
    n_jobs=-1, random_state=42,
)
rf.fit(X_train, y_train)
print(f"Trained in {time.time() - t0:.1f}s")

y_prob_rf = rf.predict_proba(X_test)[:, 1]
results.append(evaluate("Random Forest", y_test, y_prob_rf))
print(classification_report(y_test, (y_prob_rf >= 0.5).astype(int), digits=4, zero_division=0))

# %% markdown
# ### Model 2: XGBoost (scale_pos_weight)

print("\n" + "=" * 50)
print("XGBoost (scale_pos_weight)")
print("=" * 50)
t0 = time.time()
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale = neg / pos
print(f"scale_pos_weight = {scale:.1f}")

xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    scale_pos_weight=scale,
    eval_metric='aucpr', n_jobs=-1, random_state=42,
    use_label_encoder=False,
)
xgb.fit(X_train, y_train)
print(f"Trained in {time.time() - t0:.1f}s")

y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
results.append(evaluate("XGBoost", y_test, y_prob_xgb))
print(classification_report(y_test, (y_prob_xgb >= 0.5).astype(int), digits=4, zero_division=0))

# %% markdown
# ### Model 3: Isolation Forest (unsupervised - no labels at training time)

print("\n" + "=" * 50)
print("Isolation Forest (unsupervised)")
print("=" * 50)
t0 = time.time()
iso = IsolationForest(
    n_estimators=200, contamination=y_train.mean(),
    random_state=42, n_jobs=-1,
)
iso.fit(X_train) # note: NOT using y_train
print(f"Trained in {time.time() - t0:.1f}s")

# Convert anomaly scores to fraud probability (higher = more likely fraud)
iso_scores = -iso.score_samples(X_test)
# Min-max normalize to [0, 1]
iso_prob = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-9)
results.append(evaluate("Isolation Forest", y_test, iso_prob))

# %% markdown
# ### Compare all models

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("pr_auc", ascending=False).reset_index(drop=True)
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False))

# %% markdown
# ### Plot precision-recall curves

fig, ax = plt.subplots(figsize=(10, 6))
for name, probs in [
    ("Random Forest", y_prob_rf),
    ("XGBoost", y_prob_xgb),
    ("Isolation Forest", iso_prob),
]:
    p, r, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    ax.plot(r, p, label=f"{name} (PR-AUC={ap:.3f})", linewidth=2)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall: Fraud Detection Models")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("model_comparison_pr.png", dpi=150)
plt.close()
print("\nSaved model_comparison_pr.png")

# %% markdown
# ### Confusion matrix for the best model

best_name = results_df.iloc[0]["model"]
print(f"\nBest model by PR-AUC: {best_name}")

best_probs = {"Random Forest": y_prob_rf, "XGBoost": y_prob_xgb, "Isolation Forest": iso_prob}[best_name]
cm = confusion_matrix(y_test, (best_probs >= 0.5).astype(int))
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['legit', 'fraud'], yticklabels=['legit', 'fraud'], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("best_model_confusion.png", dpi=150)
plt.close()

print("\nDone. Tomorrow: threshold tuning and final project README.")
