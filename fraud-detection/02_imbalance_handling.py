# %% markdown
# ## Credit Card Fraud Detection - Day 2: Handling Severe Class Imbalance
# Fraud is typically <1% of transactions. Standard ML models will just predict
# "not fraud" everywhere and get 99% accuracy. We need explicit techniques.
#
# Strategies compared today:
#   1. Random undersampling (drop majority class samples)
#   2. SMOTE (synthetic minority oversampling)
#   3. Class weights (weight loss to penalize false negatives more)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    average_precision_score, roc_auc_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# %% markdown
# ### Load (synthetic data mimicking the credit card fraud dataset shape)

# Real-world target sparsity: ~0.17% fraud
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

print(f"Training shape: {X_train.shape}")
print(f"Train fraud rate:  {y_train.mean() * 100:.3f}%")
print(f"Test fraud rate:   {y_test.mean() * 100:.3f}%")

# %% markdown
# ### Strategy 1: Baseline (no balancing)

baseline = LogisticRegression(max_iter=1000, random_state=42)
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)
y_prob_base = baseline.predict_proba(X_test)[:, 1]

print("\n=== Baseline (no balancing) ===")
print(classification_report(y_test, y_pred_base, digits=4, zero_division=0))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_base):.4f}")
print(f"PR-AUC:  {average_precision_score(y_test, y_prob_base):.4f}")

# %% markdown
# ### Strategy 2: Random Undersampling

rus = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
X_under, y_under = rus.fit_resample(X_train, y_train)
print(f"\nAfter undersampling: {len(X_under):,} samples, "
      f"fraud rate = {y_under.mean() * 100:.2f}%")

m_under = LogisticRegression(max_iter=1000, random_state=42)
m_under.fit(X_under, y_under)
y_prob_under = m_under.predict_proba(X_test)[:, 1]
print(f"PR-AUC (undersampled): {average_precision_score(y_test, y_prob_under):.4f}")

# %% markdown
# ### Strategy 3: SMOTE oversampling

smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE: {len(X_smote):,} samples, "
      f"fraud rate = {y_smote.mean() * 100:.2f}%")

m_smote = LogisticRegression(max_iter=1000, random_state=42)
m_smote.fit(X_smote, y_smote)
y_prob_smote = m_smote.predict_proba(X_test)[:, 1]
print(f"PR-AUC (SMOTE): {average_precision_score(y_test, y_prob_smote):.4f}")

# %% markdown
# ### Strategy 4: Class weights (cheapest, often surprisingly effective)

m_weighted = LogisticRegression(
    max_iter=1000, class_weight='balanced', random_state=42
)
m_weighted.fit(X_train, y_train)
y_prob_w = m_weighted.predict_proba(X_test)[:, 1]
print(f"\nPR-AUC (class_weight=balanced): {average_precision_score(y_test, y_prob_w):.4f}")

# %% markdown
# ### Compare PR curves

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
for label, probs in [
    ("baseline", y_prob_base),
    ("undersampled", y_prob_under),
    ("smote", y_prob_smote),
    ("class_weight", y_prob_w),
]:
    p, r, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    ax.plot(r, p, label=f"{label} (AP={ap:.3f})", linewidth=2)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall: imbalance handling strategies")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("pr_curves_imbalance.png", dpi=150)
plt.close()
print("\nSaved pr_curves_imbalance.png")

# %% markdown
# ### Takeaways
#
# - Accuracy is misleading - all strategies hit ~99.7%, but PR-AUC differs significantly
# - SMOTE typically wins on PR-AUC for moderate imbalance, but can introduce noise
# - class_weight='balanced' is a free win - just one parameter, often within 1-2% of SMOTE
# - In production, combine: balanced class_weight + threshold tuning on validation set
#
# TODO: Tomorrow we train Random Forest / XGBoost / Isolation Forest with these strategies
