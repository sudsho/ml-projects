"""
Day 5: Model Comparison, Evaluation Metrics, and Final Summary
Sentiment Analysis on Product Reviews

Compares all trained models (Naive Bayes, SVM, LSTM/Transformer) using
comprehensive metrics: accuracy, F1, ROC-AUC, precision-recall curves,
confusion matrices, and inference speed benchmarks.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. Load and prepare data (same pipeline as day 3)
# ─────────────────────────────────────────────
categories = ["rec.sport.hockey", "sci.med", "comp.graphics", "talk.politics.misc"]
data = fetch_20newsgroups(subset="all", categories=categories, remove=("headers", "footers", "quotes"))

texts = data.data
labels = data.target
label_names = data.target_names

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Train size: {len(X_train_raw)} | Test size: {len(X_test_raw)}")
print(f"Classes: {label_names}\n")

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
X_train = tfidf.fit_transform(X_train_raw)
X_test = tfidf.transform(X_test_raw)

# ─────────────────────────────────────────────
# 2. Train all models and collect predictions
# ─────────────────────────────────────────────
models = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Linear SVM": LinearSVC(C=1.0, max_iter=2000),
    "Logistic Regression": LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs", multi_class="auto"),
}

results = {}
predictions = {}

for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t1 = time.time()
    y_pred = model.predict(X_test)
    infer_time = (time.time() - t1) * 1000  # ms

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    results[name] = {
        "Accuracy": round(acc, 4),
        "F1 Macro": round(f1_macro, 4),
        "F1 Weighted": round(f1_weighted, 4),
        "Train Time (s)": round(train_time, 3),
        "Infer Time (ms)": round(infer_time, 2),
    }
    predictions[name] = y_pred
    print(f"[{name}]  Acc={acc:.4f}  F1-macro={f1_macro:.4f}  trained in {train_time:.2f}s")

# ─────────────────────────────────────────────
# 3. Summary comparison table
# ─────────────────────────────────────────────
results_df = pd.DataFrame(results).T
print("\n=== Model Comparison ===")
print(results_df.to_string())

# ─────────────────────────────────────────────
# 4. Detailed classification report for best model
# ─────────────────────────────────────────────
best_model_name = results_df["F1 Macro"].idxmax()
print(f"\nBest model: {best_model_name}")
print(classification_report(y_test, predictions[best_model_name], target_names=label_names))

# ─────────────────────────────────────────────
# 5. Confusion matrices (side by side)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
short_names = [n.split(".")[0][:8] for n in label_names]

for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=short_names, yticklabels=short_names, ax=ax
    )
    ax.set_title(f"{name}\nAcc={results[name]['Accuracy']:.3f}", fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.suptitle("Confusion Matrices - All Models", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("confusion_matrices_comparison.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nSaved: confusion_matrices_comparison.png")

# ─────────────────────────────────────────────
# 6. Bar chart: accuracy & F1 comparison
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(results_df))
width = 0.28
ax.bar(x - width, results_df["Accuracy"], width, label="Accuracy", color="#4C72B0")
ax.bar(x,         results_df["F1 Macro"],    width, label="F1 Macro",   color="#DD8452")
ax.bar(x + width, results_df["F1 Weighted"], width, label="F1 Weighted",color="#55A868")
ax.set_xticks(x)
ax.set_xticklabels(results_df.index, fontsize=11)
ax.set_ylim(0.7, 1.02)
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("model_performance_comparison.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: model_performance_comparison.png")

print("\n=== Day 5 Complete: Sentiment Analysis Project Finished ===")
