"""
Sentiment Analysis on Product Reviews - Day 3
TF-IDF vectorization + traditional ML models: Naive Bayes, Logistic Regression, SVM.
Compare models with cross-validation, classification reports, and confusion matrices.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import nltk
nltk.download("movie_reviews", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize

import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# ── 1. Load and clean data (same pipeline as Day 1) ──────────────────────────

print("Loading movie_reviews corpus ...")
STOPWORDS_EN = set(stopwords.words("english"))

documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        raw_text = movie_reviews.raw(fileid)
        label = 1 if category == "pos" else 0
        documents.append({"text": raw_text, "label": label})

df = pd.DataFrame(documents).sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"Loaded {len(df):,} reviews | pos={df['label'].sum()} neg={(df['label']==0).sum()}")


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS_EN and len(t) > 2]
    return " ".join(tokens)


print("Cleaning texts ...")
df["clean_text"] = df["text"].apply(clean_text)

# ── 2. Train / test split ─────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=SEED, stratify=df["label"]
)
print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

# ── 3. TF-IDF vectorizer ──────────────────────────────────────────────────────

tfidf = TfidfVectorizer(
    max_features=30_000,
    ngram_range=(1, 2),       # unigrams + bigrams
    min_df=2,
    sublinear_tf=True,        # apply log(1+tf)
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"\nTF-IDF vocabulary size: {len(tfidf.vocabulary_):,}")
print(f"Feature matrix shape (train): {X_train_tfidf.shape}")

# ── 4. Build and evaluate models ──────────────────────────────────────────────

models = {
    "Naive Bayes (Complement)": ComplementNB(alpha=0.1),
    "Logistic Regression":      LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
    "Linear SVM":               LinearSVC(C=0.5, max_iter=2000, random_state=SEED),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = {}

print("\n── 5-Fold Cross-Validation on Training Set ──")
for name, clf in models.items():
    scores = cross_val_score(clf, X_train_tfidf, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:<30s}  acc = {scores.mean():.4f} ± {scores.std():.4f}")

# ── 5. Fit on full train, evaluate on test set ────────────────────────────────

print("\n── Test Set Evaluation ──")
test_metrics = {}
fitted_models = {}

for name, clf in models.items():
    clf.fit(X_train_tfidf, y_train)
    fitted_models[name] = clf
    y_pred = clf.predict(X_test_tfidf)
    report = classification_report(y_test, y_pred, target_names=["negative", "positive"],
                                   output_dict=True)
    acc = report["accuracy"]
    f1  = report["macro avg"]["f1-score"]
    test_metrics[name] = {"accuracy": acc, "f1_macro": f1}
    print(f"\n{name}")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

# ── 6. Confusion matrices ─────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Confusion Matrices - TF-IDF Models", fontsize=14, fontweight="bold")

for ax, (name, clf) in zip(axes, fitted_models.items()):
    y_pred = clf.predict(X_test_tfidf)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=10)

plt.tight_layout()
plt.savefig("confusion_matrices_day3.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nSaved confusion_matrices_day3.png")

# ── 7. CV score comparison bar chart ─────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
names = list(cv_results.keys())
means = [cv_results[n].mean() for n in names]
stds  = [cv_results[n].std()  for n in names]

bars = ax.barh(names, means, xerr=stds, color=["#3498db", "#2ecc71", "#e74c3c"],
               alpha=0.85, height=0.5, capsize=5)
ax.set_xlabel("5-Fold CV Accuracy")
ax.set_title("Model Comparison - TF-IDF + Traditional ML")
ax.set_xlim(0.8, 1.0)

for bar, mean in zip(bars, means):
    ax.text(mean + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{mean:.4f}", va="center", fontsize=10)

plt.tight_layout()
plt.savefig("model_comparison_day3.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved model_comparison_day3.png")

# ── 8. Top TF-IDF features per class ─────────────────────────────────────────

lr = fitted_models["Logistic Regression"]
feature_names = np.array(tfidf.get_feature_names_out())

top_n = 20
coef = lr.coef_[0]
top_pos_idx = np.argsort(coef)[-top_n:][::-1]
top_neg_idx = np.argsort(coef)[:top_n]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].barh(feature_names[top_pos_idx][::-1], coef[top_pos_idx][::-1], color="#2ecc71", alpha=0.8)
axes[0].set_title("Top 20 Positive Sentiment Features (LR coefficients)")
axes[0].set_xlabel("Coefficient")

axes[1].barh(feature_names[top_neg_idx][::-1], np.abs(coef[top_neg_idx][::-1]),
             color="#e74c3c", alpha=0.8)
axes[1].set_title("Top 20 Negative Sentiment Features (|LR coefficients|)")
axes[1].set_xlabel("|Coefficient|")

plt.tight_layout()
plt.savefig("top_features_day3.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved top_features_day3.png")

# ── 9. Summary ────────────────────────────────────────────────────────────────

print("\n── Summary ──")
summary_df = pd.DataFrame(test_metrics).T
print(summary_df.round(4))

best_model = summary_df["accuracy"].idxmax()
print(f"\nBest model: {best_model}  (test acc = {summary_df.loc[best_model,'accuracy']:.4f})")
print("\nDay 3 complete. Next: LSTM/Transformer approach with PyTorch.")
