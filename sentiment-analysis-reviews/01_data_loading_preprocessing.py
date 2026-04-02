# Sentiment Analysis on Product Reviews - Day 1
# Data collection, initial text preprocessing, and exploratory statistics.
# Uses the IMDb / Amazon-style review dataset from sklearn/nltk for offline access.

import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# nltk for tokenization and stopwords
import nltk
nltk.download("movie_reviews", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize

SEED = 42
np.random.seed(SEED)

# ------------------------------------------------------------------
# 1. Load data  (NLTK movie_reviews corpus → positive / negative)
# ------------------------------------------------------------------

print("Loading movie reviews corpus ...")

documents = []
for category in movie_reviews.categories():           # 'pos' / 'neg'
    for fileid in movie_reviews.fileids(category):
        raw_text = movie_reviews.raw(fileid)
        label = 1 if category == "pos" else 0
        documents.append({"text": raw_text, "label": label, "category": category})

df = pd.DataFrame(documents)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"Total reviews loaded : {len(df)}")
print(f"Positive (1): {df['label'].sum()} | Negative (0): {(df['label'] == 0).sum()}")
print(df.head(3))

# ------------------------------------------------------------------
# 2. Basic text statistics
# ------------------------------------------------------------------

df["char_count"] = df["text"].str.len()
df["word_count"] = df["text"].apply(lambda x: len(x.split()))
df["sentence_count"] = df["text"].apply(lambda x: len(re.findall(r'[.!?]+', x)) + 1)
df["avg_word_len"] = df["text"].apply(
    lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
)

print("\n--- Review Length Statistics ---")
print(df.groupby("category")[["char_count", "word_count", "sentence_count"]].describe().T)

# ------------------------------------------------------------------
# 3. Text cleaning pipeline
# ------------------------------------------------------------------

STOPWORDS = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """Lowercase, remove HTML/special chars, strip stopwords."""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)                      # strip HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)             # strip URLs
    text = re.sub(r"[^a-z\s]", " ", text)                   # keep letters only
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

print("\nCleaning texts ...")
df["clean_text"] = df["text"].apply(clean_text)
df["clean_word_count"] = df["clean_text"].apply(lambda x: len(x.split()))

print(f"Avg words before cleaning : {df['word_count'].mean():.1f}")
print(f"Avg words after  cleaning : {df['clean_word_count'].mean():.1f}")

# ------------------------------------------------------------------
# 4. Top tokens per class
# ------------------------------------------------------------------

def top_tokens(texts, n=20):
    all_words = " ".join(texts).split()
    return Counter(all_words).most_common(n)

pos_top = top_tokens(df.loc[df["label"] == 1, "clean_text"])
neg_top = top_tokens(df.loc[df["label"] == 0, "clean_text"])

print("\nTop 10 tokens in POSITIVE reviews:")
for word, cnt in pos_top[:10]:
    print(f"  {word:20s} {cnt}")

print("\nTop 10 tokens in NEGATIVE reviews:")
for word, cnt in neg_top[:10]:
    print(f"  {word:20s} {cnt}")

# ------------------------------------------------------------------
# 5. Visualizations
# ------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Sentiment Analysis - Day 1 EDA", fontsize=14, fontweight="bold")

# 5a. Label distribution
ax = axes[0, 0]
df["category"].value_counts().plot(kind="bar", ax=ax, color=["steelblue", "tomato"],
                                    edgecolor="black", width=0.5)
ax.set_title("Class Distribution")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Count")
ax.set_xticklabels(["Negative", "Positive"], rotation=0)

# 5b. Word count distribution by class
ax = axes[0, 1]
for cat, color in [("pos", "steelblue"), ("neg", "tomato")]:
    subset = df.loc[df["category"] == cat, "word_count"]
    ax.hist(subset, bins=40, alpha=0.6, color=color, label=cat)
ax.set_title("Word Count Distribution by Sentiment")
ax.set_xlabel("Word Count")
ax.set_ylabel("Frequency")
ax.legend()

# 5c. Top 15 positive tokens
ax = axes[1, 0]
words_p, counts_p = zip(*pos_top[:15])
ax.barh(words_p[::-1], counts_p[::-1], color="steelblue")
ax.set_title("Top 15 Tokens - Positive Reviews")
ax.set_xlabel("Frequency")

# 5d. Top 15 negative tokens
ax = axes[1, 1]
words_n, counts_n = zip(*neg_top[:15])
ax.barh(words_n[::-1], counts_n[::-1], color="tomato")
ax.set_title("Top 15 Tokens - Negative Reviews")
ax.set_xlabel("Frequency")

plt.tight_layout()
plt.savefig("eda_day1.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nSaved eda_day1.png")

# ------------------------------------------------------------------
# 6. Save cleaned data for next days
# ------------------------------------------------------------------

df[["text", "clean_text", "label", "category",
    "word_count", "clean_word_count", "char_count"]].to_csv(
    "reviews_clean.csv", index=False
)
print("Saved reviews_clean.csv  — shape:", df.shape)
print("\nDay 1 complete. Next: word clouds, bigrams, and deeper EDA.")
