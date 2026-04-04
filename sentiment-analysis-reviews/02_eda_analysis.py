"""
Sentiment Analysis on Product Reviews - Day 2
EDA: word clouds, rating distributions, text statistics, and vocabulary analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os

# WordCloud is optional - gracefully skip if not installed
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── load cleaned data ────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "reviews_clean.csv")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} reviews from {DATA_PATH}")
else:
    # Regenerate the synthetic dataset used in day 1
    from sklearn.datasets import fetch_20newsgroups

    positive_templates = [
        "this product is absolutely amazing and works perfectly",
        "great quality highly recommend to everyone",
        "excellent item fast shipping very happy with purchase",
        "love this product works exactly as described",
        "best purchase I have made in a long time fantastic",
        "wonderful product exceeded all my expectations",
        "superb quality and great value for money",
        "very satisfied with this item will buy again",
    ]
    negative_templates = [
        "terrible product broke after one day complete waste of money",
        "very disappointed with quality do not buy this item",
        "worst purchase ever stopped working immediately",
        "poor quality material feels cheap not worth it",
        "avoid this product it is a scam total garbage",
        "horrible experience product arrived damaged and useless",
        "absolute junk returned it immediately very unhappy",
        "bad quality fell apart quickly would not recommend",
    ]
    neutral_templates = [
        "product is okay nothing special average quality",
        "decent item does the job but nothing impressive",
        "its fine I guess meets basic requirements",
        "average product not great but not terrible either",
        "mediocre quality works as expected nothing more",
    ]

    np.random.seed(42)
    n = 3000
    labels, texts, ratings = [], [], []
    for _ in range(n):
        r = np.random.choice(["positive", "negative", "neutral"],
                             p=[0.55, 0.30, 0.15])
        if r == "positive":
            t = np.random.choice(positive_templates)
            rat = np.random.choice([4, 5], p=[0.3, 0.7])
        elif r == "negative":
            t = np.random.choice(negative_templates)
            rat = np.random.choice([1, 2], p=[0.6, 0.4])
        else:
            t = np.random.choice(neutral_templates)
            rat = 3
        labels.append(r)
        texts.append(t)
        ratings.append(rat)

    df = pd.DataFrame({"text": texts, "sentiment": labels, "rating": ratings})
    df["cleaned_text"] = df["text"].str.lower().str.replace(r"[^a-z\s]", "", regex=True)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Regenerated {len(df):,} synthetic reviews")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Rating distribution ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(x="rating", data=df, palette="coolwarm", ax=axes[0])
axes[0].set_title("Rating Distribution (1-5 Stars)")
axes[0].set_xlabel("Star Rating")
axes[0].set_ylabel("Count")
for bar in axes[0].patches:
    axes[0].annotate(
        f"{int(bar.get_height())}",
        (bar.get_x() + bar.get_width() / 2, bar.get_height() + 10),
        ha="center", va="bottom", fontsize=9,
    )

sentiment_counts = df["sentiment"].value_counts()
axes[1].pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct="%1.1f%%",
    colors=["#2ecc71", "#e74c3c", "#95a5a6"],
    startangle=90,
)
axes[1].set_title("Sentiment Class Distribution")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rating_distribution.png"), dpi=120)
plt.close()
print("Saved: rating_distribution.png")

# ── 2. Text length statistics ────────────────────────────────────────────────
df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
df["char_count"] = df["text"].apply(len)

print("\n── Text Length Statistics by Sentiment ──")
print(df.groupby("sentiment")[["word_count", "char_count"]].agg(["mean", "median", "std"]).round(2))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col, label in zip(
    axes,
    ["word_count", "char_count"],
    ["Word Count", "Character Count"],
):
    for sentiment, color in zip(["positive", "negative", "neutral"],
                                 ["#2ecc71", "#e74c3c", "#95a5a6"]):
        subset = df[df["sentiment"] == sentiment][col]
        ax.hist(subset, bins=20, alpha=0.6, label=sentiment, color=color)
    ax.set_title(f"{label} Distribution by Sentiment")
    ax.set_xlabel(label)
    ax.set_ylabel("Frequency")
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "text_length_distribution.png"), dpi=120)
plt.close()
print("Saved: text_length_distribution.png")

# ── 3. Top words per sentiment ───────────────────────────────────────────────
STOPWORDS = {
    "the", "a", "an", "is", "it", "this", "and", "or", "but", "in", "on",
    "at", "to", "for", "of", "with", "as", "be", "was", "are", "not",
    "i", "my", "me", "its", "that", "have", "has", "had",
}


def get_top_words(texts, n=20):
    words = []
    for text in texts:
        tokens = re.findall(r"\b[a-z]{3,}\b", str(text).lower())
        words.extend([w for w in tokens if w not in STOPWORDS])
    return Counter(words).most_common(n)


fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, sentiment, color in zip(
    axes, ["positive", "negative", "neutral"], ["#27ae60", "#c0392b", "#7f8c8d"]
):
    subset = df[df["sentiment"] == sentiment]["text"]
    top = get_top_words(subset, n=15)
    words, counts = zip(*top)
    ax.barh(words[::-1], counts[::-1], color=color, alpha=0.8)
    ax.set_title(f"Top Words — {sentiment.capitalize()}")
    ax.set_xlabel("Frequency")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_words_by_sentiment.png"), dpi=120)
plt.close()
print("Saved: top_words_by_sentiment.png")

# ── 4. Word clouds (if available) ───────────────────────────────────────────
if HAS_WORDCLOUD:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    color_map = {"positive": "Greens", "negative": "Reds", "neutral": "Greys"}
    for ax, sentiment in zip(axes, ["positive", "negative", "neutral"]):
        text_blob = " ".join(df[df["sentiment"] == sentiment]["text"].tolist())
        wc = WordCloud(
            width=600, height=400, background_color="white",
            colormap=color_map[sentiment], stopwords=STOPWORDS, max_words=80,
        ).generate(text_blob)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud — {sentiment.capitalize()}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "word_clouds.png"), dpi=120)
    plt.close()
    print("Saved: word_clouds.png")
else:
    print("Skipping word clouds (wordcloud package not installed)")

# ── 5. Rating vs sentiment heatmap ──────────────────────────────────────────
pivot = df.groupby(["rating", "sentiment"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.5)
ax.set_title("Rating vs Sentiment Heatmap")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Star Rating")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rating_sentiment_heatmap.png"), dpi=120)
plt.close()
print("Saved: rating_sentiment_heatmap.png")

# ── 6. Vocabulary statistics ─────────────────────────────────────────────────
all_words = []
for text in df["text"]:
    tokens = re.findall(r"\b[a-z]{3,}\b", str(text).lower())
    all_words.extend([w for w in tokens if w not in STOPWORDS])

vocab = set(all_words)
word_freq = Counter(all_words)

print("\n── Vocabulary Statistics ──")
print(f"Total tokens (no stopwords): {len(all_words):,}")
print(f"Unique vocabulary size:       {len(vocab):,}")
print(f"Avg tokens per review:        {len(all_words)/len(df):.1f}")
print(f"Top 10 words overall:         {word_freq.most_common(10)}")

# Words appearing only once (hapax legomena)
hapax = sum(1 for w, c in word_freq.items() if c == 1)
print(f"Hapax legomena (count=1):     {hapax:,} ({hapax/len(vocab)*100:.1f}% of vocab)")

# ── summary ──────────────────────────────────────────────────────────────────
print("\n── EDA Complete ──")
print(f"Dataset shape: {df.shape}")
print(f"Sentiment breakdown:\n{df['sentiment'].value_counts()}")
print(f"Plots saved to: {OUTPUT_DIR}")
