"""
Day 1 - Dataset preparation and text preprocessing pipeline
For text summarization project (extractive + abstractive comparison).

Using the CNN/DailyMail style setup but with a smaller subset so iterations
stay fast on a laptop. The preprocessing here focuses on getting clean,
sentence-tokenized text that works for both TextRank (extractive) and
transformer-based abstractive models later.
"""

import re
import json
import string
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# download required nltk resources once
for pkg in ["punkt", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

STOPWORDS = set(stopwords.words("english"))


def load_raw_articles(path: Path) -> pd.DataFrame:
    """Load raw article-summary pairs. Expects a parquet or csv with
    columns ['article', 'highlights']."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def clean_text(text: str) -> str:
    """Light cleaning - keep punctuation since downstream sentence
    tokenization needs it. Strip html-ish artifacts and normalize whitespace."""
    if not isinstance(text, str):
        return ""
    # remove urls
    text = re.sub(r"http\S+", "", text)
    # remove (CNN) prefix and similar wire tags
    text = re.sub(r"^\s*\(?[A-Z]{2,5}\)?\s*--?\s*", "", text)
    # collapse repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_sentences(text: str) -> list:
    """Sentence split with a small post-filter for stubs."""
    sents = sent_tokenize(text)
    return [s.strip() for s in sents if len(s.strip()) > 5]


def basic_stats(df: pd.DataFrame) -> dict:
    """Compute corpus stats - useful for sanity checks before expensive runs."""
    art_lens = df["article"].str.split().str.len()
    sum_lens = df["highlights"].str.split().str.len()
    return {
        "n_examples": int(len(df)),
        "article_words_mean": float(art_lens.mean()),
        "article_words_p95": float(art_lens.quantile(0.95)),
        "summary_words_mean": float(sum_lens.mean()),
        "compression_ratio": float((sum_lens / art_lens).mean()),
    }


def build_vocab(texts, min_count: int = 5) -> dict:
    """Build a word->index vocab over the corpus, skipping pure-punctuation
    tokens and any word seen fewer than ``min_count`` times. Indices are
    assigned by descending frequency, so common words get low ids."""
    counter = Counter()
    for t in texts:
        counter.update(w.lower() for w in word_tokenize(t) if w not in string.punctuation)
    vocab = {w: i for i, (w, c) in enumerate(counter.most_common()) if c >= min_count}
    return vocab


def preprocess_for_extractive(article: str) -> list:
    """Returns list of (sentence, tokens_no_stopwords) tuples for TextRank."""
    sents = tokenize_sentences(clean_text(article))
    out = []
    for s in sents:
        tokens = [
            w.lower()
            for w in word_tokenize(s)
            if w.isalpha() and w.lower() not in STOPWORDS
        ]
        if tokens:
            out.append((s, tokens))
    return out


def save_processed(df: pd.DataFrame, out_path: Path) -> None:
    """Save as jsonl - works well with HF datasets and streaming readers."""
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({
                "article": row["article"],
                "summary": row["highlights"],
                "n_sents": len(tokenize_sentences(row["article"])),
            }) + "\n")


def main():
    # in a real run this comes from HF datasets - placeholder for now
    sample_path = DATA_DIR / "sample_articles.csv"
    if not sample_path.exists():
        # synthetic tiny sample so the pipeline can be smoke-tested
        demo = pd.DataFrame({
            "article": [
                "The quick brown fox jumps over the lazy dog. " * 8,
                "Researchers at the lab published findings today. " * 6,
            ],
            "highlights": ["Fox jumps over dog.", "Lab publishes findings."],
        })
        demo.to_csv(sample_path, index=False)

    df = load_raw_articles(sample_path)
    df["article"] = df["article"].map(clean_text)
    df["highlights"] = df["highlights"].map(clean_text)
    df = df[df["article"].str.split().str.len() > 30].reset_index(drop=True)

    stats = basic_stats(df)
    print("corpus stats:", stats)

    vocab = build_vocab(df["article"].tolist())
    print(f"vocab size (min_count=5): {len(vocab)}")

    save_processed(df, DATA_DIR / "processed.jsonl")
    print(f"wrote {len(df)} examples to {DATA_DIR/'processed.jsonl'}")


if __name__ == "__main__":
    main()
