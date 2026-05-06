# Extractive & Abstractive Text Summarization

NLP project comparing classical extractive summarization (TextRank with TF-IDF
sentence similarity) against modern abstractive summarization with pre-trained
transformers (BART, Pegasus). All three approaches are evaluated head-to-head
on the same held-out documents using ROUGE F1 scores.

## Pipeline

1. **`01_data_prep_preprocessing.py`** — corpus loading, sentence splitting,
   token cleaning, train/eval split.
2. **`02_textrank_extractive.py`** — graph-based ranking over sentence-level
   TF-IDF cosine similarity, picking top-k sentences.
3. **`03_abstractive_transformer.py`** — wraps `facebook/bart-large-cnn` and
   `google/pegasus-xsum` with chunking for long inputs.
4. **`04_rouge_evaluation_and_comparison.py`** — ROUGE-1/2/L F1 scoring,
   per-system runtime, and leaderboard generation.

## Results

Average scores on a 50-doc evaluation slice:

| System         | ROUGE-1 | ROUGE-2 | ROUGE-L | sec/doc |
|----------------|---------|---------|---------|---------|
| bart-large-cnn |   0.412 |   0.187 |   0.378 |    2.41 |
| pegasus-xsum   |   0.395 |   0.165 |   0.361 |    2.18 |
| textrank       |   0.336 |   0.117 |   0.302 |    0.05 |

(Final numbers vary slightly with the random eval split; rerun
`04_rouge_evaluation_and_comparison.py` to regenerate `results/`.)

## Takeaways

- BART edges out Pegasus on this dataset; Pegasus tends to write shorter,
  punchier summaries, which hurts unigram recall against longer references.
- TextRank is ~50x faster than the transformer pipelines and still hits a
  respectable ROUGE-L. For latency-sensitive use cases it's a real option.
- ROUGE under-credits abstractive paraphrase. BART summaries that re-word
  the reference frequently lose bigram overlap despite being readable.

## Running

```bash
pip install -r ../requirements.txt
python 01_data_prep_preprocessing.py
python 04_rouge_evaluation_and_comparison.py
```

Outputs land under `results/` as `rows.json`, `summary.json`, and a plain-text
`leaderboard.txt`.
