# Sentiment Analysis on Product Reviews

NLP project exploring text classification using TF-IDF with traditional ML models and a deep learning approach, evaluated on multi-class text data.

## Project Overview

Built an end-to-end sentiment/topic classification pipeline, comparing classical ML models against sequence-based deep learning. Used the 20 Newsgroups dataset as a proxy for product review text classification.

## Project Structure

| File | Day | Description |
|------|-----|-------------|
| `01_data_loading_preprocessing.py` | Day 1 | Text cleaning, tokenization, stopword removal |
| `02_eda_analysis.py` | Day 2 | Word clouds, class distribution, text length stats |
| `03_tfidf_ml_models.py` | Day 3 | TF-IDF + Naive Bayes, SVM, Logistic Regression |
| `04_lstm_transformer_model.py` | Day 4 | LSTM with PyTorch, embedding layer, training loop |
| `05_model_comparison_evaluation.py` | Day 5 | Full comparison, confusion matrices, metric charts |

## Key Results

| Model | Accuracy | F1 (Macro) |
|-------|----------|------------|
| Naive Bayes | ~0.91 | ~0.91 |
| Linear SVM | ~0.95 | ~0.95 |
| Logistic Regression | ~0.95 | ~0.95 |

- **Best performer**: Linear SVM and Logistic Regression tied at ~95% accuracy
- **Fastest inference**: Naive Bayes (~1ms)
- **Key insight**: TF-IDF + linear models are highly competitive with deep learning for this task size

## Tech Stack

- `scikit-learn` — TF-IDF, Naive Bayes, SVM, Logistic Regression, metrics
- `PyTorch` — LSTM model with embedding layer
- `pandas`, `numpy` — data wrangling
- `matplotlib`, `seaborn` — visualizations (confusion matrices, metric comparison)
- `nltk` — text preprocessing

## Outputs

- `confusion_matrices_comparison.png` — side-by-side confusion matrices for all models
- `model_performance_comparison.png` — grouped bar chart of accuracy / F1 scores
