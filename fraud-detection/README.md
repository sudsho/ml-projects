# Credit Card Fraud Detection

Binary classification on a heavily imbalanced dataset (~0.2% positives), with a focus on the parts that actually move the needle in production: handling class imbalance, picking the right metric, and tuning the decision threshold to a real cost model.

## Problem

Credit card fraud is rare but expensive. A fraud-detection model that achieves 99.8% accuracy by predicting "legit" for everything is worthless. We care about catching as many fraudulent transactions as possible while keeping false positives low enough that customers and the support team don't drown in friction.

Key constraints:
- ~0.2% of transactions are fraudulent
- Cost asymmetry: a missed fraud is roughly 40x more expensive than a false alarm
- The right operating point depends on business policy, not on a default 0.5 cutoff

## Dataset

Synthetic data generated with `sklearn.datasets.make_classification`:
- 50,000 samples, 30 features (10 informative, 5 redundant)
- Class weights `[0.998, 0.002]` to mimic realistic fraud rates
- Random label flips (`flip_y=0.005`) to add noise

The same generation parameters are reused across all four scripts so models are directly comparable.

## Pipeline

| File | Purpose |
|------|---------|
| `01_data_loading_eda.py` | Load data, visualize the imbalance, basic feature distributions |
| `02_imbalance_handling.py` | SMOTE, random undersampling, class-weight comparison |
| `03_model_training.py` | Train Random Forest, XGBoost, Isolation Forest; compare PR-AUC |
| `04_threshold_tuning.py` | Threshold sweep, cost analysis, final operating point |

## Results

### Model comparison (PR-AUC on held-out test set)

| Model | PR-AUC | ROC-AUC | F1 @ 0.5 |
|-------|--------|---------|----------|
| XGBoost (scale_pos_weight) | ~0.89 | ~0.99 | ~0.81 |
| Random Forest (balanced) | ~0.85 | ~0.99 | ~0.78 |
| Isolation Forest (unsupervised) | ~0.42 | ~0.94 | ~0.31 |

XGBoost wins on PR-AUC, which is the metric that matters under heavy class imbalance. Isolation Forest is included as a baseline - it's useful when labels are unavailable, but supervised models clearly dominate when labels exist.

### Threshold tuning

The default 0.5 threshold is rarely correct. Sweeping thresholds from 0.05 to 0.95 and applying the cost model (FP=$5, FN=$200) shows the optimal operating point sits well below 0.5.

Three reasonable policies, each picks a different threshold:

| Policy | Threshold | Precision | Recall | Notes |
|--------|-----------|-----------|--------|-------|
| Max F1 | ~0.30 | ~0.83 | ~0.79 | Balanced |
| Min expected cost | ~0.20 | ~0.71 | ~0.85 | Catches more fraud, accepts more friction |
| Precision >= 0.90 | ~0.45 | ~0.91 | ~0.71 | Conservative - few false alarms |

The "right" answer depends on business policy. The threshold-tuning script makes that choice explicit and inspectable rather than burying it in a default.

## Key takeaways

1. **Accuracy is meaningless here.** Always look at PR-AUC and the confusion matrix when the positive class is rare.
2. **SMOTE helps less than you'd expect.** For tree-based models, `class_weight='balanced'` and `scale_pos_weight` give comparable or better results without synthetic samples polluting the training distribution.
3. **Threshold tuning > model tuning.** Once you have a reasonable model, picking the operating point under a real cost function yields bigger gains than another round of hyperparameter search.
4. **Isolation Forest as a fallback.** When labels aren't available or are very stale, anomaly detection still recovers most of the signal - useful for cold-start.

## How to run

```bash
pip install scikit-learn xgboost imbalanced-learn pandas matplotlib seaborn
python 01_data_loading_eda.py
python 02_imbalance_handling.py
python 03_model_training.py
python 04_threshold_tuning.py
```
