# Titanic Survival Prediction

Classic binary classification project on the Titanic dataset. The goal is to predict whether a passenger survived using passenger attributes.

## Project Overview

| Day | File | Description |
|-----|------|-------------|
| 1 | `eda.py` | Exploratory data analysis and visualizations |
| 2 | `02_feature_engineering.py` | Feature engineering and preprocessing pipeline |
| 3 | `03_model_training.py` | Baseline model comparison (LR, RF, GBT) |
| 4 | `04_hyperparameter_tuning.py` | GridSearchCV tuning and final evaluation |

## Dataset

- **Source**: Seaborn's built-in Titanic dataset (or kaggle CSV)
- **Size**: 891 rows, 15 original features
- **Target**: `survived` (0 = died, 1 = survived)
- **Class imbalance**: ~38% survived

## Feature Engineering

- **FamilySize** = SibSp + Parch + 1
- **FarePerPerson** = Fare / FamilySize
- **AgeClass** = Age × Pclass (interaction term)
- Median imputation for Age/Fare; mode for Embarked
- One-hot encoding for Embarked; binary encoding for Sex

## Models Compared

| Model | CV ROC-AUC |
|-------|-----------|
| Logistic Regression | ~0.84 |
| Random Forest | ~0.86 |
| Gradient Boosting | ~0.87 |

## Key Findings

- **Sex** and **Pclass** are the strongest predictors (women and 1st class survived at higher rates)
- **FarePerPerson** outperforms raw Fare as a socioeconomic proxy
- Gradient Boosting performs best after hyperparameter tuning
- Family size has a non-linear effect: traveling alone or in very large groups reduces survival odds

## Requirements

```
pandas
numpy
scikit-learn
seaborn
matplotlib
```

## Usage

Run scripts in order:

```bash
python eda.py
python 02_feature_engineering.py
python 03_model_training.py
python 04_hyperparameter_tuning.py
```
