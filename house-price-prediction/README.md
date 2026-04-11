# House Price Prediction

Regression project with advanced feature engineering on the California Housing dataset. Explores multiple model families — from regularized linear models to tree-based ensembles — and finishes with stacking to squeeze out the best predictive accuracy.

## Project Structure

```
house-price-prediction/
├── 01_eda_data_loading.py       # EDA, correlation heatmap, distribution plots
├── 02_feature_engineering.py    # Encoding, scaling, interaction features
├── 03_model_training.py         # Ridge, Lasso, Gradient Boosting comparison
├── 04_ensemble_final.py         # Stacking, Voting ensemble, final evaluation
└── README.md
```

## Daily Breakdown

| Day | Focus |
|-----|-------|
| 1   | Dataset loading, EDA, correlation analysis, visualizations |
| 2   | Feature engineering — encoding, scaling, feature selection |
| 3   | Model training — Linear Regression, Ridge, Lasso, Gradient Boosting |
| 4   | Ensemble methods, permutation importance, final evaluation |

## Results Summary

| Model              | Val RMSE | Val R²  |
|--------------------|----------|---------|
| Linear Regression  | ~0.155   | ~0.79   |
| Ridge              | ~0.154   | ~0.79   |
| Lasso              | ~0.155   | ~0.79   |
| Gradient Boosting  | ~0.118   | ~0.87   |
| Voting Ensemble    | ~0.120   | ~0.86   |
| Stacking Ensemble  | ~0.116   | ~0.88   |

*Predictions are on log-transformed prices. Best stacking ensemble achieves ~$19k MAE on real dollar values.*

## Key Findings

- **Feature engineering** (log transform, ratio features, binned location) improved GBM R² by ~0.04 over raw features.
- **Regularization** (Ridge/Lasso) adds minimal value over plain Linear Regression for this dataset.
- **Gradient Boosting** outperforms linear models by a large margin — housing prices have nonlinear structure.
- **Stacking** with Ridge meta-learner gives a small but consistent improvement over individual boosting.
- **Permutation importance** confirms `median_income` is by far the strongest predictor (~3× more than the next feature).

## Dependencies

```
numpy pandas scikit-learn matplotlib seaborn
```
