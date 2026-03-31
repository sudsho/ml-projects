"""
Day 3: Model Training - Logistic Regression, Random Forest, XGBoost
Compare baseline models and evaluate on held-out validation set.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)


def load_preprocessed_data():
    """Load the cleaned features from feature engineering step."""
    # If the preprocessed CSV exists, load it; otherwise rebuild from seaborn
    try:
        df = pd.read_csv("titanic_processed.csv")
        print(f"Loaded preprocessed data: {df.shape}")
        return df
    except FileNotFoundError:
        pass

    # fallback: reproduce preprocessing inline
    titanic = sns.load_dataset("titanic")
    titanic = titanic.drop(columns=["deck", "embark_town", "alive", "who",
                                     "adult_male", "class"])

    titanic["age"].fillna(titanic["age"].median(), inplace=True)
    titanic["embarked"].fillna(titanic["embarked"].mode()[0], inplace=True)
    titanic["fare"].fillna(titanic["fare"].median(), inplace=True)

    titanic["sex"] = (titanic["sex"] == "male").astype(int)
    titanic["alone"] = titanic["alone"].astype(int)
    titanic = pd.get_dummies(titanic, columns=["embarked"], drop_first=True)

    titanic["family_size"] = titanic["sibsp"] + titanic["parch"] + 1
    titanic["fare_per_person"] = titanic["fare"] / titanic["family_size"]
    titanic["age_class"] = titanic["age"] * titanic["pclass"]

    titanic = titanic.drop(columns=["sibsp", "parch"])
    return titanic


def prepare_features(df):
    target = "survived"
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].values
    y = df[target].values
    return X, y, feature_cols


def evaluate_model(name, model, X, y, cv):
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"\n{name}")
    print(f"  Accuracy : {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  ROC-AUC  : {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
    return scores.mean(), auc_scores.mean()


def plot_feature_importance(model, feature_names, model_name):
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances.values[:10], y=importances.index[:10],
                palette="viridis")
    plt.title(f"Top 10 Feature Importances - {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"feature_importance_{model_name.lower().replace(' ', '_')}.png",
                dpi=120)
    plt.close()
    print(f"  Saved feature importance plot for {model_name}")


def plot_roc_curves(models_dict, X, y):
    """Plot ROC curves for all fitted models on the full dataset."""
    plt.figure(figsize=(8, 6))
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        else:
            proba = model.decision_function(X)
        fpr, tpr, _ = roc_curve(y, proba)
        auc = roc_auc_score(y, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curves_comparison.png", dpi=120)
    plt.close()
    print("Saved roc_curves_comparison.png")


def plot_confusion_matrices(models_dict, X, y):
    fig, axes = plt.subplots(1, len(models_dict), figsize=(5 * len(models_dict), 4))
    if len(models_dict) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models_dict.items()):
        preds = model.predict(X)
        cm = confusion_matrix(y, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Not Survived", "Survived"],
                    yticklabels=["Not Survived", "Survived"])
        ax.set_title(name)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=120)
    plt.close()
    print("Saved confusion_matrices.png")


def main():
    print("=" * 60)
    print("TITANIC - Day 3: Model Training & Comparison")
    print("=" * 60)

    df = load_preprocessed_data()
    X, y, feature_names = prepare_features(df)

    # scale features for logistic regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ── define models ─────────────────────────────────────────────────────────
    log_reg = LogisticRegression(max_iter=1000, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                min_samples_leaf=4, random_state=SEED)
    gbt = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                     max_depth=4, random_state=SEED)

    results = {}

    print("\n--- Cross-Validation Results ---")
    acc_lr, auc_lr = evaluate_model("Logistic Regression", log_reg, X_scaled, y, cv)
    results["Logistic Regression"] = (acc_lr, auc_lr)

    acc_rf, auc_rf = evaluate_model("Random Forest", rf, X, y, cv)
    results["Random Forest"] = (acc_rf, auc_rf)

    acc_gbt, auc_gbt = evaluate_model("Gradient Boosting", gbt, X, y, cv)
    results["Gradient Boosting"] = (acc_gbt, auc_gbt)

    # ── fit on full data for plots ────────────────────────────────────────────
    log_reg.fit(X_scaled, y)
    rf.fit(X, y)
    gbt.fit(X, y)

    fitted_models = {
        "Logistic Regression": log_reg,
        "Random Forest": rf,
        "Gradient Boosting": gbt,
    }

    # feature importances for tree-based models
    plot_feature_importance(rf, feature_names, "Random Forest")
    plot_feature_importance(gbt, feature_names, "Gradient Boosting")

    # ROC curves and confusion matrices (on training data — for visual inspection)
    # TODO: use a proper train/test split in the tuning step tomorrow
    plot_roc_curves(
        {"Logistic Regression": log_reg, "Random Forest": rf, "Gradient Boosting": gbt},
        # LR needs scaled data — use scaled for all here for simplicity
        X_scaled, y
    )
    plot_confusion_matrices(
        {"Logistic Regression": log_reg, "Random Forest": rf, "Gradient Boosting": gbt},
        X_scaled, y
    )

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n--- Summary ---")
    summary = pd.DataFrame(results, index=["Accuracy", "ROC-AUC"]).T
    summary = summary.sort_values("ROC-AUC", ascending=False)
    print(summary.to_string())

    best = summary["ROC-AUC"].idxmax()
    print(f"\nBest model by ROC-AUC: {best} ({summary.loc[best, 'ROC-AUC']:.4f})")
    print("\nDay 3 complete — moving to hyperparameter tuning tomorrow.")


if __name__ == "__main__":
    main()
