# Titanic Survival Prediction - Day 4: Hyperparameter Tuning & Final Evaluation
# Grid-search best params for Random Forest and Gradient Boosting.
# Final held-out test evaluation and summary of the project.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)


# ------------------------------------------------------------------
# 1. Data loading (same pipeline as previous days)
# ------------------------------------------------------------------

def load_data():
    try:
        df = pd.read_csv("titanic_processed.csv")
        return df
    except FileNotFoundError:
        pass

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


df = load_data()
feature_cols = [c for c in df.columns if c != "survived"]
X = df[feature_cols].values
y = df["survived"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# ------------------------------------------------------------------
# 2. Hyperparameter tuning with GridSearchCV
# ------------------------------------------------------------------

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# -- Random Forest --
rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [2, 4, 6],
    "max_features": ["sqrt", "log2"],
}
rf_gs = GridSearchCV(
    RandomForestClassifier(random_state=SEED, n_jobs=-1),
    rf_param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0
)
rf_gs.fit(X_train, y_train)
print(f"\nRandom Forest best params : {rf_gs.best_params_}")
print(f"Random Forest best CV AUC : {rf_gs.best_score_:.4f}")

# -- Gradient Boosting --
gbt_param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 4, 5],
    "subsample": [0.8, 1.0],
}
gbt_gs = GridSearchCV(
    GradientBoostingClassifier(random_state=SEED),
    gbt_param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0
)
gbt_gs.fit(X_train, y_train)
print(f"\nGradient Boosting best params : {gbt_gs.best_params_}")
print(f"Gradient Boosting best CV AUC : {gbt_gs.best_score_:.4f}")

# -- Logistic Regression (C sweep) --
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=SEED))
])
lr_param_grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
lr_gs = GridSearchCV(lr_pipe, lr_param_grid, cv=cv,
                     scoring="roc_auc", n_jobs=-1, verbose=0)
lr_gs.fit(X_train, y_train)
print(f"\nLogistic Regression best params : {lr_gs.best_params_}")
print(f"Logistic Regression best CV AUC : {lr_gs.best_score_:.4f}")

# ------------------------------------------------------------------
# 3. Final test evaluation
# ------------------------------------------------------------------

tuned_models = {
    "Logistic Regression (tuned)": lr_gs.best_estimator_,
    "Random Forest (tuned)": rf_gs.best_estimator_,
    "Gradient Boosting (tuned)": gbt_gs.best_estimator_,
}

print("\n" + "=" * 55)
print("FINAL TEST SET RESULTS")
print("=" * 55)

test_records = {}
for name, model in tuned_models.items():
    if "Logistic" in name:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    test_records[name] = {"accuracy": acc, "roc_auc": auc,
                           "avg_precision": ap, "y_pred": y_pred, "y_prob": y_prob}
    print(f"{name}\n  Accuracy={acc:.4f}  ROC-AUC={auc:.4f}  Avg-Precision={ap:.4f}\n")

best_name = max(test_records, key=lambda k: test_records[k]["roc_auc"])
print(f"Best overall: {best_name}")
print("\nClassification Report:")
print(classification_report(y_test, test_records[best_name]["y_pred"],
                             target_names=["Died", "Survived"]))

# ------------------------------------------------------------------
# 4. Visualizations
# ------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ROC curves
for name, res in test_records.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    axes[0].plot(fpr, tpr, label=f"{name.split('(')[0].strip()} AUC={res['roc_auc']:.3f}")
axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curves (Tuned Models)")
axes[0].legend(fontsize=8)

# Precision-Recall curves
for name, res in test_records.items():
    prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
    axes[1].plot(rec, prec, label=f"{name.split('(')[0].strip()} AP={res['avg_precision']:.3f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curves (Tuned Models)")
axes[1].legend(fontsize=8)

# Confusion matrix for best model
cm = confusion_matrix(y_test, test_records[best_name]["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Died", "Survived"],
            yticklabels=["Died", "Survived"], ax=axes[2])
axes[2].set_title(f"Confusion Matrix\n{best_name.split('(')[0].strip()}")
axes[2].set_ylabel("Actual")
axes[2].set_xlabel("Predicted")

plt.tight_layout()
plt.savefig("final_evaluation.png", dpi=120, bbox_inches="tight")
plt.show()
print("\nSaved final_evaluation.png")

# ------------------------------------------------------------------
# 5. Feature importances for best tree model
# ------------------------------------------------------------------

best_tree = rf_gs.best_estimator_
importances = pd.Series(best_tree.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=True).tail(10)

plt.figure(figsize=(8, 5))
importances.plot(kind="barh", color="steelblue")
plt.title("Top 10 Feature Importances (Tuned Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importances_tuned.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved feature_importances_tuned.png")

print("\nProject complete — see README.md for full summary.")
