# %% markdown
# # Titanic Survival Prediction - Day 1: Exploratory Data Analysis
# Getting familiar with the data before jumping into modeling.
# Classic Kaggle dataset - predicting who survived the Titanic disaster.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# %%
# Load the Titanic dataset from OpenML (same data as Kaggle)
print("Loading Titanic dataset...")
titanic = fetch_openml("titanic", version=1, as_frame=True)
df = titanic.frame

print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")

# %%
# Quick overview
print("\n--- First few rows ---")
print(df.head())

print("\n--- Data types ---")
print(df.dtypes)

print("\n--- Missing values ---")
missing = df.isnull().sum()
print(missing[missing > 0])

# %%
# Let's clean up the target variable
# survived: 0 = no, 1 = yes
df["survived"] = df["survived"].astype(int)

print(f"\nSurvival rate: {df['survived'].mean():.2%}")
print(f"Total passengers: {len(df)}")
print(f"Survived: {df['survived'].sum()}")
print(f"Did not survive: {(df['survived'] == 0).sum()}")

# %%
# --- VISUALIZATION 1: Overall survival ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Count plot
sns.countplot(data=df, x="survived", ax=axes[0], palette=["#e74c3c", "#2ecc71"])
axes[0].set_title("Survival Count", fontsize=14)
axes[0].set_xticklabels(["Did Not Survive", "Survived"])
axes[0].set_xlabel("")

# Pie chart
survival_counts = df["survived"].value_counts()
axes[1].pie(
    survival_counts,
    labels=["Did Not Survive", "Survived"],
    autopct="%1.1f%%",
    colors=["#e74c3c", "#2ecc71"],
    startangle=90,
)
axes[1].set_title("Survival Distribution", fontsize=14)

plt.tight_layout()
plt.savefig("survival_overview.png", dpi=100, bbox_inches="tight")
plt.show()
print("Saved: survival_overview.png")

# %%
# --- VISUALIZATION 2: Survival by Pclass ---
# pclass is socioeconomic status: 1 = upper, 2 = middle, 3 = lower
df["pclass"] = df["pclass"].astype(int)

pclass_survival = df.groupby("pclass")["survived"].agg(["mean", "count"]).reset_index()
pclass_survival.columns = ["pclass", "survival_rate", "count"]
print("\nSurvival rate by passenger class:")
print(pclass_survival)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(data=df, x="pclass", y="survived", ax=axes[0], palette="Blues_d")
axes[0].set_title("Survival Rate by Passenger Class", fontsize=14)
axes[0].set_xlabel("Passenger Class (1=Upper, 3=Lower)")
axes[0].set_ylabel("Survival Rate")
axes[0].set_ylim(0, 1)

sns.countplot(data=df, x="pclass", hue="survived", ax=axes[1], palette=["#e74c3c", "#2ecc71"])
axes[1].set_title("Survival Count by Passenger Class", fontsize=14)
axes[1].set_xlabel("Passenger Class")
axes[1].legend(["Did Not Survive", "Survived"])

plt.tight_layout()
plt.savefig("survival_by_class.png", dpi=100, bbox_inches="tight")
plt.show()
print("Saved: survival_by_class.png")

# %%
# --- VISUALIZATION 3: Survival by Sex ---
sex_survival = df.groupby("sex")["survived"].mean()
print("\nSurvival rate by sex:")
print(sex_survival)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(data=df, x="sex", y="survived", ax=axes[0], palette=["#3498db", "#e91e63"])
axes[0].set_title("Survival Rate by Sex", fontsize=14)
axes[0].set_ylabel("Survival Rate")
axes[0].set_ylim(0, 1)

sns.countplot(data=df, x="sex", hue="survived", ax=axes[1], palette=["#e74c3c", "#2ecc71"])
axes[1].set_title("Survival Count by Sex", fontsize=14)
axes[1].legend(["Did Not Survive", "Survived"])

plt.tight_layout()
plt.savefig("survival_by_sex.png", dpi=100, bbox_inches="tight")
plt.show()
print("Saved: survival_by_sex.png")

# %%
# --- VISUALIZATION 4: Age distribution ---
df["age"] = pd.to_numeric(df["age"], errors="coerce")

print(f"\nAge stats:")
print(df["age"].describe())
print(f"Missing age values: {df['age'].isnull().sum()}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Age distribution by survival
for survived, color, label in [(0, "#e74c3c", "Did Not Survive"), (1, "#2ecc71", "Survived")]:
    subset = df[df["survived"] == survived]["age"].dropna()
    axes[0].hist(subset, bins=30, alpha=0.6, color=color, label=label)

axes[0].set_title("Age Distribution by Survival", fontsize=14)
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Count")
axes[0].legend()

# KDE plot
df[df["survived"] == 0]["age"].dropna().plot.kde(ax=axes[1], color="#e74c3c", label="Did Not Survive")
df[df["survived"] == 1]["age"].dropna().plot.kde(ax=axes[1], color="#2ecc71", label="Survived")
axes[1].set_title("Age Distribution (KDE)", fontsize=14)
axes[1].set_xlabel("Age")
axes[1].legend()

plt.tight_layout()
plt.savefig("age_distribution.png", dpi=100, bbox_inches="tight")
plt.show()
print("Saved: age_distribution.png")

# %%
# --- VISUALIZATION 5: Fare distribution ---
df["fare"] = pd.to_numeric(df["fare"], errors="coerce")

print(f"\nFare stats:")
print(df["fare"].describe())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Log-transform fare for better visualization (lots of outliers)
df["fare_log"] = np.log1p(df["fare"])

axes[0].hist(df["fare"], bins=50, color="#9b59b6", edgecolor="white", alpha=0.8)
axes[0].set_title("Fare Distribution (raw)", fontsize=14)
axes[0].set_xlabel("Fare (£)")
axes[0].set_ylabel("Count")

axes[1].hist(df["fare_log"], bins=50, color="#9b59b6", edgecolor="white", alpha=0.8)
axes[1].set_title("Fare Distribution (log scale)", fontsize=14)
axes[1].set_xlabel("log(1 + Fare)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("fare_distribution.png", dpi=100, bbox_inches="tight")
plt.show()
print("Saved: fare_distribution.png")

# %%
# --- VISUALIZATION 6: Heatmap of survival by class and sex ---
pivot = df.groupby(["pclass", "sex"])["survived"].mean().unstack()
print("\nSurvival rate by class and sex:")
print(pivot)

plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".2%", cmap="RdYlGn", vmin=0, vmax=1,
            linewidths=0.5, cbar_kws={"label": "Survival Rate"})
plt.title("Survival Rate by Class and Sex", fontsize=14)
plt.xlabel("Sex")
plt.ylabel("Passenger Class")
plt.tight_layout()
plt.savefig("survival_heatmap.png", dpi=100, bbox_inches="tight")
plt.show()
print("Saved: survival_heatmap.png")

# %%
# --- Summary stats to guide feature engineering ---
print("\n" + "="*50)
print("EDA SUMMARY")
print("="*50)
print(f"Total samples: {len(df)}")
print(f"Overall survival rate: {df['survived'].mean():.2%}")
print(f"\nKey insights:")
print("  - Women survived at ~74% vs men at ~19% (sex is a strong predictor)")
print("  - 1st class: ~63% survival, 2nd: ~47%, 3rd: ~24% (class matters a lot)")
print("  - Children had higher survival rates (women and children first)")
print(f"  - ~20% of age values are missing - will need imputation")
print(f"  - Fare has heavy right skew - log transform will help")
print("\nNext: feature engineering and preprocessing")

# Save the cleaned dataframe for next steps
df_save = df.drop(columns=["fare_log"])  # drop temp column
df_save.to_csv("titanic_raw.csv", index=False)
print("\nSaved raw data to titanic_raw.csv")
