# %% markdown
# ## Titanic - Day 2: Feature Engineering & Preprocessing
# Building on yesterday's EDA to prepare features for modeling.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# load titanic dataset
df = sns.load_dataset('titanic')

# %% markdown
# ### Handle Missing Values

# age: fill with median grouped by class and sex (more accurate than overall median)
df['age'] = df.groupby(['pclass', 'sex'])['age'].transform(
    lambda x: x.fillna(x.median())
)

# embarked: fill with mode
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# deck has too many missing values — drop it
df.drop(columns=['deck'], inplace=True)

print("Missing values after imputation:")
print(df.isnull().sum())

# %% markdown
# ### Feature Engineering

# family size - combining sibsp and parch
df['family_size'] = df['sibsp'] + df['parch'] + 1  # +1 for self

# is alone?
df['is_alone'] = (df['family_size'] == 1).astype(int)

# age buckets
df['age_group'] = pd.cut(df['age'],
                          bins=[0, 12, 18, 35, 60, 100],
                          labels=['child', 'teen', 'young_adult', 'adult', 'senior'])

# fare per person (some tickets were shared)
df['fare_per_person'] = df['fare'] / df['family_size']

# title from name
df['title'] = df['who'].map({'man': 'Mr', 'woman': 'Mrs', 'child': 'Master'})

print("\nNew features:")
print(df[['family_size', 'is_alone', 'age_group', 'fare_per_person', 'title']].head(10))

# %% markdown
# ### Encode Categorical Variables

# select features for modeling
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
            'embarked', 'family_size', 'is_alone', 'fare_per_person']

target = 'survived'

df_model = df[features + [target]].copy()

# encode sex and embarked
le = LabelEncoder()
df_model['sex'] = le.fit_transform(df_model['sex'])
df_model['embarked'] = le.fit_transform(df_model['embarked'].astype(str))

# drop any remaining nulls
df_model.dropna(inplace=True)

print(f"\nFinal dataset shape: {df_model.shape}")
print(df_model.head())

# %% markdown
# ### Feature Correlations with Target

plt.figure(figsize=(10, 8))
corr = df_model.corr()
sns.heatmap(corr[['survived']].sort_values('survived', ascending=False),
            annot=True, fmt='.2f', cmap='RdYlGn', center=0)
plt.title('Feature Correlation with Survival')
plt.tight_layout()
plt.savefig('feature_correlation.png', dpi=150)
plt.close()
print("Saved feature_correlation.png")

# %% markdown
# ### Train/Test Split + Scaling

X = df_model.drop(columns=[target])
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"Survival rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")

# save processed data for next step
np.save('X_train.npy', X_train_scaled)
np.save('X_test.npy', X_test_scaled)
np.save('y_train.npy', y_train.values)
np.save('y_test.npy', y_test.values)

print("\nPreprocessed data saved. Ready for modeling!")
# TODO: try adding interaction features (pclass * fare) in future iterations
