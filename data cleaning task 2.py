"""""
# README

## 📌 Project: Data Cleaning & Feature Engineering on Titanic Dataset

### 🔹 Dataset:
Titanic dataset (available on Kaggle / seaborn). It contains passenger information such as age, gender, class, fare, survival status, etc.

### 🔹 Objective:
- Perform data cleaning (handling missing values, duplicates, inconsistent formats, data types, outliers)
- Perform feature engineering (new features, encoding categorical variables, scaling numerical features)
- Split cleaned dataset into train/test sets
- Save final cleaned dataset into CSV files

### 🔹 Deliverables:
1. `data_cleaning_feature_engineering.py` (this script)
2. `cleaned_titanic.csv` (final dataset)
3. `requirements.txt` (dependencies)
4. `README.md` (this description)

### 🔹 Libraries Used:
- pandas, numpy, matplotlib, seaborn, scikit-learn

"""

# ================================
# 📌 Import Required Libraries
# ================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ================================
# 📌 Load Dataset
# ================================
# Titanic dataset from seaborn (you can replace with your own CSV)
titanic = sns.load_dataset('titanic')

print("Original Dataset Shape:", titanic.shape)
print(titanic.head())

# ================================
# 📌 Data Cleaning
# ================================

# 1. Handle Missing Values
print("\nMissing Values Before Cleaning:\n", titanic.isnull().sum())

# Fill age with median
titanic['age'] = titanic['age'].fillna(titanic['age'].median())

# Fill embark_town with mode
titanic['embark_town'] = titanic['embark_town'].fillna(titanic['embark_town'].mode()[0])

# Drop columns with too many missing values (e.g., 'deck')
titanic = titanic.drop(columns=['deck'])

# Drop rows with missing 'embarked'
titanic = titanic.dropna(subset=['embarked'])

# 2. Remove Duplicates
print("\nDuplicates before:", titanic.duplicated().sum())
titanic = titanic.drop_duplicates()
print("Duplicates after:", titanic.duplicated().sum())

# 3. Fix inconsistent formats (example: lowercase for categorical)
titanic['sex'] = titanic['sex'].str.lower()
titanic['class'] = titanic['class'].str.lower()

# 4. Convert Data Types
# Convert 'adult_male' (bool) to int
titanic['adult_male'] = titanic['adult_male'].astype(int)

# 5. Handle Outliers using IQR (for 'fare')
Q1 = titanic['fare'].quantile(0.25)
Q3 = titanic['fare'].quantile(0.75)
IQR = Q3 - Q1

# Define upper & lower bounds
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Cap outliers
titanic['fare'] = np.where(titanic['fare'] > upper, upper,
                          np.where(titanic['fare'] < lower, lower, titanic['fare']))

# ================================
# 📌 Feature Engineering
# ================================

# 1. Create New Features
# Family Size
titanic['family_size'] = titanic['sibsp'] + titanic['parch'] + 1

# Age Group
titanic['age_group'] = pd.cut(titanic['age'], bins=[0,12,18,35,60,100], labels=['child','teen','young_adult','adult','senior'])

# 2. Encode Categorical Variables
# Label Encoding for binary categorical
le = LabelEncoder()
titanic['sex'] = le.fit_transform(titanic['sex'])

# One-Hot Encoding for multi-categorical
titanic = pd.get_dummies(titanic, columns=['class','embark_town','age_group'], drop_first=True)

# 3. Scale Numerical Features
scaler = StandardScaler()
titanic[['age','fare','family_size']] = scaler.fit_transform(titanic[['age','fare','family_size']])

# ================================
# 📌 Dataset Splitting
# ================================

X = titanic.drop('survived', axis=1)
y = titanic['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# ================================
# 📌 Save Final Cleaned Dataset
# ================================
final_data = pd.concat([X, y], axis=1)
final_data.to_csv("cleaned_titanic.csv", index=False)
print("\n✅ Cleaned dataset saved as 'cleaned_titanic.csv'")
