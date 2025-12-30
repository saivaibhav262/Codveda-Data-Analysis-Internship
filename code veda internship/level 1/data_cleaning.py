# Task 1: Data Cleaning and Preprocessing
# Tools: Python, pandas

import pandas as pd
import numpy as np

# STEP 1: Load the dataset
input_file = "iris.csv"
output_file = "iris_cleaned_dataset.csv"

print("Loading dataset...")
df = pd.read_csv(input_file)

# STEP 2: Initial inspection
print("\nInitial Dataset Info:")
print(df.info())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape (Rows, Columns):", df.shape)

# STEP 3: Identify missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# STEP 4: Handle missing values
# Numerical columns -> fill with median
numerical_cols = df.select_dtypes(include=[np.number]).columns

for col in numerical_cols:
    median_value = df[col].median()
    df[col].fillna(median_value, inplace=True)
    print(f"Filled missing values in numerical column '{col}' with median")

# Categorical columns -> fill with mode
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    mode_value = df[col].mode()[0]
    df[col].fillna(mode_value, inplace=True)
    print(f"Filled missing values in categorical column '{col}' with mode")

# STEP 5: Remove duplicate rows
duplicates_before = df.duplicated().sum()
df.drop_duplicates(inplace=True)
duplicates_after = df.duplicated().sum()

print(f"\nDuplicates before removal: {duplicates_before}")
print(f"Duplicates after removal: {duplicates_after}")

# STEP 6: Standardize categorical data formats
for col in categorical_cols:
    df[col] = df[col].str.strip().str.lower()

print("\nStandardized categorical columns (lowercase & trimmed)")

# STEP 7: Convert date columns (if any column contains 'date')
for col in df.columns:
    if 'date' in col.lower():
        df[col] = pd.to_datetime(df[col], errors='coerce')
        print(f"Converted column '{col}' to datetime format")

# STEP 8: Final validation
print("\nFinal Dataset Info:")
print(df.info())

print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# STEP 9: Save cleaned dataset
df.to_csv(output_file, index=False)
print(f"\nCleaned dataset saved as '{output_file}'")

print("\nData Cleaning and Preprocessing Completed Successfully!")


