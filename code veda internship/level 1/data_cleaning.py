# Task 1: Data Cleaning and Preprocessing
# Tools: Python, pandas

import pandas as pd
import numpy as np

input_file = "iris.csv"
output_file = "iris_cleaned_dataset.csv"

print("Loading dataset...")
df = pd.read_csv(input_file)

print("\nInitial Dataset Info:")
print(df.info())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape (Rows, Columns):", df.shape)

print("\nMissing values in each column:")
print(df.isnull().sum())

numerical_cols = df.select_dtypes(include=[np.number]).columns

for col in numerical_cols:
    median_value = df[col].median()
    df[col].fillna(median_value, inplace=True)
    print(f"Filled missing values in numerical column '{col}' with median")

categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    mode_value = df[col].mode()[0]
    df[col].fillna(mode_value, inplace=True)
    print(f"Filled missing values in categorical column '{col}' with mode")

duplicates_before = df.duplicated().sum()
df.drop_duplicates(inplace=True)
duplicates_after = df.duplicated().sum()

print(f"\nDuplicates before removal: {duplicates_before}")
print(f"Duplicates after removal: {duplicates_after}")

for col in categorical_cols:
    df[col] = df[col].str.strip().str.lower()

print("\nStandardized categorical columns (lowercase & trimmed)")

for col in df.columns:
    if 'date' in col.lower():
        df[col] = pd.to_datetime(df[col], errors='coerce')
        print(f"Converted column '{col}' to datetime format")

print("\nFinal Dataset Info:")
print(df.info())

print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

df.to_csv(output_file, index=False)
print(f"\nCleaned dataset saved as '{output_file}'")

print("\nData Cleaning and Preprocessing Completed Successfully!")

