# Level 1 - Task 2: Exploratory Data Analysis (EDA)
# Tools: Python, pandas, matplotlib, seaborn
# Dataset: cleaned_dataset.csv

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading cleaned dataset...")
df = pd.read_csv("iris_cleaned_dataset.csv")

print("\nDataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics (Mean, Std, Min, Max, etc.):")
print(df.describe())

print("\nMode of the dataset:")
print(df.mode())

print("\nGenerating histograms...")
df.hist(figsize=(10, 8))
plt.suptitle("Distribution of Numerical Features")
plt.tight_layout()
plt.show()

print("\nGenerating boxplots...")
plt.figure(figsize=(8, 6))
sns.boxplot(data=df.drop(columns=["species"]))
plt.title("Boxplot of Numerical Features")
plt.show()

print("\nGenerating scatter plot...")
sns.scatterplot(
    x="sepal_length",
    y="petal_length",
    hue="species",
    data=df
)
plt.title("Sepal Length vs Petal Length")
plt.show()

print("\nCalculating correlation matrix...")
correlation_matrix = df.drop(columns=["species"]).corr()
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

correlation_matrix.to_csv("correlation_matrix.csv")
print("\nCorrelation matrix saved as 'correlation_matrix.csv'")

print("\nExploratory Data Analysis Completed Successfully!")

