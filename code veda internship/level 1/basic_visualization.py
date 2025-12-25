# -------------------------------------------------
# Level 1 - Task 3: Basic Data Visualization
# Tools: Python, matplotlib, seaborn
# Dataset: cleaned_dataset.csv
# -------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folder to save plots
output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)

# Load dataset
print("Loading cleaned dataset...")
df = pd.read_csv("iris_cleaned_dataset.csv")

# -----------------------------------------------
# 1. BAR PLOT - Average Sepal Length per Species
# -----------------------------------------------
plt.figure(figsize=(8, 6))
sns.barplot(x="species", y="sepal_length", data=df)
plt.title("Average Sepal Length by Species")
plt.xlabel("Species")
plt.ylabel("Sepal Length")
plt.savefig(f"{output_folder}/barplot_sepal_length.png")
plt.show()

# -----------------------------------------------
# 2. LINE CHART - Sepal Length Trend (Sample Order)
# -----------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(df.index, df["sepal_length"], label="Sepal Length")
plt.title("Sepal Length Trend")
plt.xlabel("Index")
plt.ylabel("Sepal Length")
plt.legend()
plt.savefig(f"{output_folder}/linechart_sepal_length.png")
plt.show()

# -----------------------------------------------
# 3. SCATTER PLOT - Petal Length vs Petal Width
# -----------------------------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="petal_length",
    y="petal_width",
    hue="species",
    data=df
)
plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend(title="Species")
plt.savefig(f"{output_folder}/scatterplot_petal.png")
plt.show()

print("\nBasic Data Visualization Completed Successfully!")
print("Plots saved in the 'plots' folder.")
