# LEVEL 2 - TASK 3: K-MEANS CLUSTERING

print("🔥 K-MEANS CLUSTERING STARTED 🔥")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load dataset
df = pd.read_csv("iris_cleaned_dataset.csv")
print("Dataset loaded successfully!")

# Step 2: Select numerical features
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
print("Numerical features selected")

# Step 3: Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data standardized")

# Step 4: Elbow Method
wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.savefig("elbow_method.png")
plt.show()

print("✔ Elbow method graph saved")

# Step 5: Apply K-Means with optimal K = 3
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
print("K-Means clustering applied")

# Step 6: Visualize clusters (2D)
plt.figure()
plt.scatter(df['sepal_length'], df['petal_length'],
            c=df['Cluster'], cmap='viridis')
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("K-Means Clustering Visualization")
plt.savefig("kmeans_clusters.png")
plt.show()

print("✔ Cluster visualization saved")

print("\n🎉 K-MEANS CLUSTERING COMPLETED SUCCESSFULLY 🎉")
