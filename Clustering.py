import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Enables 3D plotting
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Wheat (Kama, Rosa, Canadian) by kernel
# https://www.kaggle.com/datasets/dongeorge/seed-from-uci
df = pd.read_csv("Seed_Data.csv")
df["target"] = df["target"].replace({0: "Kama", 1: "Rosa", 2: "Canadian"})

# Compactness Kernel Length, Kernel Width
X = df.iloc[:, [5, 3, 4]].to_numpy()
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # Standardize

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=kmeans.labels_, s=20, cmap='viridis')
ax.scatter(kmeans.cluster_centers_[:, 0],
           kmeans.cluster_centers_[:, 1],
           kmeans.cluster_centers_[:, 2],
           c='red', s=50, marker='X', label='Centers')
ax.set_title("KMeans 3D")
ax.set_xlabel("Asymmetry Coefficient")
ax.set_ylabel("Kernel Length")
ax.set_zlabel("Kernel Width")
plt.legend()
plt.show()

label_map = {"Kama": 0, "Rosa": 1, "Canadian": 2}
true_labels_numeric = df["target"].map(label_map).to_numpy()

# evaluate ari score
# 0.54 average with 2D
ari_score = adjusted_rand_score(true_labels_numeric, kmeans.labels_)
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
