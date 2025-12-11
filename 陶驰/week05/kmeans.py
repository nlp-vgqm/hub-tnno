import torch
import numpy as np
from sklearn.cluster import KMeans

X = np.random.rand(100, 2)
X_tensor = torch.tensor(X, dtype=torch.float32)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

labels = kmeans.labels_

distances = []
for i in range(kmeans.n_clusters):
    cluster_points = X_tensor[labels == i]
    dist = torch.norm(cluster_points - centers[i], dim=1)
    avg_distance = torch.mean(dist)
    distances.append((i, avg_distance))

distances_sorted = sorted(distances, key=lambda x: x[1])

for cluster_id, avg_distance in distances_sorted:
    print(f"Cluster {cluster_id} - Average Intra-cluster Distance: {avg_distance:.4f}")
