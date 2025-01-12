import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Example: Replace 'lsi_vectors' with your actual LSI-reduced vectors
# lsi_vectors = np.array([...])  # Load or generate your LSI vectors
# For demonstration, let's create some random LSI data:
np.random.seed(42)
lsi_vectors = np.random.rand(10, 5)  # 10 speeches, 5 LSI dimensions

# Step 1: Compute the cosine distance matrix
distance_matrix = cosine_distances(lsi_vectors)

# Step 2: Perform Agglomerative Clustering
agg_clustering = AgglomerativeClustering(
    n_clusters=3,  # Set the number of clusters
    metric='precomputed',  # Use the precomputed cosine distance matrix
    linkage='average'  # Choose linkage type: 'average', 'complete', or 'single'
)
cluster_labels = agg_clustering.fit_predict(distance_matrix)

# Step 3: Visualize the clustering
print("Cluster Labels:", cluster_labels)

# Create a dendrogram for better understanding
linkage_matrix = linkage(distance_matrix, method='average')  # 'average' matches AgglomerativeClustering
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix)
plt.title("Dendrogram (Agglomerative Clustering with Cosine Distance)")
plt.xlabel("Speech Index")
plt.ylabel("Distance")
plt.show()
