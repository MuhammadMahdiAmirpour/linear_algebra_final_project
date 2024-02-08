import numpy as np

def k_means_clustering(data, k, max_iterations=100):
    """
    Perform K-means clustering on the given dataset.

    Parameters:
    - data: numpy array, mxn representing m points in an n-dimensional dataset.
    - k: int, the number of resulting clusters.
    - max_iterations: int, optional parameter to prevent potential infinite loops (default: 100).

    Returns:
    - labels: numpy array, cluster labels for each data point.
    - centroids: numpy array, final centroids of the clusters.
    """
    m, n = data.shape
    centroids = data[np.random.choice(m, k, replace=False)]
    for _ in range(max_iterations):
        labels = np.argmin(np.linalg.norm(centroids-data[:,np.newaxis], axis = 2), axis=1)
        old_centroids = centroids
        centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        if np.all(old_centroids - centroids < 0.000001):
            break
    return labels, centroids

if __name__ == "__main__":
    from sklearn.datasets import load_iris

    iris = load_iris()
    print(iris.data.shape)  # Output: (150, 4)
    print(iris.feature_names)  # Output: ['sepal length (cm)', ...]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(iris.data)
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    # Calculate the within-cluster sum of square across different cluster counts
    inertia = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)# Plot the elbow graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within-cluster Sum of Square')
    plt.show()
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    from sklearn.metrics import confusion_matrix, classification_report
    print(confusion_matrix(iris.target, clusters))
    print(classification_report(iris.target, clusters))
    # Choose two dimensions to plot (e.g., sepal length and width)
    plt.figure(figsize=(10, 6))
    plt.scatter(scaled_features[:,0], scaled_features[:,1], c=clusters, cmap='viridis', marker='o')
    plt.title('Visualization of clustered data', fontsize=14)
    plt.xlabel('Scaled Sepal Length', fontsize=12)
    plt.ylabel('Scaled Sepal Width', fontsize=12)
    plt.show()

    labels, centroids = k_means_clustering(iris.data, 3)
    plt.scatter(iris.data[:, 0], iris.data[:, 1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], c=range(len(centroids)), marker="*", s=200)
    plt.show()

