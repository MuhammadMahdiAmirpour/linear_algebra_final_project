from kmeans import k_means_clustering
from numpy import linalg as LA
import numpy as np

def laplacian(A):
    """
    Calculate the Laplacian matrix of the affinity matrix A using the symmetric normalized Laplacian formulation.

    Parameters:
    - A: numpy array, affinity matrix capturing pairwise relationships between data points.

    Returns:
    - L_sym: numpy array, symmetric normalized Laplacian matrix.
    """

    D = A.sum(axis=1)
    D_inv_sqrt = np.sqrt(LA.inv(np.diag(D)))
    return np.eye(A.shape[0]) - (D_inv_sqrt @ A @ D_inv_sqrt)


def spectral_clustering(affinity, k):
    """
    Perform spectral clustering on the given affinity matrix.

    Parameters:
    - affinity: numpy array, affinity matrix capturing pairwise relationships between data points.
    - k: int, number of clusters.

    Returns:
    - labels: numpy array, cluster labels assigned by the spectral clustering algorithm.
    """

    L = laplacian(affinity)
    eigvals, eigvecs = LA.eig(L)
    eigvecs = eigvecs[:, np.argsort(eigvals)[:k]]
    labels = k_means_clustering(eigvecs, k)[0]
    return labels

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import sklearn.cluster as skl_cluster
    import sklearn.datasets as skl_data

    np.random.seed(1)
    random_points = np.random.randint(0, 100, (100, 15))
    # data = make_blobs(n_samples=100, n_features=2)
    # random_points = data[0]
    A = np.array([[2, 3, 4],
                  [3, 4, 6],
                  [4, 6, 8]])
    # print(laplacian(A))
    # pairwise_distances = np.linalg.norm(random_points[:, None] - random_points, axis=2)
    m, n = random_points.shape
    pairwise_distances = np.linalg.norm(np.tile(random_points.reshape((m,1,n)),(1,m,1)) - random_points,axis=2)
    affinity_matrix = np.exp(-np.square(pairwise_distances)/(2*np.square(1.00)))
    labels = spectral_clustering(affinity_matrix, 5)
    print(labels)
    model = skl_cluster.SpectralClustering(n_clusters=5, affinity='rbf')
    labels = model.fit_predict(affinity_matrix)
    print(labels)
    # pairwise_distances = np.linalg.norm(random_points[:, None] - random_points, axis=2)
    # pairwise_distances = np.linalg.norm(A[:, None] - A, axis=2)
    # print(pairwise_distances)
    # print(laplacian(pairwise_distances))
    # print(spectral_clustering(pairwise_distances, k=3))

