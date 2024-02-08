import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

from kmeans import k_means_clustering
from spectral import spectral_clustering
from metrics import clustering_score

def construct_affinity_matrix(data, affinity_type, *, k=3, sigma=1.0):
    """
    Construct the affinity matrix for spectral clustering based on the given data.

    Parameters:
    - data: numpy array, mxn representing m points in an n-dimensional dataset.
    - affinity_type: str, type of affinity matrix to construct. Options: 'knn' or 'rbf'.
    - k: int, the number of nearest neighbors for the KNN affinity matrix (default: 3).
    - sigma: float, bandwidth parameter for the RBF kernel (default: 1.0).

    Returns:
    - affinity_matrix: numpy array, the constructed affinity matrix based on the specified type.
    """

    n = data.shape[0]
    affinity_matrix = np.zeros((n, n))
    pairwise_distances = np.einsum('ij,ij->i',data,data)[:,None] + np.einsum('ij,ij->i',data,data) - 2*np.dot(data,data.T)

    if affinity_type == 'knn':
        k_indices = np.argpartition(pairwise_distances, k+1, axis=1)[:, :k+1]
        affinity_matrix = np.zeros_like(pairwise_distances)
        k_indices_array = np.array(k_indices)
        affinity_matrix[k_indices_array, np.arange(n)[:, np.newaxis]] = 1
        affinity_matrix[np.arange(n)[:, np.newaxis], k_indices_array] = 1
        return affinity_matrix

    elif affinity_type == 'rbf':
        affinity_matrix = np.exp(-((pairwise_distances**2) / (2 * sigma ** 2)))
        return affinity_matrix

    else:
        raise Exception("invalid affinity matrix type")

def plot_scatter(X, y, title, i, j, ds_name):
    plots[i][j].scatter(X[:,0],X[:,1],c=y,marker='o',cmap="copper")
    plots[i][j].set_title("kmeans clusteration for %s" %ds_name)

if __name__ == "__main__":
    datasets = ['blobs', 'circles', 'moons']
    fig, plots = plt.subplots(3,4,figsize=(14,10))
    for i, ds_name in enumerate(datasets):
        dataset = np.load("datasets/%s.npz" % ds_name)
        X = dataset['data']     # feature points
        y = dataset['target']   # ground truth labels
        
        n = len(np.unique(y)) # number of clusters
        k = 10
        sigma = .05

        y_km, _ = k_means_clustering(X, n)

        Arbf = construct_affinity_matrix(X, 'rbf', sigma=sigma)
        y_rbf = spectral_clustering(Arbf, n)

        Aknn = construct_affinity_matrix(X, 'knn', k=k)
        y_knn = spectral_clustering(Aknn, n)

        print("K-means on %s:" % ds_name, clustering_score(y, y_km))
        print("RBF affinity on %s:" % ds_name, clustering_score(y, y_rbf))
        print("KNN affinity on %s:" % ds_name, clustering_score(y, y_knn))

        plot_scatter(X, y, "ground truth for %s", i, 0, ds_name)
        plot_scatter(X, y_km, "kmeans clusteration for %s", i, 1, ds_name)
        plot_scatter(X, y_rbf, "rbf clusteration for %s", i, 2, ds_name)
        plot_scatter(X, y_knn, "knn clusteration for %s", i, 3, ds_name)
        
    plt.show()

