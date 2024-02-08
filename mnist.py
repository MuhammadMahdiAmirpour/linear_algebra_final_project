import numpy as np
from copy import deepcopy
import sys
from numpy import linalg as LA
import matplotlib.pyplot as plt

from spectral import spectral_clustering
from metrics import clustering_score

from joblib import Parallel, delayed
from itertools import combinations


def chamfer_distance(point_cloud1, point_cloud2):
    # pairwise_distances1 = np.einsum('ij,ij->i',data,data)[:,None] + np.einsum('ij,ij->i',data,data) - 2*np.dot(data,data.T)
    pairwise_distances1 = np.sqrt(np.sum((point_cloud1[:, None] - point_cloud2)**2, axis=-1))
    nearest_neighbor_indices1 = np.argmin(pairwise_distances1, axis=1)  
    nearest_neighbor_distances1 = np.min(pairwise_distances1, axis=1) 
    pairwise_distances2 = np.sqrt(np.sum((point_cloud2[:, None] - point_cloud1)**2, axis=-1))
    nearest_neighbor_indices2 = np.argmin(pairwise_distances2, axis=1)  
    nearest_neighbor_distances2 = np.min(pairwise_distances2, axis=1)  
    chamfer_distance = np.mean(nearest_neighbor_distances1) + np.mean(nearest_neighbor_distances2)
    return chamfer_distance

def rigid_transform(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    centered_A = A - centroid_A
    centered_B = B - centroid_B
    H = centered_A.T@centered_B
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return R, t


def icp(source, target, max_iterations=100, tolerance=1e-5):
    prev_cham_dist = sys.maxsize
    for iteration in range(max_iterations):
        # Find the nearest neighbors of target in the source
        distances = np.linalg.norm(source[:, np.newaxis, :] - target, axis=2)
        nearest_neighbors = np.argmin(distances, axis=1)
        # Extract corresponding points from the source and target
        corresponding_target = target[nearest_neighbors]
        # Calculate rigid transformation
        R, t = rigid_transform(source, corresponding_target)
        # Apply transformation to source points
        # transformed_source = np.dot(transformed_source, R.T) + t
        # R must be multiplied by vectors of source one by one -> R @ source.T -> columnwise vectors as result
        # t is a row vector -> answer = (R @ source.T).T + t => source @ R.T + t
        source =  source @ R.T + t
        # Calculate Chamfer distance
        chamfer_dist = chamfer_distance(source, target)
        # Check for convergence
        if iteration > 0 and np.abs(chamfer_dist - prev_chamfer_dist) < tolerance:
            break
        prev_chamfer_dist = chamfer_dist
    return source

def construct_affinity_matrix(point_clouds):
    """
    Construct the affinity matrix for spectral clustering based on the given data.

    Parameters:
    - point_clouds: numpy array, mxnxd representing m point clouds each containing n points in a d-dimensional space.

    Returns:
    - affinity_matrix: numpy array, the constructed affinity matrix using Chamfer distance.
    """
    m, _, _ = point_clouds.shape
    pairwise_chamfer = np.zeros((m, m))
    def compute_affinity(i, j):
        return chamfer_distance(icp(point_clouds[i], point_clouds[j]), point_clouds[j])
    # Parallelize computation of affinity matrix
    results = Parallel(n_jobs=-1)(delayed(compute_affinity)(i, j) for i, j in combinations(range(m), 2))
    # Fill the affinity matrix
    for idx, (i, j) in enumerate(combinations(range(m), 2)):
        pairwise_chamfer[i, j] = results[idx]
    pairwise_chamfer += pairwise_chamfer.T
#     sigma = .8501
#     affinity_matrix = np.exp(-((pairwise_chamfer**2) / (2 * sigma ** 2)))
#     return affinity_matrix
    k = 3
    k_indices = np.argpartition(pairwise_chamfer, k+1, axis=1)[:, :k+1]
    affinity_matrix = np.zeros_like(pairwise_chamfer)
    k_indices_array = np.array(k_indices)
    affinity_matrix[k_indices_array, np.arange(m)[:, np.newaxis]] = 1
    affinity_matrix[np.arange(m)[:, np.newaxis], k_indices_array] = 1
    return affinity_matrix

#     for i in range(num_clouds):
#         for j in range(i+1, num_clouds):
#             cloud_i = point_clouds[i]
#             cloud_j = point_clouds[j]
#             # Register the point clouds with each other using ICP
#             registered_cloud_i = icp(cloud_i, cloud_j)
#             # Calculate symmetric Chamfer distance between registered clouds
#             chamfer_dist_ij = chamfer_distance(registered_cloud_i, cloud_j)
#             chamfer_dist_ji = chamfer_distance(cloud_j, registered_cloud_i)
#             # Set affinity values in the matrix
#             affinity_matrix[i, j] = chamfer_dist_ij
#             affinity_matrix[j, i] = chamfer_dist_ji
#     return affinity_matrix

"""
I used the reference below for plotting these things
https://stackoverflow.com/questions/62433465/how-to-plot-3d-point-clouds-from-an-npy-file
"""
def plot_stuff(eigenvectors, y_pred, Ach):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(eigenvectors[:, 0], eigenvectors[:, 1], eigenvectors[:, 2], c=y_pred, marker='o', cmap='copper')
#     for i in range(3):
#         ax = fig.add_subplot(2, 3, i + 1, projection='3d')
#         ax.scatter(eigenvectors[:, i], eigenvectors[:, i + 1], eigenvectors[:, i + 2], c=y_pred, cmap='copper')
#         ax.set_title(f'Eigenvectors {i + 1} - {i + 3}')
    ax = fig.add_subplot(2, 3, 4)
    im = ax.imshow(Ach, cmap='copper')
    ax.set_title('Affinity Matrix')
    fig.colorbar(im)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset = "mnist"

    dataset = np.load("datasets/%s.npz" % dataset)
    X = dataset['data']  # feature points
    y = dataset['target']  # ground truth labels
    n = len(np.unique(y))  # number of clusters

    Ach = construct_affinity_matrix(X)
    y_pred = spectral_clustering(Ach, n)

    print("Chamfer affinity on %s:" % dataset, clustering_score(y, y_pred))

    # Plot Ach using its first 3 eigenvectors
    _, eigvecs = np.linalg.eigh(Ach)
    plot_stuff(eigvecs, y_pred, Ach)

