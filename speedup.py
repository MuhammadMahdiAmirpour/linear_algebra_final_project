from spectral import spectral_clustering as spectral_clustering_old
from mnist import construct_affinity_matrix as construct_affinity_matrix_old

from metrics import clustering_score

from numba import jit, njit, prange, vectorize, guvectorize, cuda

import numpy as np
from numpy import linalg as LA
import sys
from timeit import timeit

# TODO: Rewrite the k_means_clustering function

@njit(parallel=True)
def k_means_clustering_numba(data, k, max_iterations=100):
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
    data = np.asarray(data, dtype=np.float32)
    centroids = data[np.random.choice(m, k, replace=False)]
    for _ in range(max_iterations):
        # Calculate distances in parallel
        # distances = np.linalg.norm(centroids - data[:, np.newaxis], axis=2)
        distances = np.zeros((data.shape[0], centroids.shape[0]))
        for i in range(data.shape[0]):
            for j in range(centroids.shape[0]):
                distances[i, j] = np.linalg.norm(centroids[j] - data[i])
        # Find the index of the minimum distance in parallel
        # labels = np.argmin(distances, axis=1)
        labels = np.zeros(data.shape[0], dtype=int)
        for i in range(data.shape[0]):
            min_distance_index = 0
            min_distance = distances[i, 0]
            for j in range(1, distances.shape[1]):
                if distances[i, j] < min_distance:
                    min_distance = distances[i, j]
                    min_distance_index = j
            labels[i] = min_distance_index
        old_centroids = centroids
        # Calculate new centroids in parallel
        centroids = np.array([np.mean(data[labels == i], axis=0) for i in range(k)])
        if np.all(old_centroids - centroids < 0.000001):
            break
    return labels, centroids


# # Define the k-means assignment function, which calculates the distance between each point and the centroids, and assigns the point to the closest centroid
# @njit
# def kmeans_assignment(centroids, points):
#     num_centroids, dim = centroids.shape
#     num_points, _ = points.shape
#     centroids_tiled = np.tile(centroids, (num_points, 1))
#     points_tiled = np.tile(points, (num_centroids, 1)).T
#     distances = np.sum((centroids_tiled - points_tiled) ** 2, axis=2)
#     return np.argmin(distances, axis=0)
# 
# # Implement the k-means update function, which calculates new centroids based on the assigned clusters
# @njit
# def kmeans_update(points, assignments, num_clusters):
#     num_points, dim = points.shape
#     new_centroids = np.zeros((num_clusters, dim))
#     for cluster in range(num_clusters):
#         assigned_points = points[assignments == cluster]
#         if assigned_points.size > 0:
#             new_centroids[cluster] = np.mean(assigned_points, axis=0)
#     return new_centroids
# 
# # Implement the k-means algorithm by alternating between the assignment and update steps until convergence
# def kmeans(points, num_clusters, max_iterations=100, tolerance=1e-5):
#     centroids = points[np.random.choice(points.shape[0], num_clusters, replace=False)]
#     for _ in range(max_iterations):
#         old_centroids = centroids
#         assignments = kmeans_assignment(centroids, points)
#         centroids = kmeans_update(points, assignments, num_clusters)
#         if np.linalg.norm(old_centroids - centroids) < tolerance:
#             break
#     return centroids, assignments

# TODO: Rewrite the laplacian function
@njit(parallel=True)
def laplacian_numba(A):
    """
    Calculate the Laplacian matrix of the affinity matrix A using the symmetric normalized Laplacian formulation.

    Parameters:
    - A: numpy array, affinity matrix capturing pairwise relationships between data points.

    Returns:
    - L_sym: numpy array, symmetric normalized Laplacian matrix.
    """
    n = A.shape[0]
    D = np.sum(A, axis=1)
    D_inv_sqrt = 1.0 / np.sqrt(D)
    L_sym = np.eye(n) - D_inv_sqrt[:, np.newaxis] * A * D_inv_sqrt
    return L_sym

# TODO: Rewrite the spectral_clustering function
@njit(parallel=True)
def spectral_clustering_numba(affinity, k):
    """
    Perform spectral clustering on the given affinity matrix.

    Parameters:
    - affinity: numpy array, affinity matrix capturing pairwise relationships between data points.
    - k: int, number of clusters.

    Returns:
    - labels: numpy array, cluster labels assigned by the spectral clustering algorithm.
    """
    L = laplacian_numba(affinity)
    eigvals, eigvecs = np.linalg.eigh(L)
    eigvecs = eigvecs[:, np.argsort(eigvals)[:k]]
    labels = k_means_clustering_numba(eigvecs, k)
    return labels


# TODO: Rewrite the chamfer_distance function
@njit(parallel=True)
def chamfer_distance_numba(point_cloud1, point_cloud2):
    m1, n = point_cloud1.shape
    m2 = point_cloud2.shape[0]
    pairwise_distances1 = np.empty((m1, m2), dtype=np.float64)
    pairwise_distances2 = np.empty((m2, m1), dtype=np.float64)
    for i in prange(m1):
        for j in prange(m2):
            dist1 = 0.0
            dist2 = 0.0
            for k in range(n):
                diff1 = point_cloud1[i, k] - point_cloud2[j, k]
                diff2 = point_cloud2[j, k] - point_cloud1[i, k]
                dist1 += diff1 * diff1
                dist2 += diff2 * diff2
            pairwise_distances1[i, j] = np.sqrt(dist1)
            pairwise_distances2[j, i] = np.sqrt(dist2)
    # nearest_neighbor_indices1 = np.argmin(pairwise_distances1, axis=1)
#     nearest_neighbor_indices1 = np.zeros(pairwise_distances1.shape[0], dtype=np.float32)
#     for i in range(pairwise_distances1.shape[0]):
#         min_distance_index = 0
#         min_distance = pairwise_distances1[i, 0]
#         for j in range(1, pairwise_distances1.shape[1]):
#             if pairwise_distances1[i, j] < min_distance:
#                 min_distance = pairwise_distances1[i, j]
#                 min_distance_index = j
#         nearest_neighbor_indices1[i] = min_distance_index
    # nearest_neighbor_distances1 = np.min(pairwise_distances1, axis=1)
    nearest_neighbor_distances1 = np.zeros(pairwise_distances1.shape[0], dtype=np.float32)
    for i in range(pairwise_distances1.shape[0]):
        min_distance = pairwise_distances1[i, 0]
        for j in range(1, pairwise_distances1.shape[1]):
            if pairwise_distances1[i, j] < min_distance:
                min_distance = pairwise_distances1[i, j]
        nearest_neighbor_distances1[i] = min_distance
    # nearest_neighbor_indices2 = np.argmin(pairwise_distances2, axis=1)
#     nearest_neighbor_indices2 = np.zeros(pairwise_distances2.shape[0], dtype=np.float32)
#     for i in range(pairwise_distances2.shape[0]):
#         min_distance_index = 0
#         min_distance = pairwise_distances2[i, 0]
#         for j in range(1, pairwise_distances2.shape[1]):
#             if pairwise_distances2[i, j] < min_distance:
#                 min_distance = pairwise_distances2[i, j]
#                 min_distance_index = j
#         nearest_neighbor_indices2[i] = min_distance_index
    # nearest_neighbor_distances2 = np.min(pairwise_distances2, axis=1)
    nearest_neighbor_distances2 = np.zeros(pairwise_distances2.shape[0], dtype=np.float32)
    for i in range(pairwise_distances2.shape[0]):
        min_distance = pairwise_distances2[i, 0]
        for j in range(1, pairwise_distances2.shape[1]):
            if pairwise_distances2[i, j] < min_distance:
                min_distance = pairwise_distances2[i, j]
        nearest_neighbor_distances2[i] = min_distance
    # chamfer_distance = np.mean(nearest_neighbor_distances1) + np.mean(nearest_neighbor_distances2)
    chamfer_distance = 0.0
    for distance in nearest_neighbor_distances1:
        chamfer_distance += distance
    for distance in nearest_neighbor_distances2:
        chamfer_distance += distance
    chamfer_distance /= (len(nearest_neighbor_distances1) + len(nearest_neighbor_distances2))
    return np.float32(chamfer_distance)


# TODO: Rewrite the rigid_transform function
@njit(parallel=True)
def rigid_transform_numba(A, B):
    # centroid_A = np.mean(A, axis=0)
    centroid_A = np.zeros(A.shape[1])
    for i in range(A.shape[1]):
        sum_values = 0
        for j in range(A.shape[0]):
            sum_values += A[j, i]
        centroid_A[i] = sum_values / A.shape[0]
    # centroid_B = np.mean(B, axis=0)
    centroid_B = np.zeros(B.shape[1])
    for i in range(B.shape[1]):
        sum_values = 0
        for j in range(B.shape[0]):
            sum_values += B[j, i]
        centroid_B[i] = sum_values / B.shape[0]
    centered_A = A - centroid_A
    centered_B = B - centroid_B
    H = np.dot(centered_A.T, centered_B)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    t = centroid_B - np.dot(centroid_A, R)
    return np.asarray(R, dtype=np.float32), np.asarray(t, dtype=np.float32)

# TODO: Rewrite the icp function
@njit(parallel=True)
def icp_numba(source, target, max_iterations=100, tolerance=1e-5):
    m, n = source.shape
    prev_cham_dist = sys.maxsize
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    for iteration in range(max_iterations):
        # Find the nearest neighbors of target in the source
        # distances = np.linalg.norm(source[:, np.newaxis, :] - target, axis=-1)
        distances = np.zeros((m, m))
        # Calculate distances using for loops
        for i in range(m):
            for j in range(m):
                distance = 0
                for k in range(n):
                    distance += (source[i, k] - target[j, k]) ** 2
                distances[i, j] = np.sqrt(distance)
        nearest_neighbors = np.argmin(distances, axis=1)
        # Extract corresponding points from the source and target
        corresponding_target = target[nearest_neighbors]
        # Calculate rigid transformation
        R, t = rigid_transform_numba(source, corresponding_target)
        # R = np.asarray(R, dtype=np.float32)
        # t = np.asarray(t, dtype=np.float32)
        # Apply transformation to source points
        # transformed_source = np.dot(transformed_source, R.T) + t
        # R must be multiplied by vectors of source one by one -> R @ source.T -> columnwise vectors as result
        # t is a row vector -> answer = (R @ source.T).T + t => source @ R.T + t
        source =  source @ R.T + t
        # Calculate Chamfer distance
        chamfer_dist = chamfer_distance_numba(source, target)
        # Check for convergence
        if iteration > 0 and np.abs(chamfer_dist - prev_cham_dist) < tolerance:
            break
        prev_chamfer_dist = chamfer_dist
    return source

# TODO: Rewrite the icp function
# @njit(parallel=True)
# def icp_numba(source, target, max_iterations=100, tolerance=1e-5):
#     transformed_source = deepcopy(source)
#     prev_chamfer_dist = sys.maxsize
#     for iteration in range(max_iterations):
#         # Find the nearest neighbors of target in the source
#         distances = np.linalg.norm(target[:, np.newaxis, :] - transformed_source, axis=2)
#         nearest_neighbors = np.argmin(distances, axis=1)
#         # Extract corresponding points from the source and target
#         corresponding_source = transformed_source[nearest_neighbors]
#         corresponding_target = target
#         # Calculate rigid transformation
#         R, t = rigid_transform_numba(corresponding_source, corresponding_target)
#         # Apply transformation to source points
#         transformed_source = np.dot(transformed_source, R.T) + t
#         # Calculate Chamfer distance
#         chamfer_dist = chamfer_distance_numba(transformed_source, target)
#         # Check for convergence
#         if iteration > 0 and np.abs(chamfer_dist - prev_chamfer_dist) < tolerance:
#             break
#         prev_chamfer_dist = chamfer_dist
#     return transformed_source

# TODO: Rewrite the construct_affinity_matrix function
@njit(parallel=True)
def construct_affinity_matrix_numba(point_clouds):
    num_clouds = len(point_clouds)
    affinity_matrix = np.zeros((num_clouds, num_clouds))
    for i in prange(num_clouds):
        pc1 = point_clouds[i]
        for j in prange(i + 1, num_clouds):
            pc2 = point_clouds[j]
            if np.all(pc1 == pc2):
                affinity_matrix[i, j] = 0
                affinity_matrix[j, i] = 0
            else:
                reg_cloud = icp_numba(pc1, pc2)
                cham_dist = chamfer_distance_numba(pc1, pc2)
                affinity_matrix[i, j] = cham_dist
                affinity_matrix[j, i] = cham_dist
    return affinity_matrix


if __name__ == "__main__":
    dataset = "mnist"
    dataset = np.load("datasets/%s.npz" % dataset)
    X = dataset['data']     # feature points
    y = dataset['target']   # ground truth labels
    n = len(np.unique(y))   # number of clusters
    # TODO: Run both the old and speed up version of your algorithms and capture running time
    # TODO: Compare the running time using timeit module
    Ach = construct_affinity_matrix_numba(X)
    _, eigvecs = np.linalg.eigh(Ach)
    y_pred = spectral_clustering_numba(Ach, n)
    plot_stuff(eigvecs, y_pred, Ach)
    print("Old Chamfer affinity on %s:" % dataset, clustering_score(y, y_pred))
    print("Chamfer affinity on %s:" % dataset, clustering_score(y, y_pred))
    fig = plt.figure()
    for i in range(3):
        ax = fig.add_subplot(2, 3, i + 1, projection='2d')
        ax.scatter(eigenvectors[:, i], eigenvectors[:, i + 1], eigenvectors[:, i + 2], c=y_pred, cmap=plt.hot())
        ax.set_title(f'Eigenvectors {i + 1} - {i + 3}')
    ax = fig.add_subplot(2, 3, 4)
    # im = ax.imshow(Ach, cmap='viridis')
    im = ax.imshow(Ach, cmap=plt.hot())
    ax.set_title('Affinity Matrix')
    fig.colorbar(im)
    plt.tight_layout()
    plt.show()

