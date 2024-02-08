import numpy as np
import math

"""
the reference for this file is written below
https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index%5C
"""
def contingency_table(X,Y):
    unique_labels = np.unique(X)
    XS = [list(np.where(X == label)[0]) for label in unique_labels]
    unique_labels = np.unique(Y)
    YS = [np.where(Y == label)[0].tolist() for label in unique_labels]
    contingency_table = [[np.intersect1d(XS[i], YS[j]).size for j in range(len(YS))] for i in range(len(XS))]
    return contingency_table

def clustering_score(true_labels, predicted_labels):
    """
    Calculate the clustering score to assess the accuracy of predicted labels compared to true labels.

    Parameters:
    - true_labels: List or numpy array, true cluster labels for each data point.
    - predicted_labels: List or numpy array, predicted cluster labels for each data point.

    Returns:
    - score: float, clustering score indicating the accuracy of predicted labels.
    """
    contingency = contingency_table(true_labels,predicted_labels)
    A = np.sum(contingency,axis= 0)
    B = np.sum(contingency,axis= 1)
    a = np.sum(np.vectorize(math.comb)(contingency, 2))
    combinations = np.frompyfunc(math.comb, 2, 1)
    b = np.sum(combinations(A, 2))
    c = np.sum(np.frompyfunc(math.comb, 2, 1)(B, 2))
    d = b*c/math.comb(len(true_labels),2)
    e = (b + c)/2
    f = a - d
    g = e - d
    return f/g

if __name__ == "__main__":
    # Example usage:
    true_labels = np.array([0, 0, 1, 1, 1, 2, 2, 2])
    predicted_labels = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    print(clustering_score(true_labels, predicted_labels))


