# Author: Yinghua Zhou
# Creation Date: 2023/09/14


import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score


def assign_cluster_label(X, Y):
    """
    Assign cluster labels to the transformed data matrix via KMeans clustering.
    :param X: Transformed data matrix.
    :param Y: Original labels.
    :return: Y_pred

    Acknowledgement: This function is completely from the assignment 1 instruction ipynb file of COMP4328/5328
    Advanced Machine Learning course at University of Sydney.
    """
    kmeans = KMeans(n_clusters=len(set(Y))).fit(X)
    Y_pred = np.zeros(Y.shape)
    for i in set(kmeans.labels_):
        """for each centroid, label its instances by majority"""
        ind = kmeans.labels_ == i  # get the index of instances which labeled as i
        Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0]  # assign label.
    return Y_pred


def calc_rmse(V, W, H):
    """
    Calculate the Rooted Mean Squared Error (RMSE) between the original data matrix and the reconstructed data matrix.
    :return: RMSE
    """
    return np.linalg.norm(V - W @ H, ord='fro')  # Apply Frobenius norm


def calc_aa(Y, Y_pred):
    """Calculate the Average Accuracy (AA) between the original labels and the predicted labels."""
    return accuracy_score(Y, Y_pred)


def calc_nmi(Y, Y_pred):
    """Calculate the Normalized Mutual Information (NMI) between the original labels and the predicted labels."""
    return normalized_mutual_info_score(Y, Y_pred)


def evaluate(V, W, H, Y):
    """
    Evaluate the performance of the model by calculating the following metrics:
    1. Rooted Mean Squared Error (RMSE)
    2. Average Accuracy
    3. Normalized Mutual Information (NMI)

    :return: RMSE, AA, NMI
    """
    assert V is not None, "Please provide the original data matrix from the dataset."
    assert W is not None, "Please provide the dictionary matrix."
    assert H is not None, "Please provide the transformed data matrix."
    assert Y is not None, "Please provide the original labels from the dataset."

    rmse = calc_rmse(V, W, H)

    Y_pred = assign_cluster_label(H.T, Y)
    aa = calc_aa(Y, Y_pred)
    nmi = calc_nmi(Y, Y_pred)

    return rmse, aa, nmi
