# Author: Yinghua Zhou
# Creation Date: 2023/09/14


import numpy as np
import matplotlib.pyplot as plt
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
    kmeans = KMeans(n_clusters=len(set(Y)), random_state=0).fit(X)
    Y_pred = np.zeros(Y.shape)
    for i in set(kmeans.labels_):
        """for each centroid, label its instances by majority"""
        ind = kmeans.labels_ == i  # get the index of instances which labeled as i
        Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0]  # assign label.
    return Y_pred


def calc_rre(V_clean, W, H):
    """
    Calculate the Relative Reconstruction Error (RRE) between the original data matrix and the reconstructed data matrix.
    :return: RRE
    """
    return np.linalg.norm(V_clean - W @ H, ord="fro") / np.linalg.norm(V_clean, ord="fro")


def calc_aa(Y, Y_pred):
    """Calculate the Average Accuracy (AA) between the original labels and the predicted labels."""
    return accuracy_score(Y, Y_pred)


def calc_nmi(Y, Y_pred):
    """Calculate the Normalized Mutual Information (NMI) between the original labels and the predicted labels."""
    return normalized_mutual_info_score(Y, Y_pred)


def evaluate(V_clean, W, H, Y):
    """
    Evaluate the performance of the model by calculating the following metrics:
    1. Relative Reconstruction Error (RRE)
    2. Average Accuracy
    3. Normalized Mutual Information (NMI)

    :return: RRE, AA, NMI
    """
    assert V_clean is not None, "Please provide the original non-contaminated data matrix from the dataset."
    assert W is not None, "Please provide the dictionary matrix."
    assert H is not None, "Please provide the transformed data matrix."
    assert Y is not None, "Please provide the original labels from the dataset."

    rre = calc_rre(V_clean, W, H)

    Y_pred = assign_cluster_label(H.T, Y)
    # Y_pred_ori = assign_cluster_label(V_clean.T, Y)

    aa = calc_aa(Y, Y_pred)
    nmi = calc_nmi(Y, Y_pred)

    return rre, aa, nmi


def plot_metrics(rre, aa, nmi, plot_interval):
    plt.figure(figsize=(15, 5))

    # Plot for Rooted Mean Squared Error
    plt.subplot(1, 3, 1)  # 1 row, 3 columns, first plot
    plt.plot(np.array(range(len(rre))) * plot_interval, rre)
    plt.xlabel('Steps')
    plt.ylabel('Rooted Mean Squared Error')
    plt.title('Rooted Mean Squared Error')

    # Plot for Average Accuracy
    plt.subplot(1, 3, 2)  # 1 row, 3 columns, second plot
    plt.plot(np.array(range(len(aa))) * plot_interval, aa)
    plt.xlabel('Steps')
    plt.ylabel('Average Accuracy')
    plt.title('Average Accuracy')

    # Plot for Normalized Mutual Information
    plt.subplot(1, 3, 3)  # 1 row, 3 columns, third plot
    plt.plot(np.array(range(len(nmi))) * plot_interval, nmi)
    plt.xlabel('Steps')
    plt.ylabel('Normalized Mutual Information')
    plt.title('Normalized Mutual Information')

    # Show all plots
    plt.tight_layout()
    plt.show()

