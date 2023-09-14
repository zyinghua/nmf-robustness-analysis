# Author: Yinghua Zhou
# Creation Date: 2023/09/12

import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
from time import time


class L2NormMURNMF:
    def __init__(self, rank, random_state=0):
        """
        Initialize the L2-Norm Multiplicative Update Rule Non-Negative Matrix Factorization model.
        Hyper-parameters are defined here.

        :param rank: Rank of the dictionary matrix. Latent dimension.
        :param random_state: Random seed.
        """
        assert rank is not None and rank > 0, "Please provide a valid integer for the rank of the dictionary matrix."

        self.np_rand = np.random.RandomState(random_state)
        self.rank = rank
        self.V, self.W, self.H, self.Y = None, None, None, None

    def init_factors(self, V):
        """
        Initialize the dictionary matrix and transformed data matrix *randomly*.

        :param V: Original non-contaminated data matrix.
        :return: W, H
        """

        W = self.np_rand.rand(V.shape[0], self.rank)
        H = self.np_rand.rand(self.rank, V.shape[1])

        return W, H

    def reconstruct(self, W, H):
        """
        Reconstruct the data matrix from the dictionary matrix and transformed data matrix.

        :param W: Dictionary matrix.
        :param H: Transformed data matrix.
        :return: V
        """
        return W @ H

    def fit(self, V, Y, steps=5000, e=1e-7, d=0.001, verbose=False, plot=False, plot_interval=5):
        """
        Perform *Multiplicative Update Rule* for Non-Negative Matrix Factorization.

        :param V: Original non-contaminated data matrix.
        :param Y: Original labels.
        :param steps: Number of iterations.
        :param e: epsilon, added to the updates avoid numerical instability
        :param d: delta, threshold for rate of change at each step
        :param verbose: True to print out the convergence information
        :param plot: True to plot the convergence curve on the three nominated metrics
        :param plot_interval: Plot the convergence curve on the metrics every plot_interval step
        :return: W, H

        Acknowledgement: This function is inspired by and has components from the corresponding function
        in the week 6 tutorial ipynb file of COMP4328/5328 Advanced Machine Learning course at University of Sydney.
        """
        assert V is not None, "Please provide the original data matrix from the dataset."
        assert Y is not None, "Please provide the original labels from the dataset."

        self.W, self.H = self.init_factors(V)
        self.V, self.Y = V, Y

        rmse, aa, nmi = [], [], []

        start = time()

        for s in range(steps):
            Wu = self.W * (self.V @ self.H.T) / (self.W @ self.H @ self.H.T) + e
            Hu = self.H * (self.W.T @ self.V) / (self.W.T @ self.W @ self.H) + e

            d_W = np.sqrt(np.sum((Wu-self.W)**2, axis=(0, 1)))/self.W.size
            d_H = np.sqrt(np.sum((Hu-self.H)**2, axis=(0, 1)))/self.H.size

            if d_W < d and d_H < d:
                if verbose:
                    print('Converged at step {}.'.format(s))
                break

            self.W = Wu
            self.H = Hu

            if plot and s % plot_interval == 0:
                rmse_, aa_, nmi_ = self.evaluate(self.V, self.W, self.H, self.Y)
                rmse.append(rmse_)
                aa.append(aa_)
                nmi.append(nmi_)

        if plot:
            plt.figure(figsize=(10, 8))
            plt.plot(range(len(rmse)), rmse, label='Rooted Mean Squared Error')
            plt.plot(range(len(aa)), aa, label='Average Accuracy')
            plt.plot(range(len(nmi)), nmi, label='Normalized Mutual Information')
            plt.xlabel('Steps')
            plt.ylabel('Metrics')
            plt.title('Convergence Curve')

            plt.legend()
            plt.show()

        if verbose:
            print('Training Time taken: {:.2f} seconds.'.format(time()-start))

        return self.W, self.H

    def calc_rmse(self, V, W, H):
        """
        Calculate the Rooted Mean Squared Error (RMSE) between the original data matrix and the reconstructed data matrix.
        :return: RMSE
        """
        return np.linalg.norm(V - W @ H, ord='fro')  # Apply Frobenius norm

    def calc_aa_nmi(self, H, Y):
        """
        Calculate the Average Accuracy (AA) and Normalized Mutual Information (NMI) between the original labels and
        the predicted labels based on the transformed data matrix via KMeans clustering. As a mean to measure the
        goodness of the transformed data matrix.

        :param H: Transformed data matrix.
        :param Y: Original labels.
        :return: AA, NMI

        Acknowledgement: This function is inspired by and has components from the corresponding function
        in the asm 1 instruction ipynb file of COMP4328/5328 Advanced Machine Learning course at University of Sydney.
        """
        def assign_cluster_label(X, Y_):
            kmeans = KMeans(n_clusters=len(set(Y_))).fit(X)
            Y_pred_ = np.zeros(Y_.shape)
            for i in set(kmeans.labels_):
                """for each centroid, label its instances by majority"""
                ind = kmeans.labels_ == i  # get the index of instances which labeled as i
                Y_pred[ind] = Counter(Y_[ind]).most_common(1)[0][0]  # assign label.
            return Y_pred_

        Y_pred = assign_cluster_label(H.T, Y)
        aa = accuracy_score(Y, Y_pred)
        nmi = normalized_mutual_info_score(Y, Y_pred)

        return aa, nmi

    def evaluate(self, V, W, H, Y):
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

        rmse = self.calc_rmse(V, W, H)
        aa, nmi = self.calc_aa_nmi(H, Y)

        return rmse, aa, nmi
