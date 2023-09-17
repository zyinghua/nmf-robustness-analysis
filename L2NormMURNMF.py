# Author: Yinghua Zhou
# Creation Date: 2023/09/12

import numpy as np
import matplotlib.pyplot as plt
from time import time
import metrics


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
        self.V_clean, self.V, self.W, self.H, self.Y = None, None, None, None, None

    def init_factors(self, V):
        """
        Initialize the dictionary matrix and transformed data matrix *randomly*.

        :param V: Original data matrix.
        :return: W, H
        """

        W = self.np_rand.rand(V.shape[0], self.rank)
        H = self.np_rand.rand(self.rank, V.shape[1])

        return W, H

    def reconstruct_train(self):
        """
        Reconstruct the trained data matrix from the dictionary matrix and transformed data matrix.

        :return: approximated V
        """
        return self.W @ self.H

    def fit(self, V_clean, V, Y, steps=1000, e=1e-7, d=0.001, verbose=False, plot=False, plot_interval=None):
        """
        Perform *Multiplicative Update Rule* for Non-Negative Matrix Factorization.

        :param V_clean: Original non-contaminated data matrix.
        :param V: Original data matrix. (Contains contamination)
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
        assert V_clean is not None, "Please provide the original non-contaminated data matrix from the dataset."
        assert V is not None, "Please provide the original data matrix from the dataset."
        assert Y is not None, "Please provide the original labels from the dataset."

        if plot_interval is None:
            plot_interval = steps // 10

        self.W, self.H = self.init_factors(V)
        self.V_clean, self.V, self.Y = V_clean, V, Y

        rmse, aa, nmi = [], [], []

        start = time()

        for s in range(steps):
            """Please note in the corresponding tutorial, H is updated first, then W."""
            Hu = self.H * (self.W.T @ self.V) / (self.W.T @ self.W @ self.H + e) + e  # Update H
            Wu = self.W * (self.V @ Hu.T) / (self.W @ Hu @ Hu.T + e) + e  # Update W

            d_W = np.sqrt(np.sum((Wu-self.W)**2, axis=(0, 1)))/self.W.size
            d_H = np.sqrt(np.sum((Hu-self.H)**2, axis=(0, 1)))/self.H.size

            if d_W < d and d_H < d:
                if verbose:
                    print('Converged at step {}.'.format(s))
                break

            self.W = Wu
            self.H = Hu

            if plot and s % plot_interval == 0:
                rmse_, aa_, nmi_ = metrics.evaluate(self.V_clean, self.W, self.H, self.Y)
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
