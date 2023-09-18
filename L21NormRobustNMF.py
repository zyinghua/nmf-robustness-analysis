# Author: Yinghua Zhou
# Creation Date: 2023/09/17

import numpy as np
import matplotlib.pyplot as plt
import metrics
from time import time


class L21RobustNMF:
    def __init__(self, rank, random_state=0):
        """
        Initialization of the Robust Non-negative Matrix Factorization via L1 Norm Regularization NMF model.
        Hyper-parameters are defined here.

        :param rank: Rank of the dictionary matrix. Latent dimension.
        :param random_state: Random seed.
        """
        assert rank is not None and rank > 0, "Please provide a valid integer for the rank of the dictionary matrix."

        self.np_rand = np.random.RandomState(random_state)
        self.rank = rank

        self.X_clean, self.X, self.F, self.G, self.Y = None, None, None, None, None

    def init_factors(self, X):
        """
        Initialize the dictionary matrix and transformed data matrix *randomly*.

        :param X: Original data matrix. (Contaminated)
        :return: F, G
        """

        self.F = self.np_rand.rand(X.shape[0], self.rank)
        self.G = self.np_rand.rand(self.rank, X.shape[1])

        return self.F, self.G

    def reconstruct_train(self):
        """
        Reconstruct the clean data matrix from the dictionary matrix and transformed data matrix.

        :return: approximated clean data matrix.
        """
        return self.F @ self.G

    def fit_transform(self, X_clean, X, Y, steps=100, e=1e-7, d=1e-7, verbose=False, plot=False, plot_interval=10):
        """
        Perform the model learning via the specific MURs stated in the paper.

        :param X_clean: Original non-contaminated data matrix.
        :param X: Original contaminated data matrix.
        :param Y: Original labels.
        :param steps: Number of iterations.
        :param e: epsilon, added to the updates avoid numerical instability.
        :param d: delta, threshold for rate of change at each step.
        :param verbose: True to print out the convergence information.
        :param plot: True to plot the convergence curve on the three nominated metrics.
        :param plot_interval: Plot the convergence curve on the metrics every plot_interval step.
        :return: F, G
        """
        assert X_clean is not None, "Please provide the original non-contaminated data matrix from the dataset."
        assert X is not None, "Please provide the original data matrix from the dataset."
        assert Y is not None, "Please provide the original labels from the dataset."

        self.init_factors(X)
        self.X_clean, self.X, self.Y = X_clean, X, Y

        rmse, aa, nmi = [], [], []

        start = time()

        for s in range(steps):
            D = np.diag(1 / (np.sqrt(np.sum((self.X - self.F @ self.G) ** 2, axis=0)) + e))
            Fu = self.F * ((self.X @ D @ self.G.T) / (self.F @ self.G @ D @ self.G.T + e))

            D = np.diag(1 / (np.sqrt(np.sum((self.X - Fu @ self.G) ** 2, axis=0)) + e))  # update D with the new Fu
            Gu = self.G * ((Fu.T @ self.X @ D) / (Fu.T @ Fu @ self.G @ D + e))

            d_F = np.sqrt(np.sum((Fu-self.F)**2, axis=(0, 1)))/self.F.size
            d_G = np.sqrt(np.sum((Gu-self.G)**2, axis=(0, 1)))/self.G.size

            if d_F < d and d_G < d:
                if verbose:
                    print('Converged at step {}.'.format(s))
                break

            self.F = Fu
            self.G = Gu

            if plot and s % plot_interval == 0:
                rmse_, aa_, nmi_ = metrics.evaluate(self.X_clean, self.F, self.G, self.Y)
                rmse.append(rmse_)
                aa.append(aa_)
                nmi.append(nmi_)

                if verbose:
                    print('Step: {}, RMSE: {:.4f}, AA: {:.4f}, NMI: {:.4f}'.format(s, rmse_, aa_, nmi_))

        if plot:
            metrics.plot_metrics(rmse, aa, nmi, plot_interval)

        if verbose:
            print('Training Time taken: {:.2f} seconds.'.format(time()-start))

        return self.F, self.G
