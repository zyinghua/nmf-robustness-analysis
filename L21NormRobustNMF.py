# Author: Yinghua Zhou
# Creation Date: 2023/09/17

import numpy as np
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

        self.X_clean, self.X, self.W, self.H, self.Y = None, None, None, None, None

    def init_factors(self, X):
        """
        Initialize the dictionary matrix and transformed data matrix *randomly*.

        :param X: Original data matrix. (Contaminated)
        :return: W, H
        """

        avg = np.sqrt(X.mean() / self.rank)
        self.H = avg * self.np_rand.standard_normal(size=(self.rank, X.shape[1])).astype(X.dtype, copy=False)
        self.W = avg * self.np_rand.standard_normal(size=(X.shape[0], self.rank)).astype(X.dtype, copy=False)

        np.abs(self.H, out=self.H)
        np.abs(self.W, out=self.W)

        return self.W, self.H

    def reconstruct_train(self):
        """
        Reconstruct the clean data matrix from the dictionary matrix and transformed data matrix.

        :return: approximated clean data matrix.
        """
        return self.W @ self.H

    def fit_transform(self, X_clean, X, Y, steps=500, e=1e-7, d=1e-6, verbose=False, plot=False, plot_interval=50):
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
        :return: W, H
        """
        assert X_clean is not None, "Please provide the original non-contaminated data matrix from the dataset."
        assert X is not None, "Please provide the original data matrix from the dataset."
        assert Y is not None, "Please provide the original labels from the dataset."

        self.init_factors(X)
        self.X_clean, self.X, self.Y = X_clean, X, Y

        rre, aa, nmi = [], [], []

        if verbose:
            print("Start training...")

        start = time()

        for s in range(steps):
            D = np.diag(1 / (np.sqrt(np.sum((self.X - self.W.dot(self.H)) ** 2, axis=0)) + e))

            Wu = self.W * ((self.X.dot(D.dot(self.H.T))) / (self.W.dot(self.H.dot(D.dot(self.H.T))) + e))
            Hu = self.H * ((Wu.T.dot(self.X.dot(D))) / (Wu.T.dot(Wu.dot(self.H.dot(D))) + e))

            d_W = np.sqrt(np.sum((Wu - self.W) ** 2, axis=(0, 1))) / self.W.size
            d_H = np.sqrt(np.sum((Hu - self.H) ** 2, axis=(0, 1))) / self.H.size

            if d_W < d and d_H < d:
                if verbose:
                    print('Converged at step {}.'.format(s))
                break

            self.W = Wu
            self.H = Hu

            if plot and s % plot_interval == 0:
                rre_, aa_, nmi_ = metrics.evaluate(self.X_clean, self.W, self.H, self.Y)
                rre.append(rre_)
                aa.append(aa_)
                nmi.append(nmi_)

                if verbose:
                    print('Step: {}, RRE: {:.4f}, AA: {:.4f}, NMI: {:.4f}'.format(s, rre_, aa_, nmi_))

        if plot:
            metrics.plot_metrics(rre, aa, nmi, plot_interval)

        if verbose:
            print('Training Time taken: {:.2f} seconds.'.format(time() - start))

        return self.W, self.H

