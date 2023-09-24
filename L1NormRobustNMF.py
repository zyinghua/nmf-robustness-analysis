# Author: Yinghua Zhou
# Creation Date: 2023/09/14

import numpy as np
import metrics
from time import time


class L1NormRobustNMF:
    def __init__(self, rank, lambda_=0.04, random_state=0):
        """
        Initialization of the Robust Non-negative Matrix Factorization via L1 Norm Regularization NMF model.
        Hyper-parameters are defined here.

        :param rank: Rank of the dictionary matrix. Latent dimension.
        :param lambda_: Regularization parameter.
        :param random_state: Random seed.
        """
        assert rank is not None and rank > 0, "Please provide a valid integer for the rank of the dictionary matrix."
        assert lambda_ is not None and lambda_ > 0, "Please provide a valid numeric for the regularization parameter."

        self.np_rand = np.random.RandomState(random_state)
        self.k = rank
        self.lambda_ = lambda_

        self.X_clean, self.X, self.W, self.H, self.E, self.Y = None, None, None, None, None, None
        self.m, self.n = None, None  # Number of features/pixels, and number of samples (rows and cols)

    def init_factors(self, X):
        """
        Initialize the dictionary matrix and transformed data matrix and
        the noise matrix *randomly* according to Normal Distribution.

        :param X: Original data matrix.
        :return: W, H, E
        """

        self.m, self.n = X.shape

        avg = np.sqrt(X.mean() / self.k)
        self.H = avg * self.np_rand.standard_normal(size=(self.k, self.n)).astype(X.dtype, copy=False)
        self.W = avg * self.np_rand.standard_normal(size=(self.m, self.k)).astype(X.dtype, copy=False)
        self.E = avg * self.np_rand.standard_normal(size=(self.m, self.n)).astype(X.dtype, copy=False)

        np.abs(self.H, out=self.H)
        np.abs(self.W, out=self.W)
        np.abs(self.E, out=self.E)

        return self.W, self.H, self.E

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
        :param e: epsilon, for numerical instability.
        :param d: delta, param change threshold for stopping.
        :param verbose: True to print out the convergence information.
        :param plot: True to plot the convergence curve on the three nominated metrics.
        :param plot_interval: Plot the convergence curve on the metrics every plot_interval step.
        :return: U, V, E
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
            X_hat = self.X - self.E
            Wu = self.W * ((X_hat.dot(self.H.T)) / (self.W.dot(self.H.dot(self.H.T)) + e))

            Ep = (np.abs(self.E) + self.E) / 2
            En = (np.abs(self.E) - self.E) / 2

            V_tilde = np.vstack((self.H, np.vstack((Ep, En))))

            X_tilde = np.vstack((self.X, np.zeros(self.n)))

            I = np.eye(self.m)

            e_m = np.full((1, self.m), np.sqrt(self.lambda_) * np.exp(1))
            U_tilde = np.vstack((np.hstack((Wu, I, -I)), np.hstack((np.zeros((1, self.k)), e_m, e_m))))
            S = U_tilde.T.dot(U_tilde)

            V_tilde = np.maximum(0, V_tilde - ((V_tilde * (U_tilde.T.dot(U_tilde.dot(V_tilde)))) / (np.abs(S.dot(V_tilde)) + e)) + ((V_tilde * (U_tilde.T.dot(X_tilde))) / (np.abs(S.dot(V_tilde)) + e)))

            Hu = V_tilde[:self.k, :]
            Epu = V_tilde[self.k:self.k + self.m, :]
            Enu = V_tilde[self.k + self.m:, :]

            Eu = Epu - Enu  # Mathematically, this operation gives you back E

            d_W = np.sqrt(np.sum((Wu - self.W) ** 2, axis=(0, 1))) / self.W.size
            d_H = np.sqrt(np.sum((Hu - self.H) ** 2, axis=(0, 1))) / self.H.size
            d_E = np.sqrt(np.sum((Eu - self.E) ** 2, axis=(0, 1))) / self.E.size

            if d_W < d and d_H < d and d_E < d:
                if verbose:
                    print('Converged at step {}.'.format(s))
                break

            self.W = Wu
            self.H = Hu
            self.E = Eu

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

        return self.W, self.H, self.E
