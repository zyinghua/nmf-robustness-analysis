# Author: Yinghua Zhou
# Creation Date: 2023/09/14

import numpy as np
import matplotlib.pyplot as plt
import metrics
from time import time


class RobustNMF:
    def __init__(self, rank, lambda_, random_state=0):
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

        self.X_clean, self.X, self.U, self.V, self.E, self.Y = None, None, None, None, None, None
        self.m, self.n = None, None  # Number of features/pixels, and number of samples (rows and cols)

    def init_factors(self, X):
        """
        Initialize the dictionary matrix and transformed data matrix *randomly*.

        :param X: Original data matrix.
        :return: U, V, E
        """

        self.U = self.np_rand.rand(X.shape[0], self.k)
        self.V = self.np_rand.rand(self.k, X.shape[1])
        self.E = self.np_rand.rand(X.shape[0], X.shape[1])

        self.m, self.n = X.shape

        return self.U, self.V, self.E

    def reconstruct_train(self):
        """
        Reconstruct the clean data matrix from the dictionary matrix and transformed data matrix.

        :return: approximated clean data matrix.
        """
        return self.U @ self.V

    def fit(self, X_clean, X, Y, steps=5000, e=1e-7, d=0.001, verbose=False, plot=False, plot_interval=100):
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
        :return: U, V, E
        """
        assert X_clean is not None, "Please provide the original non-contaminated data matrix from the dataset."
        assert X is not None, "Please provide the original data matrix from the dataset."
        assert Y is not None, "Please provide the original labels from the dataset."

        self.init_factors(X)
        self.X_clean, self.X, self.Y = X_clean, X, Y

        rmse, aa, nmi = [], [], []

        start = time()

        for s in range(steps):
            # Extract/Synthesize update components specified in the paper
            X_hat = self.X - self.E
            Uu = self.U * ((X_hat @ self.V.T) / (self.U @ self.V @ self.V.T)) + e

            X_tilde = np.vstack((self.X, np.zeros((1, self.n))))

            I = np.eye(self.m)

            e_m = np.full((1, self.m), np.sqrt(self.lambda_) * np.exp(1))
            U_tilde = np.vstack((np.hstack((Uu, I, -I)), np.hstack((np.zeros((1, self.k)), e_m, e_m))))
            S = np.abs(U_tilde.T @ U_tilde)

            Ep = (np.abs(self.E) + self.E) / 2
            En = (np.abs(self.E) - self.E) / 2

            V_tilde = np.vstack((self.V, np.vstack((Ep, En))))

            V_tilde = np.maximum(0, V_tilde - ((V_tilde * (U_tilde.T @ U_tilde @ V_tilde)) / (S @ V_tilde))
                                 + ((V_tilde * (U_tilde.T @ X_tilde)) / (S @ V_tilde)))

            Vu = V_tilde[:self.k, :]
            Epu = V_tilde[self.k:self.k+self.m, :]
            Enu = V_tilde[self.k+self.m:, :]

            Eu = Epu - Enu  # Mathematically, this operation gives you back E

            d_U = np.sqrt(np.sum((Uu-self.U)**2, axis=(0, 1)))/self.U.size
            d_V = np.sqrt(np.sum((Vu-self.V)**2, axis=(0, 1)))/self.V.size
            d_E = np.sqrt(np.sum((Eu-self.E)**2, axis=(0, 1)))/self.E.size

            if d_U < d and d_V < d and d_E < d:
                if verbose:
                    print('Converged at step {}.'.format(s))
                break

            self.U = Uu
            self.V = Vu
            self.E = Eu

            if plot and s % plot_interval == 0:
                rmse_, aa_, nmi_ = metrics.evaluate(self.X_clean, self.U, self.V, self.Y)
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

        return self.U, self.V, self.E
