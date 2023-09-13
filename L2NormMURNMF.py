# Author: Yinghua Zhou
# Date: 2023/09/12

import numpy as np
import matplotlib.pyplot as plt
from time import time


class L2NormMURNMF:
    def __init__(self, V=None, rank=None):
        self.np_rand = np.random.RandomState(0)
        self.V = V
        self.rank = rank
        self.W, self.H = None, None

        if self.V is not None and self.rank is not None:
            self.W, self.H = self.init_factors(self.V, self.rank)

    def init_factors(self, V, rank):
        """
        Initialize the dictionary matrix and transformed data matrix *randomly*.

        :param V: Original non-contaminated data matrix.
        :param rank: Rank of the dictionary matrix.
        :return: W, H
        """

        W = self.np_rand.rand(V.shape[0], rank)
        H = self.np_rand.rand(rank, V.shape[1])

        return W, H

    def nmf_multiplicative_update(self, V, W, H, steps=5000, e=1e-7, d=0.001, verbose=False):
        """
        Perform Multiplicative Update Rule for Non-Negative Matrix Factorization.

        :param V: Original non-contaminated data matrix.
        :param W: Dictionary matrix.
        :param H: Transformed data matrix.
        :param steps: Number of iterations.
        :param e: epsilon, added to the updates avoid numerical instability
        :param d: delta, threshold for rate of change at each step
        :param verbose: True to print out the convergence information
        :return: D, R

        Acknowledgement: This function is inspired by and has components from the corresponding function
        in the week 6 tutorial ipynb file of COMP4328/5328 Advanced Machine Learning course at University of Sydney.
        """

        for s in range(steps):
            Wu = W * (V @ H.T) / (W @ H @ H.T) + e
            Hu = H * (W.T @ V) / (W.T @ W @ H) + e

            d_W = np.sqrt(np.sum((Wu-W)**2, axis=(0, 1)))/W.size
            d_H = np.sqrt(np.sum((Hu-H)**2, axis=(0, 1)))/H.size

            if d_W < d and d_H < d:
                if verbose:
                    print('Converged at step {}.'.format(s))
                break

            W = Wu
            H = Hu

        return W, H
