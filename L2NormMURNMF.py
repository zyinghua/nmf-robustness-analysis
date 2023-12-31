import numpy as np
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

        avg = np.sqrt(V.mean() / self.rank)
        self.H = avg * self.np_rand.standard_normal(size=(self.rank, V.shape[1])).astype(V.dtype, copy=False)
        self.W = avg * self.np_rand.standard_normal(size=(V.shape[0], self.rank)).astype(V.dtype, copy=False)

        np.abs(self.H, out=self.H)
        np.abs(self.W, out=self.W)

    def reconstruct_train(self):
        """
        Reconstruct the trained data matrix from the dictionary matrix and transformed data matrix.

        :return: approximated V
        """
        return self.W @ self.H

    def fit_transform(self, V_clean, V, Y, steps=500, e=1e-7, d=1e-7, verbose=False, plot=False, plot_interval=50):
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

        self.init_factors(V)
        self.V_clean, self.V, self.Y = V_clean, V, Y

        rre, aa, nmi = [], [], []

        if verbose:
            print("Start training...")

        start = time()

        for s in range(steps):
            Hu = self.H * (self.W.T.dot(self.V)) / (self.W.T.dot(self.W.dot(self.H)) + e) + e  # Update H
            Wu = self.W * (self.V.dot(Hu.T)) / (self.W.dot(Hu.dot(Hu.T)) + e) + e  # Update W

            d_W = np.sqrt(np.sum((Wu - self.W) ** 2, axis=(0, 1))) / self.W.size
            d_H = np.sqrt(np.sum((Hu - self.H) ** 2, axis=(0, 1))) / self.H.size

            if d_W < d and d_H < d:
                if verbose:
                    print('Converged at step {}.'.format(s))
                break

            self.W = Wu
            self.H = Hu

            if plot and s % plot_interval == 0:
                rre_, aa_, nmi_ = metrics.evaluate(self.V_clean, self.W, self.H, self.Y)
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
