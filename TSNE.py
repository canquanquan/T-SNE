import pandas as pd
import numpy as np
import tensorflow_probability as tfp
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sb


class TSNE:
    def __init__(self, X, perp, T, eta, alpha, n_components=2):
        self.X = X
        self.perp = perp
        self.T = T
        self.eta = eta
        self.alpha = alpha
        self.n_components = n_components

    def fit_transform(self):
        p = self.P()
        Y = 0
        Y = self.optimize(Y)
        return Y

    # This function computes the pairwise similarities among datapoints in the high-dimensional dataset X
    # Use Gaussian distribution here
    # return an n*n matrix p
    # p[i][j] = p_{j|i}
    def Pi_cond(self, i, var):
        xi = self.X[i]
        pi = np.array([np.exp(-np.linalg.norm(xi - xi) ** 2 / (2 * var ** 2)) for xj in self.X])
        pi[i] = 0
        return pi / np.sum(pi)

    def P_cond(self, var):
        num_datapoints = self.X.shape[0]
        p_cond = []
        for i in range(num_datapoints):
            p_cond.append(self.Pi_cond(i, var))
        return np.array(p_cond)

    # p_{ij} = (p_{j|i} + p_{i|j}) / 2
    def P(self, var):
        num_datapoints = self.X.shape[0]
        p_cond = self.P_cond(var)
        p = np.zeros((num_datapoints, num_datapoints))
        for i in range(num_datapoints):
            for j in range(num_datapoints):
                p[i][j] = (p_cond[i][j] + p_cond[j][i]) / 2
        return p

    # This function computes the pairwise similarities among datapoints in the low-dimensional map Y
    # User Student t-distribution here
    def Qi(self, i, Y):
        yi = Y[i]
        qi = [(1 + np.linalg.norm(yi - yj) ** 2) ** (-1) for yj in Y]
        qi[i] = 0
        return qi

    def Q(self, Y):
        q = [self.Qi(i, Y) for i in range(Y.shape[0])]
        return q

    def cost_gradient(self, p, q, i, Y):
        j = 0
        gradient = (p[i][j] - q[i][j]) * (Y[i] - Y[j]) * (1 + np.linalg.norm(Y[i], Y[j]) ** 2) ** (-1)
        return gradient

    def update(self):
        return

    def check_convergence(self):
        return

    def optimize(self, Y):
        t = 0
        is_convergence = False
        while t < self.T and not is_convergence:
            q = self.Q(Y)
            gradient = self.cost_gradient(Y)
            self.update()

            if self.check_convergence(): break

        return


if __name__ == '__main__':
    X = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    test = TSNE(X, perp=3, T=5, eta=10, alpha=10, n_components=2)

    print(test.P(1))
