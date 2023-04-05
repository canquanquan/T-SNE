import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

class TSNE:
    def __init__(self, X, perp, T, eta, alpha, n_components = 2):
        self.X = X
        self.perp = perp
        self.T = T
        self.eta = eta
        self.alpha = alpha
        self.n_components = n_components

    def fit_transform(self):
        p = self.P()
        Y = 0
        self.optimize(Y)

    """
    This function computes the pairwise similarities among datapoints in the high-dimensional dataset X
    """
    def P(self):
        return

    """
    This function computes the pairwise similarities among datapoints in the low-dimensional map Y
    """
    def Q(self, Y):
        return

    def cost_gradient(self, Y):
        return

    def update(self):
        return

    def check_convergence(self):
        return

    def optimize(self, Y):
        t = 0
        is_convergence = False
        while t<self.T and not is_convergence:
            q = self.Q(Y)
            gradient = self.cost_gradient(Y)
            self.update()

            if self.check_convergence(): break

        return