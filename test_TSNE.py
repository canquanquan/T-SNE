from unittest import TestCase
import pandas as pd
import numpy as np

from TSNE import TSNE


class TestTSNE(TestCase):
    X = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    TestCase.tsne = TSNE(X, perp=3, T=5, eta=10, alpha=10, n_components=2)

    def test_pi(self):
        self.fail()

    def test_p(self):
        self.fail()


if __name__ == '__main__':
    t = TestTSNE()
