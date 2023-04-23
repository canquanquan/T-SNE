import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.manifold._utils import _binary_search_perplexity
from sklearn.manifold import _barnes_hut_tsne


class TSNE(BaseEstimator):
    def __init__(self, n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=100, n_iter=1000, ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = 300
        self.min_grad_norm = 1e-7
        self.verbose = 0

    # Calculate high-dimensional affinities P
    def joint_prob(self, X):
        n_samples = X.shape[0]
        n_neighbors = min(n_samples - 1, int(3.0 * self.perplexity + 1))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', )
        knn.fit(X)
        dist = knn.kneighbors_graph(mode="distance")
        del knn
        dist.data **= 2
        dist.sort_indices()
        distances_data = dist.data.reshape(n_samples, -1)
        distances_data = distances_data.astype(np.float32, copy=False)
        conditional_P = _binary_search_perplexity(distances_data, self.perplexity, self.verbose)
        P = csr_matrix((conditional_P.ravel(), dist.indices, dist.indptr), shape=(n_samples, n_samples), )
        P = P + P.T
        return P / np.maximum(P.sum(), np.finfo(np.double).eps)

    # Calculate gradient & KL-divergence of P, Q
    def compute_kl_divergence(self, Q, P, n_samples, n_components, ):
        Q = Q.astype(np.float32, copy=False)
        Y = Q.reshape(n_samples, n_components)
        val_P = P.data.astype(np.float32, copy=False)
        neighbors = P.indices.astype(np.int64, copy=False)
        indptr = P.indptr.astype(np.int64, copy=False)
        grad = np.zeros(Y.shape, dtype=np.float32)
        error = _barnes_hut_tsne.gradient(val_P, Y, neighbors, indptr, grad, 0.5, n_components, False,
                                          dof=self.dof, compute_error=True, num_threads=1, )
        c = 2.0 * (self.dof + 1.0) / self.dof
        grad = grad.ravel()
        grad *= c
        return error, grad

    # Optimize the embedding using gradient descent
    def grad_descent(self, Q0, it, n_iter, n_iter_without_progress=300, momentum=0.8,
                     args=None, ):
        Q = Q0.copy().ravel()
        update = np.zeros_like(Q)
        gains = np.ones_like(Q)
        error = np.finfo(float).max
        best_error = np.finfo(float).max
        best_iter = i = it
        for i in range(it, n_iter):
            # Compute gradient & KL-divergence
            error, grad = self.compute_kl_divergence(Q, *args)
            # Update Y
            inc = update * grad < 0.0
            gains[inc] += 0.2
            gains[np.invert(inc)] *= 0.8
            np.clip(gains, 0.01, np.inf, out=gains)
            grad *= gains
            update = momentum * update - self.learning_rate * grad
            Q += update
            grad_norm = linalg.norm(grad)
            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                break
            if grad_norm <= self.min_grad_norm: break
        return Q, error, i

    # Optimize the Y embedding using KL-divergence
    def optimize(self, P, n_samples, Y, ):
        Q = Y.ravel()
        opt_args = {
            "it": 0,
            "args": [P, n_samples, self.n_components],
            "n_iter_without_progress": 250,
            "n_iter": 250,
            "momentum": 0.5,
        }
        P *= self.early_exaggeration
        Q, kl_divergence, it = self.grad_descent(Q, **opt_args)
        P /= self.early_exaggeration
        if it < 250 or self.n_iter - 250 > 0:
            opt_args["n_iter"] = self.n_iter
            opt_args["it"] = it + 1
            opt_args["momentum"] = 0.8
            opt_args["n_iter_without_progress"] = self.n_iter_without_progress
            Q, kl_divergence, it = self.grad_descent(Q, **opt_args)
        Y = Q.reshape(n_samples, self.n_components)
        return Y

    def fit(self, X):
        n_samples = X.shape[0]
        # 1) Calculate high-dimensional affinities P using KNN
        P = self.joint_prob(X)

        # Initialize the Y embedding
        # Take embedding produced using PCA as the initial embedding
        pca = PCA(n_components=self.n_components, svd_solver="randomized", )
        pca.set_output(transform="default")
        Y = pca.fit_transform(X).astype(np.float32, copy=False)

        # Optimize Y
        self.dof = max(self.n_components - 1, 1)
        return self.optimize(P, n_samples, Y, )

    def fit_transform(self, X, y=None):
        self.Y = self.fit(X)
        return self.Y
