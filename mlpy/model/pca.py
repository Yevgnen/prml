#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy as sp
from scipy import linalg as spla
from scipy import random as sprd
from scipy.stats import multivariate_normal

from .kernel import GaussianKernel, LinearKernel, PolynomialKernel


class PCABase(object):
    def __init__(self, M):
        self.n_components = M

    def _eig_decomposition(self, A, largest=True):
        n_features = A.shape[0]

        if (largest):
            eig_range = (n_features - self.n_components, n_features - 1)
        else:
            eig_range = (0, self.n_components - 1)

        eigvals, eigvecs = spla.eigh(A, eigvals=eig_range)

        if (largest):
            eigvals = eigvals[::-1]
            eigvecs = sp.fliplr(eigvecs)

        return eigvals, eigvecs


class PCA(PCABase):
    def __init__(self, M):
        super(PCA, self).__init__(M)

    def fit(self, X):
        cov = sp.cov(X, rowvar=0)
        eigvals, eigvecs = self._eig_decomposition(cov)

        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.mean = X.mean(axis=0)

        return sp.dot(X, eigvecs)

    def reconstruct(self, X):
        reconstructed = self.mean + sp.dot(sp.dot(X - self.mean, self.eigvecs), self.eigvecs.T)

        return reconstructed


class ProbabilisticPCA(PCABase):
    def __init__(self, M, method='ml', max_iter=int(1e1), tol=1e-3):
        super(ProbabilisticPCA, self).__init__(M)
        self.latent_mean = 0
        self.latent_cov = 1
        self.method = method
        self.max_iter = max_iter
        self.tol = tol

    def _maximum_likelihood(self, X):
        n_samples, n_features = X.shape if X.ndim > 1 else (1, X.shape[0])
        n_components = self.n_components

        # Predict mean
        mu = X.mean(axis=0)

        # Predict covariance
        cov = sp.cov(X, rowvar=0)
        eigvals, eigvecs = self._eig_decomposition(cov)
        sigma2 = ((sp.sum(cov.diagonal()) - sp.sum(eigvals.sum())) /
                  (n_features - n_components))  # FIXME: M < D?

        weight = sp.dot(eigvecs, sp.diag(sp.sqrt(eigvals - sigma2)))
        M = sp.dot(weight.T, weight) + sigma2 * sp.eye(n_components)
        inv_M = spla.inv(M)

        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.predict_mean = mu
        self.predict_cov = sp.dot(weight, weight.T) + sigma2 * sp.eye(n_features)
        self.latent_mean = sp.transpose(sp.dot(inv_M, sp.dot(weight.T, X.T - mu[:, sp.newaxis])))
        self.latent_cov = sigma2 * inv_M
        self.sigma2 = sigma2    # FIXME!
        self.weight = weight
        self.inv_M = inv_M

        return self.latent_mean

    def _em(self, X):
        # Constants
        n_samples, n_features = X.shape
        n_components = self.n_components
        max_iter = self.max_iter
        # tol = self.tol

        mu = X.mean(axis=0)
        X_centered = X - sp.atleast_2d(mu)

        # Initialize parameters
        latent_mean = 0
        sigma2 = 1
        weight = sprd.randn(n_features, n_components)

        # Main loop of EM algorithm
        for i in range(max_iter):
            # E step
            M = sp.dot(weight.T, weight) + sigma2 * sp.eye(n_components)
            inv_M = spla.inv(M)
            latent_mean = sp.dot(inv_M, sp.dot(weight.T, X_centered.T)).T

            # M step
            expectation_zzT = n_samples * sigma2 * inv_M + sp.dot(latent_mean.T, latent_mean)

            # Re-estimate W
            weight = sp.dot(sp.dot(X_centered.T, latent_mean), spla.inv(expectation_zzT))
            weight2 = sp.dot(weight.T, weight)

            # Re-estimate \sigma^2
            sigma2 = ((spla.norm(X_centered)**2 -
                       2 * sp.dot(latent_mean.ravel(), sp.dot(X_centered, weight).ravel()) +
                       sp.trace(sp.dot(expectation_zzT, weight2))) /
                      (n_samples * n_features))

        self.predict_mean = mu
        self.predict_cov = sp.dot(weight, weight.T) + sigma2 * sp.eye(n_features)
        self.latent_mean = latent_mean
        self.latent_cov = sigma2 * inv_M
        self.sigma2 = sigma2
        self.weight = weight
        self.inv_M = inv_M

        return self.latent_mean

    def fit(self, X):
        func = {
            'ml': self._maximum_likelihood,
            'em': self._em
        }

        return func[self.method](X)

    def reconstruct(self, X):
        n_features = sp.atleast_2d(X).shape[1]
        latent = sp.dot(self.inv_M, sp.dot(self.weight.T, (X - self.predict_mean).T))
        eps = sprd.multivariate_normal(sp.zeros(n_features), self.sigma2 * sp.eye(n_features))
        recons = sp.dot(self.weight, latent) + self.predict_mean + eps

        return recons


class KernelPCA(PCABase):
    def __init__(self, M, kernel='gaussian', sigma=0.2, degree=3, coef0=1):
        super(KernelPCA, self).__init__(M)
        self.supported_kernel = {
            'gaussian': GaussianKernel(sigma=sigma),
            'polynomial': PolynomialKernel(degree=degree, coef0=coef0),
            'linear': LinearKernel()
        }
        self.kernel = self.supported_kernel[kernel]

    def fit(self, X):
        n_samples, n_featurs = X.shape
        K = self.kernel.inner(X, X)
        I1N = sp.ones((n_samples, n_samples))
        K_centered = K - sp.dot(I1N, K) - sp.dot(K, I1N) + sp.dot(sp.dot(I1N, K), I1N)

        eigvals, eigvecs = self._eig_decomposition(K_centered)

        self.eigvals = eigvals
        self.eigvecs = eigvecs

        Y = sp.dot(K, eigvecs)

        return Y
