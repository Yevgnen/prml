#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy as sp
from scipy import linalg as spla
from scipy import random as sprd

from .kernel import GaussianKernel, LinearKernel


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
    def __init__(self, M):
        super(ProbabilisticPCA, self).__init__(M)
        self.latent_mean = 0
        self.latent_cov = 1

    def fit(self, X):
        n_samples, n_features = X.shape if X.ndim > 1 else (1, X.shape[0])
        n_components = self.n_components

        # Predict mean
        mu = X.mean(axis=0)

        # Predict covariance
        cov = sp.cov(X, rowvar=0)
        eigvals, eigvecs = self._eig_decomposition(cov)
        sigma2 = (sp.sum(cov.diagonal()) - sp.sum(eigvals.sum())) / (n_features - n_components) # FIXME: M < D?

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

    def reconstruct(self, X):
        latent = sp.dot(self.inv_M, sp.dot(self.weight.T, (X - self.predict_mean).T))
        eps = sprd.normal(0, sp.sqrt(self.sigma2)) # FIXME???
        recons = sp.dot(self.weight, latent) + self.predict_mean + eps

        return recons


class KernelPCA(PCABase):
    def __init__(self, M, kernel='gaussian', sigma=0.2):
        super(KernelPCA, self).__init__(M)
        self.supported_kernel = {
            'gaussian': GaussianKernel(sigma=sigma),
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
