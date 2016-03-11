#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy as sp
from scipy import linalg as spla
from scipy import random as sprd


class PCA(object):
    def __init__(self, M):
        self.n_components = M

    def minimize_error(self, X):
        cov = sp.cov(X, rowvar=0)
        return

    def _eig_decomposition(self, A, largest=True):
        n_dims = A.shape[0]
        if (largest):
            eig_range = (n_dims - self.n_components, n_dims - 1)
        else:
            eig_range = (0, self.n_components - 1)

        eigvals, eigvecs = spla.eigh(A, eigvals=eig_range)

        if (largest):
            eigvals = eigvals[::-1]
            eigvecs = sp.fliplr(eigvecs)

        return eigvals, eigvecs

    def fit(self, X):
        cov = sp.cov(X, rowvar=0)
        eigvals, eigvecs = self._eig_decomposition(cov, largest=True)

        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.mean = X.mean(axis=0)

        return sp.dot(X, eigvecs)

    def reconstruct(self, X):
        reconstructed = self.mean + sp.dot(sp.dot(X - self.mean, self.eigvecs), self.eigvecs.T)

        return reconstructed
