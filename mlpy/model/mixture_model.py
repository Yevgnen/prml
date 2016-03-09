#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy as sp
from scipy import linalg as spla
from scipy import random as sprand


class KMean(object):
    def __init__(self, K=2, max_iter=int(1e2), tol=1e-3):
        self.n_classes = K
        self.max_iter = max_iter
        self.tol = tol

        return

    def fit(self, X):
        """Given the input data, cluster into K classes."""
        n_samples, n_features = X.shape
        n_classes = self.n_classes
        max_iter = self.max_iter
        tol = self.tol

        rand_center_idx = sprand.permutation(n_samples)[0:n_classes]
        center = X[rand_center_idx].T
        responsilibity = sp.zeros((n_samples, n_classes))

        for iter in range(max_iter):
            # E step
            dist = sp.expand_dims(X, axis=2) - sp.expand_dims(center, axis=0)
            dist = spla.norm(dist, axis=1)**2
            min_idx = sp.argmin(dist, axis=1)
            responsilibity.fill(0)
            responsilibity[sp.arange(n_samples), min_idx] = 1

            # M step
            center_new = sp.dot(X.T, responsilibity) / sp.sum(responsilibity, axis=0)
            diff = center_new - center
            print('{0:5d} {1:4e}'.format(iter, spla.norm(diff) / spla.norm(center)))
            if (spla.norm(diff) < tol * spla.norm(center)):
                break

            center = center_new

        self.center = center
        self.responsibility = responsilibity

        return self
