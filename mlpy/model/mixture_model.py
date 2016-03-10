#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy as sp
from scipy import linalg as spla
from scipy import random as sprand
from scipy.stats import multivariate_normal


class KMean(object):
    def __init__(self, K=2, max_iter=int(1e2), tol=1e-3):
        self.n_classes = K
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
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
            print('K-Means: {0:5d} {1:4e}'.format(iter, spla.norm(diff) / spla.norm(center)))
            if (spla.norm(diff) < tol * spla.norm(center)):
                break

            center = center_new

        self.center = center.T
        self.responsibility = responsilibity

        return self

    def cluster(self, X):
        self.fit(X)

        cluster = [X[sp.argmax(self.responsibility, axis=1) == k] for k in range(self.n_classes)]
        mean = self.center
        cov = [sp.cov(c, rowvar=0, ddof=0) for c in cluster]

        return cluster, mean, cov

class GMM(object):
    def __init__(self, K=2, init='kmean', max_iter=int(1e2), tol=1e-3):
        self.n_components = K
        self.init = init
        self.max_iter = max_iter
        self.tol = tol

    def _init_params(self, X):
        init = self.init
        n_samples, n_features = X.shape
        n_components = self.n_components

        if (init == 'kmean'):
            km = KMean(n_components)
            clusters, mean, cov = km.cluster(X)
            coef = sp.array([c.shape[0] / n_samples for c in clusters])
            comps = [multivariate_normal(mean[i], cov[i], allow_singular=True) for i in range(n_components)]
        elif (init == 'rand'):
            coef = sp.absolute(sprand.randn(n_components))
            coef = coef / coef.sum()
            means = X[sprand.permutation(n_samples)[0: n_components]]
            clusters = [[] for i in range(n_components)]
            for x in X:
                idx = sp.argmin([spla.norm(x - mean) for mean in means])
                clusters[idx].append(x)

            comps = []
            for k in range(n_components):
                mean = means[k]
                cov = sp.cov(clusters[k], rowvar=0, ddof=0)
                comps.append(multivariate_normal(mean, cov, allow_singular=True))

        self.coef = coef
        self.comps = comps

    def pdf(self, X, k=None):
        # import ipdb; ipdb.set_trace()
        if (k is None):
            comps = self.comps
            coef = self.coef
        else:
             comps = [self.comps[k]]
             coef = [self.coef[k]]

        probability = sp.array([comp.pdf(X) for comp in comps])
        weight_probability = sp.dot(coef, probability)

        # return sp.sum(weight_probability, axis=0)
        return weight_probability

    def log_likelihood(self, X):
        return sp.sum(sp.log(self.pdf(X)))

    def fit(self, X):
        # Constants
        max_iter = self.max_iter
        tol = self.tol
        n_samples, n_features = X.shape
        n_components = self.n_components

        # Initialize parameters
        self._init_params(X)

        # Initialize
        responsibility = sp.empty((n_samples, n_components))
        log_likelihood = self.log_likelihood(X)

        for iter in range(max_iter):

            # E step
            for n in range(n_samples):
                for k in range(n_components):
                    responsibility[n][k] = self.pdf(X[n], k) / self.pdf(X[n])

            # M step
            eff = sp.sum(responsibility, axis=0)
            for k in range(n_components):
                # Update mean
                mean = sp.dot(responsibility[:, k], X) / eff[k]

                # Update covariance
                cov = sp.zeros((n_features, n_features))
                for n in range(n_samples):
                    cov += responsibility[n][k] * sp.outer(X[n] - mean, X[n] - mean)
                cov /= eff[k]

                # Update the k component
                self.comps[k] = multivariate_normal(mean, cov, allow_singular=True)

                # Update mixture coefficient
                self.coef[k] = eff[k] / n_samples

            # Convergent test
            log_likelihood_new = self.log_likelihood(X)
            diff = log_likelihood_new - log_likelihood
            print('GMM: {0:5d}: {1:10.5e} {2:10.5e}'.format(iter, log_likelihood_new, spla.norm(diff) / spla.norm(log_likelihood)))
            if (spla.norm(diff) < tol * spla.norm(log_likelihood)):
                break

            log_likelihood = log_likelihood_new

        return self
