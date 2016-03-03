#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as linalg

from kernel import GaussianKernel


class GaussianProcess(object):
    def __init__(self):
        self.kernel = GaussianKernel()

    def predict_one(self, X, T, x, beta, C):
        """Predict one single new point ``x``.
        The hyperparameter ``beta`` and Gram matrix ``C`` should be estimated
        and calculated from the training data, respectively. """

        kXx = self.kernel.inner(X, x)
        kxx = self.kernel.inner(x, x)

        inv_C = linalg.inv(C)
        c = kxx + 1 / beta

        self.pred_mean = kXx.T.dot(inv_C).dot(T)
        self.pred_cov = c - kXx.T.dot(inv_C).dot(kXx)  # Why the cov so small ?

        return self.pred_mean, self.pred_cov

    def hyperparameter_estimate(self,
                                X,
                                T,
                                beta=10000.0, # How to choose initial point ?
                                max_iter=int(1e3),
                                tol=1e-5,
                                step=0.8,
                                eta=0.9):
        """Use maximum likelihood to estimate ``beta``."""

        n_samples = X.shape[0]
        kXX = self.kernel.inner(X, X)

        C = kXX + np.eye(n_samples) / beta
        for iter in range(max_iter):
            inv_C = linalg.inv(C)
            dC_dbeta = -1 / beta**2 * np.eye(n_samples)
            dbeta = -0.5 * np.trace(inv_C.dot(dC_dbeta)) + 0.5 * T.T.dot(
                inv_C).dot(dC_dbeta).dot(inv_C).dot(T)
            delta = step * dbeta

            if (np.abs(delta) / np.abs(beta) < tol):
                break

            beta = beta + delta
            C = kXX + np.eye(n_samples) / beta
            step = step * eta

        return beta, C

    def predict(self, X, T, X_new):
        """Predict ``X_new`` with given traning data ``(X, T)``."""

        beta, C = self.hyperparameter_estimate(X, T)

        n_tests = X_new.shape[0]
        pred_mean = np.zeros(n_tests)
        pred_cov = np.zeros(n_tests)
        for i in range(n_tests):
            pred_mean[i], pred_cov[i] = self.predict_one(X, T, X_new[i], beta,
                                                         C)
        return pred_mean, pred_cov

    def score(self, X, T):
        Y = self.predict(X)

        return np.mean(Y == T)
