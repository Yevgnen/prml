#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as linalg

from kernel import GaussianKernel


class GaussianProcess(object):
    def __init__(self, beta=100.0):
        self.beta = beta
        self.kernel = GaussianKernel()

    def predict_one(self, X, T, x):
        n_sample = X.shape[0]

        k = self.kernel.inner(X, x)
        c = self.kernel.inner(x, x) + 1 / self.beta
        C = self.kernel.inner(X, X) + np.eye(n_sample) / self.beta
        inv_C = linalg.inv(C)

        self.pred_mean = k.T.dot(inv_C).dot(T)
        self.pred_cov = c - k.T.dot(inv_C).dot(k)

        return self.pred_mean, self.pred_cov

    def predict(self, X, T, X_new):
        n_tests = X_new.shape[0]
        pred_mean = np.zeros(n_tests)
        pred_cov = np.zeros(n_tests)
        for i in range(n_tests):
            pred_mean[i], pred_cov[i] = self.predict_one(X, T, X_new[i])

        return pred_mean, pred_cov


    def score(self, X, T):
        Y = self.predict(X)

        return np.mean(Y == T)
