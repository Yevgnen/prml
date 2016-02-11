#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as linalg


class LinearLeastSquare(object):
    def __init__(self, basis_functions=None, lamb=0):

        if (basis_functions == None):
            raise Exception('Please privode basis functions!')

        self.basis_functions = basis_functions
        self.n_basis = len(self.basis_functions)
        self.lamb = lamb

        return

    def nonlinear_transformation(self, x):
        if (x.ndim < 2):
            return np.array([phi(x) for phi in self.basis_functions])
        else:
            n_sample = x.shape[0]
            phi = np.zeros((n_sample, self.n_basis))
            for i in range(n_sample):
                phi[i, :] = self.nonlinear_transformation(x[i, :])

            return phi

    def fit(self, X, T):
        Phi = self.nonlinear_transformation(X)

        # MLE for w
        self.w = linalg.solve(
            Phi.T.dot(Phi) + np.eye(self.n_basis) * self.lamb, Phi.T.dot(T))

        # MLE for beta
        n_output = 1 if T.ndim < 2 else T.shape[1]
        self.beta = n_output / np.mean(linalg.norm(T - Phi.dot(self.w))**2)

        return self

    def predict(self, X):
        Phi = self.nonlinear_transformation(X)

        if (self.w.ndim == 1):
            y = self.w.dot(Phi.T).flatten()
        else:
            y = Phi.dot(self.w)  # self.w.T.dot(Phi.T).T

        return y


