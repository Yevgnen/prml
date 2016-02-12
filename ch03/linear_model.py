#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as linalg
from scipy.optimize import minimize as minimize


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

        if (self.w.ndim < 2):
            y = self.w.dot(Phi.T).flatten()
        else:
            y = Phi.dot(self.w)  # self.w.T.dot(Phi.T).T

        return y


class BayesianLinearRegression(object):
    def __init__(self, basis_functions=None, mean=0, cov=1):
        if (basis_functions == None):
            raise Exception('Please privode basis functions!')

        self.basis_functions = basis_functions
        self.n_basis = len(self.basis_functions)
        self.mean = mean
        self.cov = cov

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

    def fit(self, X, T, beta):
        Phi = self.nonlinear_transformation(X)
        PhiTPhi = np.outer(Phi, Phi) if Phi.ndim < 2 else Phi.T.dot(Phi)

        prior_mean = self.mean
        prior_cov = self.cov
        inv_prior_cov = linalg.inv(prior_cov)

        post_cov = linalg.inv(inv_prior_cov + beta * PhiTPhi)
        post_mean = post_cov.dot(inv_prior_cov.dot(prior_mean) + beta *
                                 Phi.T.dot(T))

        self.mean = post_mean
        self.cov = post_cov

    def predict(self, X, beta):
        Phi = self.nonlinear_transformation(X)

        w = self.mean
        if (w.ndim < 2):
            y = w.dot(Phi.T).flatten()
        else:
            y = Phi.dot(w)

        cov = 1 / beta + np.diag(Phi.dot(self.cov).dot(Phi.T))

        return y, cov


class EmpiricalBayes(object):
    def __init__(self, basis_functions=None):
        if (basis_functions == None):
            raise Exception('Please privode basis functions!')

        self.basis_functions = basis_functions
        self.n_basis = len(self.basis_functions)
        self.prior_precision = 0
        self.post_precision = 0

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

    def evidence_approximation(self,
                               X,
                               T,
                               alpha=1e-2,
                               beta=1e-2,
                               tol=1e-3,
                               max_iter=int(1e2)):
        n_basis = self.n_basis
        n_sample = X.shape[0]

        Phi = self.nonlinear_transformation(X)
        PhiTPhi = np.outer(Phi, Phi) if Phi.ndim < 2 else Phi.T.dot(Phi)

        lamb = linalg.eigvalsh(PhiTPhi)
        gamma = np.sum(lamb / (lamb + alpha))

        for iteration in range(max_iter):
            post_cov = linalg.inv(alpha * np.eye(n_basis) + beta * PhiTPhi)
            post_mean = beta * post_cov.dot(Phi.T.dot(T))

            alpha = gamma / linalg.norm(post_mean)**2
            beta = (n_sample -
                    gamma) / linalg.norm(T - Phi.dot(post_mean))**2  # FIXME

            gamma_new = np.sum(lamb / (lamb + alpha))

            if (linalg.norm(gamma_new - gamma) / linalg.norm(gamma) < tol):
                break

            gamma = gamma_new

        self.mean = post_mean
        self.prior_precision = alpha
        self.noise_precision = beta

        return self

    def fit(self, X, T):
        return self.evidence_approximation(X, T)

    def predict(self, X):
        Phi = self.nonlinear_transformation(X)

        if (self.mean.ndim < 2):
            y = self.mean.dot(Phi.T).flatten()
        else:
            y = Phi.dot(self.mean)

        cov = 1 / self.noise_precision + np.diag(Phi.dot(np.eye(
            self.n_basis) / self.noise_precision).dot(Phi.T))

        return y, cov
