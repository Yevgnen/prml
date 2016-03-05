#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as linalg
from scipy.optimize import minimize as minimize

from .kernel import GaussianKernel, LinearKernel


class SVC(object):
    def __init__(self, kernel='gaussian', C=1e4):
        self.supported_kernel = {
            'gaussian': GaussianKernel(),
            'linear': LinearKernel()
        }
        self.kernel = self.supported_kernel[kernel]
        self.C = C

    def fit(self, X, T, max_iter=int(1e3), tol=1e-5):
        """Use training data ``X`` and ``T`` to fit a SVC models."""
        n_samples = X.shape[0]
        # Compute the Gram matrix of training data
        K = self.kernel.inner(X, X)

        # The target function: (1/2)*x'*Q*x + p'*x
        Q = T.reshape(1, -1) * K * T.reshape(-1, 1)
        p = -np.ones(n_samples)
        lagrange = lambda x: (0.5 * x.dot(Q).dot(x) + p.dot(x), Q.dot(x) + p)

        # The equality constraints: H(x) = 0
        A = T
        cons = ({'type': 'eq', 'fun': lambda x: A.dot(x), 'jac': lambda x: A})

        # The inequality constaints: 0 <= G(x) <= C
        bnds = [(0, self.C) for i in range(n_samples)]

        # Solve the quadratic program
        opt_solution = minimize(lagrange,
                                np.zeros(n_samples),
                                method='SLSQP',
                                constraints=cons,
                                bounds=bnds,
                                tol=tol,
                                jac=True,
                                options={'maxiter': max_iter,
                                         'disp': True})

        self.dual_var = opt_solution.x
        self.sv_indices = np.nonzero((1 - np.isclose(self.dual_var, 0)))[0]
        self.inner_sv_indices = np.nonzero(
            (1 - np.isclose(self.dual_var, 0)) *
            (1 - np.isclose(self.dual_var, self.C)))[0]

        return self

    def predict(self, X, T, X_new):
        """Predict ``X_new`` with given traning data ``(X, T)``."""
        n_tests = X_new.shape[0]
        Y = np.zeros(n_tests)

        dual_var = self.dual_var
        sv_indices = self.sv_indices
        inner_sv_indices = self.inner_sv_indices

        sv_dual_var = dual_var[sv_indices]
        sv_X = X[sv_indices]
        inner_sv_X = X[inner_sv_indices]
        sv_T = T[inner_sv_indices]

        K = self.kernel.inner(inner_sv_X, sv_X)
        b = 1 / inner_sv_indices.size * (
            sv_T.sum() - K.dot(sv_dual_var * sv_T).sum())

        Y = (sv_T * sv_dual_var).dot(self.kernel.inner(sv_X, X_new)) + b

        Y[Y > 0] = 1
        Y[Y < 0] = -1

        return Y

    def score(self, X_train, T_train, X_test, T_test):
        Y = self.predict(X_train, T_train, X_test)

        return np.mean(np.isclose(Y, T_test))


class SVR(object):
    def __init__(self, kernel='gaussian', C=1e4, eps=1e-1):
        self.supported_kernel = {
            'gaussian': GaussianKernel(),
            'linear': LinearKernel()
        }
        self.kernel = self.supported_kernel[kernel]
        self.C = C
        self.eps = eps

    def fit(self, X, T, max_iter=int(1e3), tol=1e-5):
        """Use training data ``X`` and ``T`` to fit a SVC models."""
        n_samples = X.shape[0]
        n_dual_vars = 2 * n_samples
        # Compute the Gram matrix of training data
        K = self.kernel.inner(X, X)

        # The equality constraints: H(x) = 0
        ones = np.ones(n_samples)
        A = np.concatenate((ones, -ones))
        cons = ({'type': 'eq', 'fun': lambda x: A.dot(x), 'jac': lambda x: A})

        # The inequality constaints: 0 <= G(x) <= C
        bnds = [(0, self.C) for i in range(n_dual_vars)]

        # The target function: (1/2)*x'*Q*x + p'*x
        Q = np.array(np.bmat([[K, -K], [-K, K]]))
        p = self.eps - A * np.concatenate((T, T))
        lagrange = lambda x: (0.5 * x.dot(Q).dot(x) + p.dot(x), Q.dot(x) + p)

        # Solve the quadratic program
        opt_solution = minimize(lagrange,
                                np.zeros(n_dual_vars),
                                method='SLSQP',
                                constraints=cons,
                                bounds=bnds,
                                tol=tol,
                                jac=True,
                                options={'maxiter': max_iter,
                                         'disp': True})

        self.dual_var = np.array([None, None], dtype=np.object)
        self.dual_var[0] = opt_solution.x[:n_samples]
        self.dual_var[1] = opt_solution.x[n_samples:]

        self.sv_indices = np.array([None, None], dtype=np.object)
        self.sv_indices[0] = np.nonzero((1 - np.isclose(self.dual_var[0], 0)))[
            0]
        self.sv_indices[1] = np.nonzero((1 - np.isclose(self.dual_var[1], 0)))[
            0]

        self.union_sv_inices = np.union1d(*self.sv_indices)

        self.inner_sv_indices = np.array([None, None], dtype=np.object)
        self.inner_sv_indices[0] = np.nonzero(
            (1 - np.isclose(self.dual_var[0], 0)) *
            (1 - np.isclose(self.dual_var[0], self.C)))[0]
        self.inner_sv_indices[1] = np.nonzero(
            (1 - np.isclose(self.dual_var[1], 0)) *
            (1 - np.isclose(self.dual_var[1], self.C)))[0]

        return self

    def predict(self, X, T, X_new):
        """Predict ``X_new`` with given traning data ``(X, T)``."""
        eps = self.eps
        dual_var = self.dual_var
        union_sv_inices = self.union_sv_inices
        inner_sv_indices = self.inner_sv_indices

        K = self.kernel.inner(X[inner_sv_indices[0]], X[union_sv_inices])
        b = 1 / inner_sv_indices[0].size * (
            T[inner_sv_indices[0]].sum() - eps * inner_sv_indices[0].size -
            K.dot(dual_var[0][union_sv_inices] - dual_var[1][union_sv_inices]).sum())

        Y = (dual_var[0][union_sv_inices] - dual_var[1][union_sv_inices]).dot(
            self.kernel.inner(X[union_sv_inices], X_new)) + b

        return Y
