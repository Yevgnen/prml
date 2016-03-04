#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as linalg
from scipy.optimize import minimize as minimize

from kernel import GaussianKernel, LinearKernel


class SVM(object):
    def __init__(self, kernel='gaussian'):
        self.supported_kernel = {
            'gaussian': GaussianKernel(),
            'linear': LinearKernel()
        }
        self.kernel = self.supported_kernel[kernel]

    def fit(self, X, T, C=np.inf, max_iter=int(1e3), tol=1e-5):
        """Use training data ``X`` and ``T`` to fit a SVM models. """

        n_samples = X.shape[0]
        # Compute the Gram matrix of training data
        K = self.kernel.inner(X, X)

        # The target function: (1/2)*x'*Q*x + p'*x
        Q = T.reshape(1, -1) * K * T.reshape(-1, 1)
        p = -np.ones(n_samples)
        lagrange = lambda x: (0.5 * x.dot(Q).dot(x) + p.dot(x), Q.dot(x) + p)

        # The equality constraints: H(x) = 0
        A = T
        cons = ({'type': 'eq',
                 'fun': lambda x: A.dot(x),
                 'jac': lambda x: A})

        # The inequality constaints: 0 <= G(x) <= C
        bnds = [(0, C) for i in range(n_samples)]

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
        self.sv_indices = np.nonzero(1 - np.isclose(self.dual_var, 0))[0]

        return self

    def predict(self, X, T, X_new):
        """Predict ``X_new`` with given traning data ``(X, T)``."""

        n_tests = X_new.shape[0]
        Y = np.zeros(n_tests)

        dual_var = self.dual_var
        sv_indices = self.sv_indices

        sv_dual_var = dual_var[sv_indices]
        sv_X = X[sv_indices]
        sv_T = T[sv_indices]

        K = self.kernel.inner(sv_X, sv_X)
        # b = 1 / sv_indices.size * (sv_T.sum() - sv_dual_var.dot(K).dot(sv_T))
        b = 1 / sv_indices.size * (sv_T.sum() - K.dot(sv_dual_var * sv_T).sum())

        Y = (sv_T * sv_dual_var).dot(self.kernel.inner(sv_X, X_new)) + b
        YY = (sv_T * sv_dual_var).dot(self.kernel.inner(sv_X, X[self.sv_indices])) + b

        # import ipdb; ipdb.set_trace()
        Y[Y > 0] = 1
        Y[Y < 0] = -1

        return Y

    def score(self, X_train, T_train, X_test, T_test):
        Y = self.predict(X_train, T_train, X_test)

        return np.mean(np.isclose(Y, T_test))
