#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as linalg
from scipy.optimize import minimize as minimize

from cvxopt import matrix, solvers
from kernel import GaussianKernel


class SVM(object):
    def __init__(self):
        self.kernel = GaussianKernel()

    def fit(self, X, T, max_iter=int(1e3), tol=1e-5):
        """Use training data ``X`` and ``T`` to fit a SVM models. """

        n_samples = X.shape[0]
        K = self.kernel.inner(X, X)

        # The target function: (1/2)*x'*Q*x + p'*x
        Q = T.reshape(1, -1) * K * T.reshape(-1, 1)
        p = -np.ones(n_samples)
        # The inequality constrains: G(x) >= 0
        G = np.eye(n_samples)
        # The equality constrains: H(x) = 0
        A = T

        lagrange = lambda x: (0.5 * x.dot(Q).dot(x) + p.dot(x), Q.dot(x) + p)
        cons = ({'type': 'ineq',
                 'fun': lambda x: G.dot(x),
                 'jac': lambda x: G},
                {'type': 'eq',
                 'fun': lambda x: A.dot(x),
                 'jac': lambda x: A})

        # solve the quadratic program
        opt_solution = minimize(lagrange,
                                # a,
                                np.zeros(n_samples),
                                method='SLSQP',
                                constraints=cons,
                                tol=tol,
                                jac=True,
                                options={'maxiter': max_iter,
                                         'disp': True})

        self.dual_var = opt_solution.x
        self.support_vector_indices = np.nonzero(1 - np.isclose(self.dual_var, 0))[0]
        print(self.support_vector_indices.size)

        return self

    def predict(self, X, T, X_new):
        """Predict ``X_new`` with given traning data ``(X, T)``."""

        n_tests = X_new.shape[0]
        Y = np.zeros(n_tests)

        dual_var = self.dual_var
        S = self.support_vector_indices

        sub_dual_var = dual_var[S]
        sub_X = X[S]
        sub_T = T[S]

        K = self.kernel.inner(sub_X, sub_X)
        b = 1 / S.size * (sub_T.sum() - sub_dual_var.dot(K).dot(sub_T))

        Y = (sub_T * sub_dual_var).dot(self.kernel.inner(sub_X, X_new)) + b

        Y[Y > 0] = 1
        Y[Y < 0] = -1

        return Y

    def score(self, X_train, T_train, X_test, T_test):
        Y = self.predict(X_train, T_train, X_test)

        return np.mean(np.isclose(Y, T_test))
