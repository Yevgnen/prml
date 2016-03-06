#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy as sp
import scipy.linalg as spla

from .kernel import GaussianKernel, LinearKernel


class RVR(object):
    def __init__(self,
                 kernel='gaussian',
                 mean=None,
                 cov=None,
                 beta=None):
        """
        Build alpha relenvance vector machine mode.

        The kernel function is Gaussian default.
        The prior mean ``mean`` is alpha zero vector by default.
        The prior cov matrix ``precison`` should be given by alpha 1d numpy array.
        The cov of the distribution of target conditioning on the isput is given by ``beta``.
        """
        self.supported_kernel = {
            'gaussian': GaussianKernel(sigma=0.2),  # FIXED me!
            'linear': LinearKernel()
        }
        self.kernel = self.supported_kernel[kernel]
        self.rv_indices = None
        self.cov = cov
        self.mean = mean
        self.beta = beta

    def _init_hyperparameters(self, X, T):
        n_samples = X.shape[0]

        if (self.mean is None):
            self.mean = sp.zeros(n_samples + 1)

        if (self.cov is None):
            self.cov = sp.ones(n_samples + 1)

        if (self.beta is None):
            self.beta = 1

        return

    def _compute_design_matrix(self, X, X_new=None):
        if (X_new is None):
            X_new = X

        K = self.kernel.inner(X_new, X)

        return K

    def fit(self, X, T, max_iter=int(1e2), tol=1e-3, bound=1e10):
        """Fit a RVM model with the training data ``(X, T)``."""
        # Initialize the hyperparameters
        self._init_hyperparameters(X, T)

        # Compute design matrix
        n_samples = X.shape[0]
        phi = sp.c_[sp.ones(n_samples), self._compute_design_matrix(X)]  # Add x0

        alpha = self.cov
        beta = self.beta

        log_evidence = -1e10
        for iter in range(max_iter):
            alpha[alpha >= bound] = bound
            rv_indices = sp.nonzero(alpha < bound)[0]
            rv_phi = phi[:, rv_indices]
            rv_alpha = alpha[rv_indices]

            # Compute the posterior distribution
            post_cov = spla.inv(sp.diag(rv_alpha) + beta * sp.dot(rv_phi.T, rv_phi))
            post_mean = beta * sp.dot(post_cov, sp.dot(rv_phi.T, T))

            # Re-estimate the hyperparameters
            gamma = 1 - rv_alpha * post_cov.diagonal()
            rv_alpha = gamma / (post_mean * post_mean)
            beta = (n_samples + 1 - gamma.sum()) / spla.norm(T - sp.dot(rv_phi, post_mean))**2

            # Evalueate the log evidence and test the relative change
            C = sp.eye(rv_phi.shape[0]) / beta + rv_phi.dot(sp.diag(1.0 / rv_alpha)).dot(rv_phi.T)
            log_evidence_new = -0.5 * (sp.log(spla.det(C)) + T.dot(spla.inv(C)).dot((T)))
            diff = spla.norm(log_evidence_new - log_evidence)
            if (diff < tol * spla.norm(log_evidence)):
                break

            log_evidence = log_evidence_new
            alpha[rv_indices] = rv_alpha

        # Should re-compute the posterior distribution
        self.rv_indices = rv_indices
        self.cov = post_cov
        self.mean = post_mean
        self.beta = beta

        return self

    def predict(self, X, T, X_new):
        """Predict ``X_new`` with given traning data ``(X, T)``."""
        n_tests = X_new.shape[0]
        phi = sp.r_[sp.ones(n_tests).reshape(1, -1), self._compute_design_matrix(X_new, X)]  # Add x0
        phi = phi[self.rv_indices, :]

        predict_mean = sp.dot(self.mean, phi)
        predict_cov = 1 / self.beta + sp.dot(phi.T, sp.dot(self.cov, phi)).diagonal()

        return predict_mean, predict_cov

    def score(self, X_train, T_train, X_test, T_test):
        Y = self.predict(X_train, T_train, X_test)

        return sp.mean(sp.isclose(Y, T_test))


class RVC(object):
    def __init__(self):
        return
