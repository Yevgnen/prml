#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as linalg
from matplotlib import pyplot as plt


class LinearLeastSquare(object):
    # FIXME
    def __init__(self,
                 basis_functions=[(lambda i: (lambda x: x**i))(i)
                                  for i in range(7)]):

        self.basis_functions = basis_functions
        self.n_basis = len(self.basis_functions)

        return

    def nonlinear_transformation(self, x):
        return np.array([phi(x) for phi in self.basis_functions]).T

    def fit(self, X, T):
        n_sample = X.shape[0]
        Phi = np.zeros((n_sample, self.n_basis))

        for i in range(n_sample):
            Phi[i, :] = self.nonlinear_transformation(X[i, :])

        self.w = linalg.solve(Phi.T.dot(Phi), Phi.T.dot(T))
        self.beta = 1 / np.mean((T - Phi.dot(self.w))**2)

        return self

    def predict(self, X):
        n_sample = X.shape[0]
        Phi = np.zeros((n_sample, self.n_basis))

        for i in range(n_sample):
            Phi[i, :] = self.nonlinear_transformation(X[i, :])
        y = self.w.dot(Phi.T).flatten()

        return y


class RegressionSample(object):
    def __init__(self, f, N, mu=0, sigma=1.0):
        ori = np.random.normal(size=1)
        x_min = ori - 1
        x_max = ori + 1
        self.f = f
        self.x = np.array([np.linspace(x_min, x_max, N)]).T
        self.t = self.f(self.x) + np.random.normal(mu, sigma, size=N)
        self.t = self.t.flatten()

        return


def main():

    N = 20
    f = lambda x: np.sin(2 * np.pi * x).flatten()
    sample = RegressionSample(f, N, sigma=0.2)
    x = sample.x
    t = sample.t

    x_min, x_max = x.min(), x.max()
    t_min, t_max = t.min(), t.max()

    x_test = np.array([np.linspace(x_min, x_max, 200)]).T
    t_test = f(x_test)

    lls = LinearLeastSquare()
    lls.fit(x, t)
    predictions = lls.predict(x_test)

    plt.figure()
    plt.scatter(x, t, marker='x')

    plt.plot(x_test[:, 0], t_test, 'r-')
    plt.plot(x_test[:, 0], predictions, 'g-')
    plt.plot(x_test[:, 0], predictions + 1 / lls.beta, 'g--')
    plt.plot(x_test[:, 0], predictions - 1 / lls.beta, 'g--')

    plt.show()


if __name__ == '__main__':
    main()
