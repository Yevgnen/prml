#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as linalg
from matplotlib import pyplot as plt


class LinearLeastSquare(object):
    # FIXME
    def __init__(self,
                 basis_functions=[(lambda i: (lambda x: x**i))(i)
                                  for i in range(5)]):

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

        # MLE for w 
        self.w = linalg.solve(Phi.T.dot(Phi), Phi.T.dot(T))

        # MLE for beta
        n_output = T.shape[1]
        self.beta = n_output / np.mean(linalg.norm(T - Phi.dot(self.w))**2)

        return self

    def predict(self, X):
        n_sample = X.shape[0]
        Phi = np.zeros((n_sample, self.n_basis))

        for i in range(n_sample):
            Phi[i, :] = self.nonlinear_transformation(X[i, :])

        if (self.w.ndim == 1):
            y = self.w.dot(Phi.T).flatten()
        else:
            y = Phi.dot(self.w) # self.w.T.dot(Phi.T).T

        return y


class RegressionSample(object):
    def __init__(self, f, N, mu=0, sigma=1.0):
        ori = np.random.normal(size=1)
        x_min = ori - 0.5
        x_max = ori + 0.5
        self.f = f
        self.x = np.array([np.linspace(x_min, x_max, N)]).T

        n_fun = 1 if len(np.shape(self.f)) == 0 else np.shape(self.f)[0]
        if (n_fun == 1):
            self.t = self.f(self.x) + np.random.normal(mu, sigma, size=N)
            self.t = self.t.flatten()
        else:
            self.t = np.zeros((N, n_fun))
            for i, fun in enumerate(f):
                self.t[:, i] = self.f[i](self.x)

            self.t += np.random.multivariate_normal(np.ones(n_fun) * mu, sigma * np.eye(n_fun), size=N)

        return


def main():

    N = 10
    f = [lambda x: np.sin(2 * np.pi * x).flatten(),
         lambda x: np.cos(2 * np.pi * x).flatten()]

    sample = RegressionSample(f, N, sigma=0.2)
    x = sample.x
    t = sample.t

    x_min, x_max = x.min(), x.max()
    t_min, t_max = t.min(), t.max()

    x_test = np.array([np.linspace(x_min, x_max, 200)]).T
    t_test = np.array([fun(x_test) for fun in f]).T

    lls = LinearLeastSquare()
    lls.fit(x, t)
    predictions = lls.predict(x_test)

    plt.figure()
    fig_col = 2
    fig_row = np.ceil(len(f) / fig_col)
    for i, f in enumerate(f):

        plt.subplot(fig_row, fig_col, i + 1)
        plt.scatter(x, t[:, i], marker='x')

        plt.plot(x_test[:, 0], t_test[:, i], 'r-')
        plt.plot(x_test[:, 0], predictions[:, i], 'g-')
        plt.plot(x_test[:, 0], predictions[:, i] + 1 / lls.beta, 'g--')
        plt.plot(x_test[:, 0], predictions[:, i] - 1 / lls.beta, 'g--')
        plt.title('The ${0}$th component of output'.format(i))

    plt.show()


if __name__ == '__main__':
    main()
