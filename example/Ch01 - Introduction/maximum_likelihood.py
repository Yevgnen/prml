#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt


def y(x, w):
    x = [x ** j for j in np.arange(0, w.size)]
    wTx = w[:, np.newaxis].T.dot(x).T
    return wTx[:, 0]


def least_squares(M, x, t):
    x = np.array([x ** j for j in np.arange(0, M + 1)])
    return np.linalg.solve(x.dot(x.T), x.dot(t))


def maximum_likelihood(m, x, t):
    w = least_squares(m, x, t)
    beta_inv = np.sum((y(x, w) - t) ** 2) / x.size
    return w, beta_inv


def main():
    noise_mu = 0
    noise_sigma = 0.2
    x = np.linspace(0, 1, 10)
    t = np.sin(2 * np.pi * x) + np.random.normal(noise_mu, noise_sigma, x.size)

    xs = np.linspace(0, 1, 500)
    t_ideal = np.sin(2 * np.pi * xs)

    M = [1, 3, 6, 9]
    fig_row = 2
    fig_col = np.ceil(len(M) / fig_row)
    fig = plt.figure()
    for i, m in enumerate(M):
        w, beta_inv = maximum_likelihood(m, x, t)
        pred_mean = y(xs, w)

        fig.add_subplot(fig_row, fig_col, i + 1)
        plt.plot(x, t, 'b.', label='Training data')
        plt.plot(xs, t_ideal, 'g-', label='$f(x) = \sin(2\pi x)$')

        plt.plot(xs, pred_mean, 'r-', label='Maximum likelihood')
        sigma = np.sqrt(beta_inv)
        plt.plot(xs, pred_mean - sigma, 'r--')
        plt.plot(xs, pred_mean + sigma, 'r--')
        plt.legend()
        plt.title('M = {0}'.format(m))
        plt.xlim(0, 1)
        plt.ylim(-1.5, 1.5)

    plt.show()


if __name__ == '__main__':
    main()
