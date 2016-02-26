#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def y(x, w):
    x = [x ** j for j in np.arange(0, w.size)]
    wTx = w[:, np.newaxis].T.dot(x).T
    return wTx[:, 0]


def regularized_least_squares(M, x, t, l):
    x = np.array([x ** j for j in np.arange(0, M + 1)])
    return np.linalg.solve(x.dot(x.T) + l * np.eye(t.size), x.dot(t))


def main():
    # Training data
    noise_mu = 0
    noise_sigma = 0.2
    N = 10
    x = np.linspace(0, 1, N)
    t = np.sin(2 * np.pi * x) + np.random.normal(noise_mu, noise_sigma, x.size)

    # True curve
    xs = np.linspace(0, 1, 500)
    t_ideal = np.sin(2 * np.pi * xs)

    M = 9
    lamb = [np.e ** p for p in [-np.inf, -18]]
    fig_row = 1
    fig_col = 2
    fig = plt.figure(1)
    for i, l in enumerate(lamb):
        w = regularized_least_squares(M, x, t, l)

        fig.add_subplot(fig_row, fig_col, i + 1)
        plt.plot(x, t, 'b.', label='Training data')
        plt.plot(xs, t_ideal, 'g-', label='$f(x) = \sin(2\pi x)$')
        plt.plot(xs, y(xs, w), 'r-', label='Regularized LS')
        plt.legend()
        plt.title('$\lambda$ = {0}'.format(l))
        plt.xlim(0, 1)
        plt.ylim(-1.5, 1.5)

    # Compute RMS for different \lambda
    n_lamb = 100
    p = np.linspace(-36, 0, n_lamb)
    lamb = [np.e ** pi for pi in p]
    rms_training = np.zeros((n_lamb, 1))
    rms_test = np.zeros((n_lamb, 1))
    fig = plt.figure(2)
    for i, l in enumerate(lamb):
        w = regularized_least_squares(M, x, t, l)
        rms_training[i] = np.sqrt(np.sum((y(x, w) - t) ** 2) / N)
        rms_test[i] = np.sqrt(np.sum((y(xs, w) - t_ideal) ** 2) / xs.size)

    plt.title('RMS of different regularization parameters')
    plt.plot(p, rms_training, label='Training')
    plt.plot(p, rms_test, label='Test')
    plt.xlabel('$\ln \lambda$')
    plt.ylabel('$E_{\mathrm{RMS}}$')
    plt.xlim(p[0], p[-1])
    plt.ylim(0, 1)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
