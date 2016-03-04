#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def phi(x, M):
    return np.array([x ** j for j in np.arange(0, M + 1)])


def S(x_new, x, M, alpha, beta):
    return np.linalg.inv(S)


def mean(x_new, x, t, M, beta, S):
    phi_new = phi(x_new, M)
    return beta * phi_new.dot(S).dot(phi(x, M).dot(t))


def variance(x_new, M, beta, S):
    phi_new = phi(x_new, M)
    return 1 / beta + phi_new.dot(S).dot(phi_new)


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
    alpha = 5e-3
    beta = 11.1

    phi_x = phi(x, M)
    S = np.linalg.inv(alpha * np.eye(M + 1) + beta * phi_x.dot(phi_x.T))

    pred_mean = [mean(x_new, x, t, M, beta, S) for x_new in xs]
    pred_variance = [variance(x_new, M, beta, S) for x_new in xs]
    pred_sigma = np.sqrt(pred_variance)

    plt.figure(1)
    plt.plot(x, t, 'b.', label='Training data')
    plt.plot(xs, t_ideal, 'g-', label='$f(x) = \sin(2\pi x)$')
    plt.plot(xs, pred_mean, 'r-', label='Bayesian Curve fitting')
    plt.plot(xs, pred_mean - pred_sigma, 'r--')
    plt.plot(xs, pred_mean + pred_sigma, 'r--')
    plt.legend()
    plt.title('M = {0}'.format(M))
    plt.xlim(0, 1)
    plt.ylim(-1.5, 1.5)

    plt.show()


if __name__ == '__main__':
    main()
