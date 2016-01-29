#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def y(x, w):
    return w.dot([x ** j for j in np.arange(0, w.size)])


def least_squares(M, x, t):
    x = np.array([x ** j for j in np.arange(0, M + 1)])
    return np.linalg.solve(x.dot(x.T), x.dot(t))


def main():
    noise_mu = 0
    noise_sigma = 0.2
    x = np.linspace(0, 1, 10)
    t = np.sin(2 * np.pi * x) + np.random.normal(noise_mu, noise_sigma, x.size)

    xs = np.linspace(0, 1, 500)
    t_ideal = np.sin(2 * np.pi * xs)

    M = [0, 1, 3, 9]
    fig_row = 2
    fig_col = np.ceil(len(M) / fig_row)
    fig = plt.figure(figsize=(12, 8))
    for i, m in enumerate(M):
        w = least_squares(m, x, t)
        fig.add_subplot(fig_row, fig_col, i + 1)
        plt.plot(x, t, 'b.', label="Training data")
        plt.plot(xs, t_ideal, 'g-', label="$f(x) = \sin(2\pi x)$")
        plt.plot(xs, [y(x, w) for x in xs], 'r-',
                 label="Polynomial Curve fitting")
        plt.legend()
        plt.title("M = {0}".format(m))
    plt.savefig("ls.png", dpi=160)
    plt.show()


if __name__ == '__main__':
    main()
