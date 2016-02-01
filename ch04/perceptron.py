#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt


def phi(x):
    return x


def percetpron(x, t, w):
    N = x.shape[1]
    x = np.vstack([np.ones(N), phi(x)])
    correct = 0
    while (correct < N):
        for i in range(N):
            xn = x[:, i]
            tn = t[i]
            if (w.T.dot(xn) * tn < 0):
                w = w + xn * tn
                correct = 0
                break
            else:
                correct += 1
    return w


def sample(N):
    w = np.zeros(3)
    x = np.random.normal(size=(2, N))
    x1 = x[0, :]
    x2 = x[1, :]
    [x1_min, x1_max] = [x1.min(), x1.max()]
    [x2_min, x2_max] = [x2.min(), x2.max()]

    w[2] = -1
    w[1] = (x2_max - x2_min) / (x1_max - x1_min)
    w[0] = x1_min - w[1] * x1_min + np.random.randn()

    t = np.ones(N)
    t[np.vstack([np.ones(N), x]).T.dot(w) < 0] = -1
    return x, t, w


def main():
    N = 500
    x, t, w_true = sample(N)

    w = np.array([1.0, 1.0, 1.0])
    w = percetpron(x, t, w)

    x1 = np.linspace(x[0, :].min() - 1, x[0, :].max() + 1, 500)
    plt.figure()
    color = np.array(['b', 'r'])

    plt.scatter(x[0, :], x[1, :], c=color[[(lambda x: 1 if x > 0 else 0)(tn) for tn in t]])
    plt.plot(x1, -(w_true[0] + w_true[1] * x1) / w_true[2], "r-", label="True boundary")
    plt.plot(x1, -(w[0] + w[1] * x1) / w[2], "g--", label="Decision boundary")
    plt.legend()
    plt.xlim(x[0, :].min(), x[0, :].max())
    plt.ylim(x[1, :].min(), x[1, :].max())
    plt.show()


if __name__ == '__main__':
    main()
