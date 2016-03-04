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


def sample(N, label=np.array([1, 0]), mean1=np.array([-1, 1]), mean2=np.array([1, -1]),
           cov=np.array([[1, 0.5], [0.5, 1]])):
    M = int(N / 2)
    c1 = np.random.multivariate_normal(mean1, cov, M)
    c2 = np.random.multivariate_normal(mean2, cov, M)
    x = np.vstack([c1, c2]).T
    t = np.concatenate([np.ones(M) * label[0], np.ones(M) * label[1]])
    p = np.random.permutation(N)

    return x[:, p], t[p]


def main():
    N = 500
    x, t = sample(N, label=np.array([-1, 1]), cov=np.array([[0.2, 0.1], [0.1, 0.2]]))

    w = np.array([1.0, 1.0, 1.0])
    w = percetpron(x, t, w)

    x1 = np.linspace(x[0, :].min() - 1, x[0, :].max() + 1, 500)
    plt.figure()
    color = np.array(['b', 'r'])

    plt.scatter(x[0, :], x[1, :], c=color[[(lambda x: 1 if x > 0 else 0)(tn) for tn in t]])
    plt.plot(x1, -(w[0] + w[1] * x1) / w[2], 'g-', label='Decision boundary')
    plt.legend()
    plt.xlim(x[0, :].min(), x[0, :].max())
    plt.ylim(x[1, :].min(), x[1, :].max())
    plt.show()


if __name__ == '__main__':
    main()
