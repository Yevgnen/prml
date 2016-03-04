#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as linalg
from scipy.special import expit


def phi(x):
    return x


def logistic_regression(x, t, w, eps=1e-2, max_iter=int(1e3)):
    N = x.shape[1]
    Phi = np.vstack([np.ones(N), phi(x)]).T

    for k in range(max_iter):
        y = expit(Phi.dot(w))
        R = np.diag(np.ones(N) * (y * (1 - y)))
        H = Phi.T.dot(R).dot(Phi)
        g = Phi.T.dot(y - t)

        w_new = w - linalg.solve(H, g)

        diff = linalg.norm(w_new - w) / linalg.norm(w)
        if (diff < eps):
            break

        w = w_new
        print('{0:5d} {1:10.6f}'.format(k, diff))

    return w


def sample(N, label=np.array([1, 0]), mean1=np.array([-1, 1]), mean2=np.array([1, -1]),
           cov=np.array([[0.5, 0.3], [0.3, 0.5]])):
    M = int(N / 2)
    c1 = np.random.multivariate_normal(mean1, cov, M)
    c2 = np.random.multivariate_normal(mean2, cov, M)
    x = np.vstack([c1, c2]).T
    t = np.concatenate([np.ones(M) * label[0], np.ones(M) * label[1]])
    p = np.random.permutation(N)

    return x[:, p], t[p]


def main():
    N = 500
    x, t = sample(N, cov=np.array([[0.8, 0.2], [0.2, 0.8]]))

    w = np.array([1.0, 1.0, 1.0])
    w = logistic_regression(x, t, w)

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
