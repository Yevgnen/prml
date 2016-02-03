# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import linalg as linalg


def softmax(a):
    a = np.exp(a - a.max())
    if (a.ndim == 1):
        return a / a.sum()
    else:
        return a / np.array([a.sum(axis=1)]).T


class SoftmaxRegression(object):
    def __init__(self, X, T, M, K):
        self.X = X
        self.T = T
        self.W = np.zeros((M, K))
        self.b = np.zeros(K)

    def train(self, tol=1e-5, max_iter=int(1e3), lr=1, eta=0.95):
        for i in range(max_iter):
            Y = softmax(self.X.dot(self.W) + self.b)
            err = Y - self.T

            self.W += lr * -self.X.T.dot(err)
            self.b += lr * -err.sum(axis=0)

            if (linalg.norm(self.W, 1) < tol):
                break

    def predict(self, x):
        return softmax(x.dot(self.W) + self.b)


class ClassificationSample(object):
    def __init__(self, N=100, K=2, mean=np.array([[1, -1], [-1, 1]]),
                 cov=np.array([[0.2, 0.1], [0.1, 0.2]])):
        n = int(N / K)
        X = np.zeros((N, 2))
        T = np.zeros((N, K), dtype=int)

        pos = 0
        for k in range(K):
            X[pos:pos + n, :] = np.random.multivariate_normal(mean[k, :], cov, n)
            T[pos:pos + n, k] = 1
            pos = pos + n

        p = np.random.permutation(N)
        self.X = X[p, :]
        self.T = T[p, :]


def main():
    N = 500
    K = 5
    mean = np.array([[5, 0], [0, 5], [-5, 0], [0, -5], [0, 0]])
    cov = np.array([[1, -1], [1, 1]])
    sample = ClassificationSample(N, K, mean, cov)
    X = sample.X
    T = sample.T

    M = 2
    classifier = SoftmaxRegression(X, T, M, K)
    classifier.train(tol=1e-5, max_iter=int(1e3), lr=1e-1, eta=0.95)

    x_min, y_min = X[:, 0].min() - 1, X[:, 1].min() - 1
    x_max, y_max = X[:, 0].max() + 1, X[:, 1].max() + 1

    plt.figure()
    color = np.array(['r', 'g', 'b', 'c', 'y'])
    cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#3380e6', '#00dcdc', '#fbf896'])

    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(), yy.ravel()]
    pred = classifier.predict(mesh).argmax(axis=1)
    pred = pred.reshape(xx.shape)
    plt.pcolormesh(xx, yy, pred, cmap=cmap)

    for n in range(N):
        plt.scatter(X[n, 0], X[n, 1], c=color[np.argmax(T[n, :])])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

if __name__ == '__main__':
    main()
