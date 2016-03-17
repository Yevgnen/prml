#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy as sp
from matplotlib import pyplot as plt
from scipy import linalg as spla
from scipy import random as sprd
from sklearn import datasets as datasets

from mlpy.model import PCA, KernelPCA, ProbabilisticPCA

n_samples = 400
sigma = 5e-2
X, T = datasets.make_circles(n_samples, noise=sigma, factor=0.5)

n_components = 2

plt.figure(figsize=(8, 6))
n_rows, n_cols = 2, 2
marker = 'x'
color = T
cmap = plt.cm.Paired

plt.subplot(n_rows, n_cols, 1)
plt.scatter(X[:, 0], X[:, 1], marker=marker, color=T, cmap=cmap)
plt.title('Origin')

pca = PCA(n_components)
Y_pca = pca.fit(X)
plt.subplot(n_rows, n_cols, 2)
plt.scatter(Y_pca[:, 0], Y_pca[:, 1], marker=marker, color=T, cmap=cmap)
plt.title('PCA')

sigma_gaussian = 0.2
kpca = KernelPCA(n_components, kernel='gaussian', sigma=sigma_gaussian)
Y_kpca = kpca.fit(X)
plt.subplot(n_rows, n_cols, 3)
plt.scatter(Y_kpca[:, 0], Y_kpca[:, 1], marker=marker, color=T, cmap=cmap)
plt.title('KPCA(Gaussian): $\sigma$={0}'.format(sigma_gaussian))

degree = 9
coef0 = 1
kpca = KernelPCA(n_components, kernel='polynomial', degree=degree, coef0=coef0)
Y_kpca = kpca.fit(X)
plt.subplot(n_rows, n_cols, 4)
plt.scatter(Y_kpca[:, 0], Y_kpca[:, 1], marker=marker, color=T, cmap=cmap)
plt.title('KPCA(Polynomial): M={0}, c={1}'.format(degree, coef0))

plt.show()

