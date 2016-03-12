#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy as sp
from matplotlib import pyplot as plt
from scipy import linalg as spla
from scipy import random as sprd

from mlpy.model import ProbabilisticPCA
from mlpy.util import loadMNISTImages, loadMNISTLabels

X = loadMNISTImages('../../mnist/train-images-idx3-ubyte')
T = loadMNISTLabels('../../mnist/train-labels-idx1-ubyte')

digit3_idx = sp.nonzero(T == 3)[0]
digit3_T = T[digit3_idx]
digit3_X = X[digit3_idx]

n_digit3, n_features = digit3_X.shape
n_sqrt_features = int(sp.sqrt(n_features))
n_components = 4

ppca = ProbabilisticPCA(M=n_components)
Y = ppca.fit(digit3_X)

cov = sp.cov(digit3_X, rowvar=0)
eigvals, eigvecs = ppca.eigvals, ppca.eigvecs
digit3_mean = digit3_X.mean(axis=0)

# Show the Figures 12.3 on Page 566 of the book
n_figrows, n_figcols = 2, n_components + 1
plt.figure(figsize=(10, 4))
plt.subplot(n_figrows, n_figcols, 1)
plt.imshow(
    digit3_mean.reshape((n_sqrt_features, n_sqrt_features)),
    cmap=plt.cm.Greys_r,
    interpolation='none')
plt.title('Mean')

for i in range(n_components):
    plt.subplot(n_figrows, n_figcols, i + 2)
    plt.imshow(
        digit3_mean.reshape((n_sqrt_features, n_sqrt_features)),
        cmap=plt.cm.Greys_r,
        interpolation='none')
    plt.imshow(eigvecs[:, i].reshape((n_sqrt_features, n_sqrt_features)),
               cmap=plt.cm.Greys_r,
               interpolation='none')
    plt.title('$\lambda$ = {0:3.1e}'.format(eigvals[i]))

# ProbabilisticPCA reconstruction
digit3 = digit3_X[0]
plt.subplot(n_figrows, n_figcols, n_figcols + 1)
plt.imshow(
    digit3.reshape((n_sqrt_features, n_sqrt_features)),
    cmap=plt.cm.Greys_r,
    interpolation='none')
plt.title('Origin')

M = [1, 10, 50, 250]
for i, n_components in enumerate(M):
    plt.subplot(n_figrows, n_figcols, n_figcols + 1 + i + 1)
    ppca = ProbabilisticPCA(M=n_components)
    ppca.fit(digit3_X)
    reconstructed = ppca.reconstruct(digit3)
    plt.imshow(
        reconstructed.reshape((n_sqrt_features, n_sqrt_features)),
        cmap=plt.cm.Greys_r,
        interpolation='none')
    plt.title('M = {0}'.format(n_components))

plt.show()
