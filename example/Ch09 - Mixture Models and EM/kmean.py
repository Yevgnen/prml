#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy as sp
from matplotlib import pyplot as plt

from mlpy.model import Kmeans
from mlpy.util import ClassificationSample

n_samples = 300
n_classes = 3
mean = np.array([[-1, 0], [1, 0], [0, sp.sqrt(2)]])
cov = np.array([[0.5, 0], [0, 0.5]])
sample = ClassificationSample(n_samples, n_classes, mean)
X, T = sample.X, sample.T

km = Kmeans(K=n_classes)
km.fit(X)

center = km.center

plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='x', color=T, cmap=plt.cm.Paired)
plt.scatter(center[:, 0], center[:, 1], marker='*', color='m', s=500, facecolors='none')

plt.show()
