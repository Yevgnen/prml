#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy as sp
from matplotlib import pyplot as plt

from mlpy.model import GMM
from mlpy.util import ClassificationSample

# Generate sample
n_samples = 600
n_classes = 3
mean = sp.array([[-1, 0], [1, 0], [0, sp.sqrt(2)]])
cov = sp.array([[0.2, 0.1], [-0.2, 0.2]])
sample = ClassificationSample(n_samples, n_classes, mean=mean, cov=cov)
X, T = sample.X, sample.T

# Fit the GMM
gmm = GMM(K=n_classes, init='kmeans')
gmm.fit(X)

# Visualization
color = np.array(['r', 'g', 'b', 'y'])
plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', color=color[T], facecolors='none', cmap=plt.cm.Paired, zorder=5)

h = 0.01
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = sp.meshgrid(sp.arange(x_min, x_max, h), sp.arange(y_min, y_max, h))
Z = gmm.pdf(sp.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Purples, zorder=0)

plt.show()
