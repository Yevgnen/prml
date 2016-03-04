#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

from mlpy.model import SVM

# Generate dataset
n_samples = 100
sigma = 2e-1
X, T = datasets.make_moons(n_samples, noise=sigma)
T[T == 0] = -1

half = int(n_samples / 2)
X_train = X[0:half, :]
T_train = T[0:half]
X_test = X[half:, :]
T_test = T[half:]

# Fit the neural network
classifier = SVM(kernel='gaussian')
classifier.fit(X_train, T_train, C=np.inf)

# Predcition
prediction = classifier.predict(X_train, T_train, X_test)
score = classifier.score(X_train, T_train, X_test, T_test)
print(score)

# Visualization
plt.scatter(X_train[:, 0],
            X_train[:, 1],
            marker='.',
            s=40,
            zorder=10,
            c=T_train,
            cmap=plt.cm.Paired)

plt.scatter(X_test[:, 0],
            X_test[:, 1],
            marker='x',
            s=40,
            zorder=10,
            c=T_test,
            cmap=plt.cm.Paired)

svs = X_train[classifier.sv_indices]
svst = T_train[classifier.sv_indices]

plt.scatter(svs[:, 0],
            svs[:, 1],
            marker='o',
            facecolors='none',
            s=80,
            zorder=5,
            c=svst,
            cmap=plt.cm.Paired)

h = 0.01
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = classifier.predict(X_train, T_train, np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

plt.show()
