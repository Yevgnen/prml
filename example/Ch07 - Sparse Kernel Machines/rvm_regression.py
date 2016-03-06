#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy as sp
from matplotlib import pyplot as plt

from mlpy.model import RVR
from mlpy.util import RegressionSample

# Generate training data
n_samples = 25
f = lambda x: sp.sin(2 * sp.pi * x).flatten()
sigma = 0.2
sample = RegressionSample(f, n_samples, sigma=sigma, width=1)
x = sample.x
t = sample.t

# Geenerate test data
x_min, x_max = x.min(), x.max()
t_min, t_max = t.min(), t.max()
x_test = sp.array([sp.linspace(x_min, x_max, 200)]).T
t_test = f(x_test)

# Prediction
rvr = RVR()
rvr.fit(x, t)
rv_indices = rvr.rv_indices
print(rvr.mean)
print('n_basises: n_samples = {0}:{1}'.format(rv_indices.size, n_samples))
predict_mean, predict_cov = rvr.predict(x, t, x_test)
print(rvr.beta)
print(predict_cov)

rv_indices = rvr.rv_indices[rvr.rv_indices > 0] - 1
rvs = x[rv_indices]
rvst = t[rv_indices]

# Visualization
plt.figure()
plt.scatter(x, t, marker='x', zorder=10, label='Samples')
plt.scatter(rvs, rvst, marker='o', s=80, facecolors='none', zorder=15, label='Relavance vectors')
plt.plot(x_test[:, 0], t_test, 'r-', label='The curve')
plt.plot(x_test[:, 0], predict_mean, 'g-', label='Prediction')
plt.plot(x_test[:, 0], predict_mean + predict_cov, 'g-.', label='Prediction + sigma')
plt.plot(x_test[:, 0], predict_mean - predict_cov, 'g--', label='Prediction - sigma')
plt.legend()

plt.show()
