#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as linalg
from matplotlib import pyplot as plt

from mlpy.model import SVR
from mlpy.util import RegressionSample

# Generate training data
N = 40
f = lambda x: np.sin(2 * np.pi * x).flatten()
sigma = 0.2
sample = RegressionSample(f, N, sigma=sigma, width=1)
x = sample.x
t = sample.t

# Geenerate test data
x_min, x_max = x.min(), x.max()
t_min, t_max = t.min(), t.max()
x_test = np.array([np.linspace(x_min, x_max, 200)]).T
t_test = f(x_test)

# Prediction
svr = SVR()
svr.fit(x, t)
prediction = svr.predict(x, t, x_test)

# Visualization
plt.figure()
plt.scatter(x, t, marker='x', zorder=10)
plt.plot(x_test[:, 0], t_test, 'r-')
plt.plot(x_test[:, 0], prediction, 'g-')
plt.plot(x_test[:, 0], prediction + svr.eps, 'g--')
plt.plot(x_test[:, 0], prediction - svr.eps, 'g--')

svs = x[svr.union_sv_inices]
svst = t[svr.union_sv_inices]

plt.scatter(svs,
            svst,
            marker='o',
            facecolors='none',
            s=80,
            zorder=5)

plt.show()
