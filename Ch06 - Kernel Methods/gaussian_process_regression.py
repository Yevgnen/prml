#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import scipy.linalg as linalg
from matplotlib import pyplot as plt

from kernel_models import GaussianProcess
from utils.regression_sample import RegressionSample


def main():

    # Generate training data
    N = 100
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
    gp = GaussianProcess()
    [pred_mean, pred_cov] = gp.predict(x, t, x_test)

    plt.figure()
    plt.scatter(x, t, marker='x')
    plt.plot(x_test[:, 0], t_test, 'r-')
    plt.plot(x_test[:, 0], pred_mean, 'g-')
    plt.plot(x_test[:, 0], pred_mean - pred_cov, 'g--')
    plt.plot(x_test[:, 0], pred_mean + pred_cov, 'g--')

    plt.show()


if __name__ == '__main__':
    main()
