#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as linalg
from matplotlib import pyplot as plt

from mlpy.model import LinearLeastSquare
from mlpy.util import RegressionSample


def main():

    # Generate training data
    N = 10
    f = [lambda x: np.sin(2 * np.pi * x).flatten(),
         lambda x: np.cos(2 * np.pi * x).flatten()]

    sample = RegressionSample(f, N, sigma=0.2)
    x = sample.x
    t = sample.t

    # Generate test data
    x_min, x_max = x.min(), x.max()
    t_min, t_max = t.min(), t.max()
    x_test = np.array([np.linspace(x_min, x_max, 200)]).T
    t_test = np.array([fun(x_test) for fun in f]).T

    # Construct basis functions
    basis_functions = []
    locs = np.linspace(-1, 1, 10)
    for loc in locs:
        basis = (lambda loc: (lambda x: np.e**(-0.5*linalg.norm(x - loc)**2)))(loc)
        basis_functions.append(basis)

    # Model fitting
    lls = LinearLeastSquare(basis_functions=basis_functions)
    lls.fit(x, t)

    # Prediction
    predictions = lls.predict(x_test)
    std = np.sqrt(1 / lls.beta)

    plt.figure()
    fig_col = 2
    fig_row = np.ceil(len(f) / fig_col)
    for i, f in enumerate(f):

        plt.subplot(fig_row, fig_col, i + 1)
        plt.scatter(x, t[:, i], marker='x')

        plt.plot(x_test[:, 0], t_test[:, i], 'r-')
        plt.plot(x_test[:, 0], predictions[:, i], 'g-')
        plt.plot(x_test[:, 0], predictions[:, i] + std, 'g--')
        plt.plot(x_test[:, 0], predictions[:, i] - std, 'g--')
        plt.title('The ${0}$th component of output'.format(i))

    plt.show()


if __name__ == '__main__':
    main()
