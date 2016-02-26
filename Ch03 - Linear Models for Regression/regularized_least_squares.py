#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as linalg
from matplotlib import pyplot as plt
from regression_sample import RegressionSample
from linear_model import LinearLeastSquare


def main():

    # Generate training data
    N = 10
    f = lambda x: np.sin(2 * np.pi * x).flatten()
    sample = RegressionSample(f, N, sigma=0.2)
    x = sample.x
    t = sample.t

    # Generate test data
    x_min, x_max = x.min(), x.max()
    t_min, t_max = t.min(), t.max()
    x_test = np.array([np.linspace(x_min, x_max, 200)]).T
    t_test = f(x_test)

    # Construct basis functions
    basis_functions = []
    locs = np.linspace(-1, 1, 100)
    for loc in locs:
        basis = (lambda loc: (lambda x: np.e**(-0.5*linalg.norm(x - loc)**2)))(loc)
        basis_functions.append(basis)

    plt.figure()
    lambs = [1e-8, 1e-6, 1e-4, 1e-2, 0]
    fig_col = 3
    fig_row = np.ceil(len(lambs) / fig_col)
    for i, lamb in enumerate(lambs):
        # Model fitting
        lls = LinearLeastSquare(basis_functions=basis_functions, lamb=lamb)
        lls.fit(x, t)

        # Prediction
        predictions = lls.predict(x_test)
        std = np.sqrt(1 / lls.beta)

        plt.subplot(fig_row, fig_col, i + 1)
        plt.scatter(x, t, marker='x')
        plt.plot(x_test[:, 0], t_test, 'r-')
        plt.plot(x_test[:, 0], predictions, 'g-')
        plt.plot(x_test[:, 0], predictions + std, 'g--')
        plt.plot(x_test[:, 0], predictions - std, 'g--')
        plt.title('$\lambda$ = {0}'.format(lamb))

    plt.show()


if __name__ == '__main__':
    main()
