#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as linalg
from matplotlib import pyplot as plt

from mlpy.model import LinearLeastSquare
from mlpy.util import RegressionSample


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
    locs = np.linspace(-1, 1, 10)
    for loc in locs:
        basis = (lambda loc: (lambda x: np.e**(-0.5*linalg.norm(x - loc)**2)))(loc)
        basis_functions.append(basis)

    # Model fitting
    lls = LinearLeastSquare(basis_functions=basis_functions)
    lls.fit(x, t)

    # Phi = np.zeros((N, lls.n_basis))
    # for i in range(N):
    #     Phi[i, :] = lls.nonlinear_transformation(x[i, :])

    # print(lls.w[0] - t.mean() - lls.w[1:].dot(Phi.mean(axis=0)[1:]))  # (3.19)
    # Prediction
    predictions = lls.predict(x_test)
    std = np.sqrt(1 / lls.beta)

    plt.figure()
    plt.scatter(x, t, marker='x')
    plt.plot(x_test[:, 0], t_test, 'r-')
    plt.plot(x_test[:, 0], predictions, 'g-')
    plt.plot(x_test[:, 0], predictions + std, 'g--')
    plt.plot(x_test[:, 0], predictions - std, 'g--')

    plt.show()


if __name__ == '__main__':
    main()
