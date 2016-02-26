#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from regression_sample import RegressionSample
from scipy.stats import norm as norm_dist
from scipy import linalg as linalg
from linear_model import EmpiricalBayes


def main():

    # Generate training data
    N = 25
    f = lambda x: np.sin(2 * np.pi * x).flatten()
    sigma = 0.2
    sample = RegressionSample(f, N, sigma=sigma)
    x = sample.x
    t = sample.t

    # Geenerate test data
    x_min, x_max = x.min(), x.max()
    t_min, t_max = t.min(), t.max()
    x_test = np.array([np.linspace(x_min, x_max, 200)]).T
    t_test = f(x_test)

    # Construct basis functions
    basis_functions = []
    locs = np.linspace(x.min(), x.max(), 9)
    for loc in locs:
        basis = (
            lambda loc: (lambda x: np.e**(-(linalg.norm(x - loc)**2) / (2 * 1e-2)))
        )(loc)
        basis_functions.append(basis)
    n_basis = len(basis_functions)

    # Model fitting
    blr = EmpiricalBayes(basis_functions=basis_functions)
    blr.fit(x, t)

    print('Prior precision: {0}'.format(blr.prior_precision))
    print('Std of prior: {0}'.format(np.sqrt(1 / blr.prior_precision)))
    print('Precision of noise: {0}'.format(blr.noise_precision))
    print('Std of noise: {0}'.format(np.sqrt(1 / blr.noise_precision)))

    # Prediction
    pred_mean, pred_cov = blr.predict(x_test)

    plt.figure()
    plt.scatter(x, t, marker='x')
    plt.plot(x_test[:, 0], t_test, 'r-')
    plt.plot(x_test[:, 0], pred_mean, 'g-')

    for i, basis in enumerate(basis_functions):
        plt.plot(x_test[:, 0], blr.mean[i] *
                 np.array([basis(x) for x in x_test[:, 0]]), 'm--')

    plt.show()


if __name__ == '__main__':
    main()
