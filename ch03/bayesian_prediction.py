#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from regression_sample import RegressionSample
from scipy.stats import norm as norm_dist
from scipy import linalg as linalg
from linear_model import BayesianLinearRegression


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
    alpha = 2.0
    prior_cov = np.eye(n_basis) / alpha
    prior_mean = np.zeros(n_basis)
    blr = BayesianLinearRegression(basis_functions=basis_functions,
                                   mean=prior_mean,
                                   cov=prior_cov)

    # Sequential learning
    beta = (1 / sigma)**2
    subplot_list = [0, 1, 3, 24]
    fig_row = 2
    fig_col = np.ceil(len(subplot_list) / fig_row)
    fig_no = 0
    plt.figure()
    for i in range(N):
        blr.fit(x[i, :], t[i], beta)
        if (all(np.in1d(i, subplot_list))):
            # Prediction
            pred_mean, pred_cov = blr.predict(x_test, beta)

            plt.subplot(fig_row, fig_col, fig_no + 1)
            plt.scatter(x[0:subplot_list[fig_no] + 1, :],
                        t[0:subplot_list[fig_no] + 1],
                        marker='x',
                        zorder=20)
            plt.plot(x_test[:, 0], t_test, 'r-', zorder=10)
            plt.plot(x_test[:, 0], pred_mean, 'g-', zorder=10)

            color = np.array(['w', 'r'])
            cmap = ListedColormap(['#FFFFFF', '#FFAAAA'])
            h = 0.03
            xx, yy = np.meshgrid(
                np.arange(x_min, x_max, h), np.arange(t_min, t_max, h))
            mesh = np.c_[xx.ravel(), yy.ravel()]
            mesh_pred_mean, mesh_pred_cov = blr.predict(np.c_[mesh[:, 0]],
                                                        beta)
            mesh_pred_std = np.sqrt(mesh_pred_cov)
            upper = mesh_pred_mean + mesh_pred_std
            lower = mesh_pred_mean - mesh_pred_std
            inbound = np.array(
                (mesh[:, 1] > lower) * (mesh[:, 1] < upper),
                dtype=int).reshape(xx.shape)
            plt.pcolormesh(xx, yy, inbound, cmap=cmap)

            fig_no += 1

    plt.show()


if __name__ == '__main__':
    main()
