#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class RegressionSample(object):
    def __init__(self, f, N, mu=0, sigma=1.0, width=0.5):
        order = np.random.permutation(N)
        ori = np.random.normal(size=1)

        x_min = ori - width
        x_max = ori + width
        self.f = f
        self.x = np.array([np.linspace(x_min, x_max, N)]).T
        self.x = self.x[order, :]

        n_fun = 1 if len(np.shape(self.f)) == 0 else np.shape(self.f)[0]
        if (n_fun == 1):
            self.t = self.f(self.x) + np.random.normal(mu, sigma, size=N)
            self.t = self.t.flatten()
        else:
            self.t = np.zeros((N, n_fun))
            for i, fun in enumerate(f):
                self.t[:, i] = self.f[i](self.x)

            self.t += np.random.multivariate_normal(np.ones(n_fun) * mu, sigma * np.eye(n_fun), size=N)

        return
