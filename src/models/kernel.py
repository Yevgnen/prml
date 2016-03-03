#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as linalg
from sklearn.metrics.pairwise import rbf_kernel


def to2d(x, axis=0):
    if (x.ndim < 2):
        return x.reshape(1, -1) if axis == 0 else x.reshape(-1, 1)
    else:
        return x

class GaussianKernel(object):
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def inner(self, x, y):
        gamma = 0.5 / self.sigma**2

        return rbf_kernel(to2d(x), to2d(y), gamma)


# class ExponentialQuadraticKernel(object):
#     def __init__(self, theta=(1.0, 5.0, 0, 5.0)):
#         self.theta = theta

#     def inner(self, x, y):
#         return self.theta[0] * np.exp(
#             -0.5 * self.sigma[1] * linalg.norm(x - y)**
#             2) + self.sigma[2] + self.sigma[3] * np.dot(x, y)
