#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel


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

class LinearKernel(object):
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def inner(self, x, y):
        return linear_kernel(to2d(x), to2d(y))

class PolynomialKernel(object):
    def __init__(self, degree=3, coef0=1):
        self.degree = degree
        self.coef0 = coef0

    def inner(self, x, y):
        return polynomial_kernel(to2d(x), to2d(y), degree=self.degree, coef0=self.coef0)

# class ExponentialQuadraticKernel(object):
#     def __init__(self, theta=(1.0, 5.0, 0, 5.0)):
#         self.theta = theta

#     def inner(self, x, y):
#         return self.theta[0] * np.exp(
#             -0.5 * self.sigma[1] * linalg.norm(x - y)**
#             2) + self.sigma[2] + self.sigma[3] * np.dot(x, y)
