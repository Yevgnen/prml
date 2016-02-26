#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import reduce

import numpy as np
import scipy.linalg as linalg
from scipy.optimize import minimize as minimize
from scipy.special import expit as expit


def softmax(a):
    a = np.exp(a - a.max())
    if (a.ndim == 1):
        return a / a.sum()
    else:
        return a / np.array([a.sum(axis=1)]).T


def sigmoid(a):
    h = expit(a)

    return h, h * (1 - h)


def tanh(a):
    h = scipy.tanh(a)

    return h, 1 - h**2


def reg(a):
    h = a

    return h, np.ones_like(a)


class Layer(object):
    def __init__(self, type, size):
        self.type = type
        self.supported_activate_function = {
            'Sigmoid': sigmoid,
            'Tanh': tanh,
            'Softmax': softmax,
            'Regression': lambda x: x
        }
        self.n_input = size[0]
        self.n_output = size[1]
        self.weight = np.random.randn(self.n_output, self.n_input)
        self.bias = np.random.randn(self.n_output)

    def activate(self, input):
        activate = self.supported_activate_function[self.type]
        a = self.weight.dot(input) + np.c_[self.bias]
        h, dh = activate(a)

        return h, dh


class NeuralNetwork(object):
    def __init__(self, layers, lambd=1e-4, max_iter=int(1e2), tol=1e-5):
        self.layers = layers
        self.lambd = lambd
        self.max_iter = max_iter
        self.tol = tol

    def flatten(self):
        weight = reduce(lambda x, y: np.concatenate((x, y)),
                        [layer.weight.reshape(-1) for layer in self.layers])

        bias = reduce(lambda x, y: np.concatenate((x, y)),
                      [layer.bias.reshape(-1) for layer in self.layers])

        return np.concatenate((weight, bias))

    def stack(self, thetas):
        n_layers = len(self.layers)

        n_weights = reduce(lambda x, y: x + y, [layer.weight.size
                                                for layer in self.layers])
        weight = thetas[0:n_weights]
        bias = thetas[n_weights:]

        weight_start = 0
        bias_start = 0
        for i, layer in enumerate(self.layers):
            weight_end = weight_start + layer.weight.size
            layer.weight = np.array(weight[weight_start:weight_end]).reshape(
                layer.weight.shape)

            bias_end = bias_start + layer.n_output
            layer.bias = np.array(bias[bias_start:bias_end]).reshape(
                layer.bias.shape)

            weight_start = weight_end
            bias_start = bias_end

        return self

    def format(self, X, T=None):

        if (T is None):
            return X.T

        if (T.ndim > 1):
            T = T.T
        X = X.T

        return X, T

    def feed_forward(self, input):
        # column as a sample
        n_layers = len(self.layers)
        z = [None for i in range(n_layers)]
        dh = [None for i in range(n_layers)]

        for i, layer in enumerate(self.layers):
            z[i], dh[i] = layer.activate(input)
            input = z[i]

        return z, dh

    def compute_cost(self, thetas, X, T):

        # Each column of X is a sample, so is T
        X, T = self.format(X, T)

        # Make the params in matrix shape
        self.stack(thetas)

        # lambd = 0
        lambd = self.lambd
        layers = self.layers
        n_layers = len(layers)
        n_samples = X.shape[1]

        # Feed-feed_forward
        z, dh = self.feed_forward(X)

        # Cost function
        cost = 0.5 * np.sum((z[-1] - T)**2) + 0.5 * lambd * thetas.dot(thetas)

        # Initialize the gradients
        dW = [None for i in range(n_layers)]
        db = [None for i in range(n_layers)]

        for i, layer in enumerate(layers):
            dW[i] = np.zeros_like(layer.weight)
            db[i] = np.zeros_like(layer.bias)

        # Error Backproppagation
        delta = [np.zeros_like(zi) for zi in z]
        for i, layer in reversed(list(enumerate(layers))):
            if (i == n_layers - 1):
                delta[i] = dh[i] * (z[i] - T)
            else:
                delta[i] = dh[i] * layers[i + 1].weight.T.dot(delta[i + 1])

            if (i == 0):
                dW[i] = delta[i].dot(X.T) + lambd * layers[i].weight
                db[i] = delta[i].sum(axis=1)
            else:
                dW[i] = delta[i].dot(z[i - 1].T) + lambd * layers[i].weight
                db[i] = delta[i].sum(axis=1)

        # Flatten the gradients
        dW = np.concatenate([grad.ravel() for grad in dW])
        db = np.concatenate([grad.ravel() for grad in db])

        grad = np.concatenate((dW, db))

        return cost, grad

    def numerical_gradient(self, thetas, X, T, eps=1e-4):
        n_params = thetas.size
        I = np.eye(n_params)

        grad = np.zeros_like(thetas)

        for i in range(n_params):
            delta = I[:, i] * eps
            cost2 = self.compute_cost(thetas + delta, X, T)[0]
            cost1 = self.compute_cost(thetas - delta, X, T)[0]
            grad[i] = (cost2 - cost1) / (2.0 * eps)

        return grad

    def check_gradient(self, X, T, eps=1e-10):
        thetas = self.flatten()

        grad1 = self.numerical_gradient(thetas, X, T)
        _, grad2 = self.compute_cost(thetas, X, T)

        diff = linalg.norm(grad1 - grad2) / linalg.norm(grad1 + grad2)
        print(np.c_[grad1, grad2, np.abs(grad1 - grad2)])
        print('diff = {0}'.format(diff))

        return diff < eps

    def fit(self, X, T):
        print(self.max_iter)

        opt_solution = minimize(self.compute_cost,
                                self.flatten(),
                                args=(X, T),
                                method='L-BFGS-B',
                                tol=self.tol,
                                jac=True,
                                options={'maxiter': self.max_iter,
                                         'disp': True})
        print(opt_solution.success)
        print(opt_solution.fun)

        self.stack(opt_solution.x)

        return self

    def predict(self, X):
        X = self.format(X)

        if (self.layers[-1].type == 'Sigmoid'):
            Y, _ = self.feed_forward(X)
            prediction = Y[-1] > 0.5
            prediction = prediction.astype(np.int, copy=False)

        return prediction

    def score(self, X, T):
        Y = self.predict(X)

        return np.mean(Y == T)
