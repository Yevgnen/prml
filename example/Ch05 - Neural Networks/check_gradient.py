import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

from mlpy.model import Layer, NeuralNetwork

# Generate dataset
n_samples = 400
sigma = 2e-1
X, T = datasets.make_moons(n_samples, noise=sigma)

# Fit the neural network
max_iter = int(1e2)
tol = 1e-5

nn = NeuralNetwork(
    [Layer('Sigmoid', (2, 3)),
     Layer('Sigmoid', (3, 4)),
     Layer('Sigmoid', (4, 5)),
     Layer('Sigmoid', (5, 1))
    ], max_iter=max_iter, tol=tol)

nn.check_gradient(X, T)
