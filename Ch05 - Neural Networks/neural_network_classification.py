import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

from neural_network import Layer, NeuralNetwork

# Generate dataset
n_samples = 400
sigma = 2e-1
X, T = datasets.make_moons(n_samples, noise=sigma)

half = int(n_samples / 2)
X_train = X[0:half, :]
T_train = T[0:half]
X_test = X[half:, :]
T_test = T[half:]

# Fit the neural network
max_iter = int(1e2)
tol = 1e-5

nn = NeuralNetwork(
    [Layer('Sigmoid', (2, 3)),  Layer('Sigmoid', (3, 1))], max_iter=max_iter, tol=tol)
nn.fit(X_train, T_train)

# Predcition
prediction = nn.predict(X_test)
score = nn.score(X_test, T_test)

print(score)

# Visualization
plt.scatter(X_train[:, 0],
            X_train[:, 1],
            s=40,
            c=T_train,
            cmap=plt.cm.Spectral)
# plt.show()
