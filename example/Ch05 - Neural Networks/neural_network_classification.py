import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

from mlpy.model import Layer, NeuralNetwork

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
max_iter = int(1e3)
tol = 1e-5

nn = NeuralNetwork(
    [Layer('Tanh', (2, 5)),
     Layer('Tanh', (5, 1))], max_iter=max_iter, tol=tol)
nn.fit(X_train, T_train)

# Predcition
prediction = nn.predict(X_test)
score = nn.score(X_test, T_test)

print(score)

# Visualization
plt.scatter(X_train[:, 0],
            X_train[:, 1],
            s=40,
            zorder=10,
            c=T_train,
            cmap=plt.cm.Paired)

h = 0.01
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.show()
