import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

from svm import SVM


# Generate dataset
# n_samples = 40
# sigma = 2e-2
# X, T = datasets.make_moons(n_samples, noise=sigma)

class ClassificationSample(object):
    def __init__(self,
                 N=100,
                 K=2,
                 mean=np.array([[1, -1], [-1, 1]]),
                 cov=np.array([[0.2, 0.1], [0.1, 0.2]])):
        n = int(N / K)
        X = np.zeros((N, 2))
        T = np.zeros((N), dtype=int)
        label = np.arange(K)
        pos = 0
        for k in range(K):
            X[pos:pos + n, :] = np.random.multivariate_normal(mean[k, :], cov, n)
            T[pos:pos + n] = label[k]
            pos = pos + n
        p = np.random.permutation(N)
        self.X = X[p, :]
        self.T = T[p]

N = 100
n_features = 2
n_classes = 2
K = n_classes
mean = np.array([[5, 0], [0, 5]])
cov = np.array([[2, -1], [1, 1]])
sample = ClassificationSample(N, K, mean, cov)
X = sample.X
T = sample.T
T[T==0] = -1
n_samples = N

half = int(n_samples / 2)
X_train = X[0:half, :]
T_train = T[0:half]
X_test = X[half:, :]
T_test = T[half:]

# Fit the neural network
max_iter = int(1e3)
tol = 1e-5

classifier = SVM()
classifier.fit(X_train, T_train)

# Predcition
prediction = classifier.predict(X_train, T_train, X_test)
score = classifier.score(X_train, T_train, X_test, T_test)
print(score)

# Visualization
plt.scatter(X_train[:, 0],
            X_train[:, 1],
            marker='.',
            s=40,
            zorder=10,
            c=T_train,
            cmap=plt.cm.Paired)

plt.scatter(X_test[:, 0],
            X_test[:, 1],
            marker='x',
            s=40,
            zorder=10,
            c=T_test,
            cmap=plt.cm.Paired)

svs = X_train[classifier.support_vector_indices]
svst = T_train[classifier.support_vector_indices]

plt.scatter(svs[:, 0],
            svs[:, 1],
            marker='o',
            facecolors='none',
            s=80,
            zorder=5,
            c=svst,
            cmap=plt.cm.Paired)

h = 0.01
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = classifier.predict(X_train, T_train, np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

plt.show()
