import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


class SoftmaxRegression(object):
    def __init__(self, n_features, n_classes, max_iter=100):
        self.n_features = n_features + 1
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.weight = np.zeros(self.n_features * self.n_classes)

    def get_ground_true(self, T):
        n_samples = T.shape[0]
        n_classes = T.max() + 1
        ground_truth = np.zeros((n_classes, n_samples))
        ground_truth[T, np.arange(n_samples)] = 1

        return ground_truth

    def softmax(self, activation):
        activation = np.exp(activation - activation.max(axis=0))
        return activation / activation.sum(axis=0)

    def cost_func(self, weight, X, T):
        ground_truth = self.get_ground_true(T)
        weight = weight.reshape(self.n_classes, self.n_features)
        X = np.r_[np.array([np.ones(X.shape[1])]), X]

        activation = weight.dot(X)
        Y = self.softmax(activation)

        cost = -np.sum(ground_truth * np.log(Y))
        grad = -X.dot((ground_truth - Y).T)
        grad = grad.reshape(self.n_features * self.n_classes, order='F')

        return [cost, grad]

    def fit(self, X, T):
        opt_solution = scipy.optimize.minimize(self.cost_func,
                                               self.weight,
                                               args=(X, T),
                                               method='L-BFGS-B',
                                               jac=True,
                                               options={'maxiter': self.max_iter})
        print(opt_solution.success)
        print(opt_solution.fun)
        self.weight = opt_solution.x

        return self.weight

    def predict(self, X):
        X = np.r_[np.array([np.ones(X.shape[1])]), X]
        weight = self.weight.reshape(self.n_classes, self.n_features)

        return self.softmax(weight.dot(X)).argmax(axis=0)


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
        self.X = X[p, :].T
        self.T = T[p]


def main():
    N = 4000
    n_features = 2
    n_classes = 4
    K = n_classes
    mean = np.array([[5, 0], [0, 5], [-5, 8], [0, 0]])
    cov = np.array([[2, -1], [1, 1]])
    sample = ClassificationSample(N, K, mean, cov)
    X = sample.X
    T = sample.T
    classifier = SoftmaxRegression(n_features, n_classes)
    classifier.fit(X, T)

    x_min, y_min = X[0, :].min() - 1, X[1, :].min() - 1
    x_max, y_max = X[0, :].max() + 1, X[1, :].max() + 1
    color = np.array(['r', 'g', 'b', 'y'])
    cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#3380e6', '#fbf896'])

    plt.figure()
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(), yy.ravel()]
    predictions = classifier.predict(mesh.T)
    predictions = predictions.reshape(xx.shape)
    plt.pcolormesh(xx, yy, predictions, cmap=cmap)
    plt.scatter(X[0, :], X[1, :], marker='x', c=color[T])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

if __name__ == '__main__':
    main()
