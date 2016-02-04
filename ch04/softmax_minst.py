# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import struct
import array
import numpy as np
import scipy.optimize as optimize


def loadMNISTImages(file_name):
    image_file = open(file_name, 'rb')
    head1 = image_file.read(4)
    head2 = image_file.read(4)
    head3 = image_file.read(4)
    head4 = image_file.read(4)

    num_examples = struct.unpack('>I', head2)[0]
    num_rows = struct.unpack('>I', head3)[0]
    num_cols = struct.unpack('>I', head4)[0]
    dataset = np.zeros((num_rows * num_cols, num_examples))
    images_raw = array.array('B', image_file.read())
    image_file.close()

    for i in range(num_examples):
        limit1 = num_rows * num_cols * i
        limit2 = num_rows * num_cols * (i + 1)
        dataset[:, i] = images_raw[limit1:limit2]

    return dataset / 255


def loadMNISTLabels(file_name):
    label_file = open(file_name, 'rb')
    head1 = label_file.read(4)
    head2 = label_file.read(4)

    num_examples = struct.unpack('>I', head2)[0]
    labels = np.zeros(num_examples, dtype=np.int)
    labels_raw = array.array('b', label_file.read())
    label_file.close()

    labels[:] = labels_raw[:]

    return labels


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
        opt_solution = optimize.minimize(
            self.cost_func,
            self.weight,
            args=(X, T),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': self.max_iter})
        self.weight = opt_solution.x

        return self.weight

    def predict(self, X):
        X = np.r_[np.array([np.ones(X.shape[1])]), X]
        weight = self.weight.reshape(self.n_classes, self.n_features)

        return self.softmax(weight.dot(X)).argmax(axis=0)


def main():
    X = loadMNISTImages("../minst/t10k-images-idx3-ubyte")
    T = loadMNISTLabels("../minst/t10k-labels-idx1-ubyte")

    n_features = X.shape[0]
    n_classes = T.max() + 1
    classifier = SoftmaxRegression(n_features, n_classes)
    classifier.fit(X, T)

    X_test = loadMNISTImages("../minst/train-images-idx3-ubyte")
    T_test = loadMNISTLabels("../minst/train-labels-idx1-ubyte")

    predictions = classifier.predict(X_test)

    correct = np.array(predictions == T_test)
    print("ACCURACY: {0:5f}".format(correct.mean()))


if __name__ == '__main__':
    main()
