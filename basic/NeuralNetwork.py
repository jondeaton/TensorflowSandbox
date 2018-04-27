#!/usr/bin/env python
"""
File: NeuralNetwork
Date: 4/14/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import numpy as np
from activations import *
import math
from collections import namedtuple
from enum import Enum


def is_power(num, base):
    """
    Determines if a number is a power of some base

    :param num: The number to test
    :param base: The base of exponentiation
    :return: True if there exists some k for which base^k = num, false otherwise
    """
    if base == 1 and num != 1: return False
    if base == 1 and num == 1: return True
    if base == 0 and num != 1: return False
    power = int (math.log(num, base) + 0.5)
    return base ** power == num


def random_mini_batches(X, Y, mini_batch_size=64, seed=0, axis=1):
    """
    Creates a list of random mini-batches from (X, Y)

    :param X: input data, of shape (input size, number of examples)
    :param Y: true "label" vector of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    :param mini_batch_size: size of the mini-batches, integer
    :param seed: Random seed
    :return: list of synchronous (mini_batch_X, mini_batch_Y)
    """

    if axis == 0:
        X = X.T
        Y = Y.T

    m = X.shape[1]
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = m // mini_batch_size

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if axis == 1:
        return mini_batches
    else:
        return [(X.T, Y.T) for X, Y in mini_batches]


class OptimizationStrategy(Enum):
    normal = 1
    momentum = 2
    rmsprop = 3
    adam = 4


class HyperParameters(object):

    def __init__(self):

        self.mini_batch_size = 256

        # Regularization Parameters
        self.regularize = False
        self.lambd = 1

        self.dropout = False

        # Optimization Parameters
        self.optimization_strategy = OptimizationStrategy.normal

        self.num_epochs = 10000
        self.initial_learning_rate = 0.075
        self.min_learning_rate = 0.001

        self.learning_rate = self.initial_learning_rate
        self.learning_rate_decay = 0.0009

        def default_learning_rate_decay_function(lr, epoch):
            initial = self.initial_learning_rate
            decay = 1 / (1 + self.learning_rate_decay * epoch)
            new_lr = initial * decay
            return max(self.min_learning_rate, new_lr)

        self.learning_rate_decay_function = default_learning_rate_decay_function

        # Adam Optimization Parameters
        self.beta1 = 0.9
        self.beta2 = 0.999

        # Batch Normalization
        self.batch_norm = False

    def update_learning_rate(self, epoch):
        """
        Update the learning rate
        :param epoch: The iteration/epoch number over training
        :return: None
        """
        self.learning_rate = self.learning_rate_decay_function(self.learning_rate, epoch)


class NeuralNetwork(object):

    def __init__(self, dimensions):

        self.L = len(dimensions)  # number of dimensions
        self.dimensions = dimensions

        # Learning parameters
        self.W = [np.empty((0, 0)) for _ in range(self.L)]
        self.b = [np.empty((0, 0)) for _ in range(self.L)]

        # Cached values
        self.A = [np.empty((0, 0)) for _ in range(self.L)]
        self.Z = [np.empty((0, 0)) for _ in range(self.L)]

        self.dW = [np.empty((0, 0)) for _ in range(self.L)]
        self.db = [np.empty((0, 0)) for _ in range(self.L)]

        self.drop = [np.empty((0, 0)) for _ in range(self.L)]

        # Hyper-parameters
        self.hps = HyperParameters()

    def train(self, X, Y, hyper_params=None):
        """
        Train the Neural Network on a set of labeled data

        :param X: Features with shape (num features, num examples)
        :param Y: Labels for each example in X, should be size (1, num examples)
        :param hyper_params: HyperParameters object containing training hyper-params
        :return: Cost as a function of epoch
        """
        if hyper_params is not None:
            if type(hyper_params) is not HyperParameters:
                raise TypeError("parameters must be a HyperParameters")
            else:
                self.hps = hyper_params

        if X.shape[1] != Y.shape[1]:
            raise ValueError("Number of examples in X %s "
                             "not equal to number in Y %s" % (X.shape, Y.shape))

        if Y.shape[0] != 1:
            raise ValueError("Shape of Y %s, doesn't start with 1" % Y.shape)

        costs = []
        self.initialize_parameters()

        for epoch in range(self.hps.num_epochs):

            for mb_X, mb_Y in random_mini_batches(X, Y,
                                                  mini_batch_size=self.hps.mini_batch_size):
                AL = self.forward_prop(mb_X)

                cost = self.compute_cost(AL, mb_Y)
                costs.append(cost)

                self.back_prop(AL, mb_Y)
                self.parameter_update(self.hps.learning_rate)

            # Update the learning rate!
            self.hps.update_learning_rate(epoch)

            if epoch in [0, 1, 2] or \
                    is_power(epoch, 10) or \
                    epoch % (self.hps.num_epochs / 100) == 0:
                accuracy = self.compute_accuracy(self.predict(X), Y)
                print("Epoch %d\tcost %s\t train accuracy %s" % (epoch, costs[-1], accuracy))

        return costs

    def predict(self, X):
        """
        Make a prediction on data using the trained model
        :param X: Data/features to make predictions on
        :return: The predictions of the model
        """
        return (self.forward_prop(X) > 0.5).astype(int)

    def initialize_parameters(self):
        for l in range(1, self.L):
            # Initialize bias to zero
            self.b[l] = np.zeros((self.dimensions[l], 1))

            # He Initialization
            # Initialize weights to normal distribution with variance
            # equal to the size of the previous layer in order to keep
            # the variance of activations constant between layers of the network
            self.W[l] = np.random.randn(self.dimensions[l], self.dimensions[l - 1])
            self.W[l] *= np.sqrt(2 / self.dimensions[l - 1])

    def forward_prop(self, X, keep_prob=1.0):
        self.A[0] = X
        for l in range(1, self.L - 1):
            W, b = self.W[l], self.b[l]
            self.Z[l] = np.dot(W, self.A[l - 1]) + b
            self.A[l] = relu(self.Z[l])

            # todo: this is broken
            # self.drop[l] = np.random.rand(self.A[l].shape[0], self.A[l].shape[1]) < keep_prob
            # self.A[l] = np.multiply(self.A[l], self.drop[l]) / keep_prob

        ZL = np.dot(self.W[self.L - 1], self.A[self.L - 2]) + self.b[self.L - 1]
        AL = sigmoid(ZL)

        self.Z[self.L - 1] = ZL
        self.A[self.L - 1] = AL
        return AL

    def back_prop(self, AL, Y):
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dZ = sigmoid_backward(dAL, self.Z[self.L - 1])
        dA, dW, db = self.linear_backwards(dZ, self.L - 1)

        self.dW[self.L - 1] = dW
        self.db[self.L - 1] = db

        dA_prev = dA
        for l in reversed(range(1, self.L - 1)):
            dZ = relu_backward(dA_prev, self.Z[l])
            dA_prev, dW, db = self.linear_backwards(dZ, l)
            self.dW[l] = dW
            self.db[l] = db

    def linear_backwards(self, dZ, layer):
        A_prev, W, b = self.A[layer - 1], self.W[layer], self.b[layer]

        m = A_prev.shape[1]
        dW = np.dot(dZ, A_prev.T) / m
        if self.hps.regularize:
            dW += self.hps.lambd * W / m

        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def parameter_update(self, learning_rate):
        for l in range(1, self.L):
            self.W[l] -= learning_rate * self.dW[l]
            self.b[l] -= learning_rate * self.db[l]

    def compute_cost(self, AL, Y):
        cost = - np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL), axis=1) / Y.shape[1]
        cost = float(np.squeeze(cost))
        if self.hps.regularize:
            m = Y.shape[0]
            k = self.hps.lambd / (2 * m)
            cost += k * sum(map(lambda w: np.sum(np.square(w)), self.W))
        return cost

    def compute_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.shape[1]

