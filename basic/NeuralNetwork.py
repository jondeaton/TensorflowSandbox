#!/usr/bin/env python
"""
File: NeuralNetwork
Date: 4/14/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import numpy as np
from activations import *
import math

def is_power(num, base):
    if base == 1 and num != 1: return False
    if base == 1 and num == 1: return True
    if base == 0 and num != 1: return False
    power = int (math.log(num, base) + 0.5)
    return base ** power == num


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

        self.regularize = False
        self.lambd = 1

    def train(self, X, Y, iterations=20000, learning_rate=0.0075,
              regularize=False,
              dropout=False):
        self.regularize = regularize
        self.dropout = dropout

        costs = []
        self.initialize_parameters()
        self.A[0] = X
        for i in range(iterations):
            AL = self.forward_prop(X)
            costs.append(self.compute_cost(AL, Y))
            self.back_prop(AL, Y)
            self.parameter_update(learning_rate)

            learning_rate = max(0.0075, learning_rate * 0.9999)

            if i in [0, 1, 2] or is_power(i, 10) or i % (iterations / 100) == 0:
                accuracy = self.compute_accuracy(self.predict(X), Y)
                print("Iteration  %d \t cost: %s \t accuracy: %s" % (i, costs[-1], accuracy))

        return costs

    def predict(self, X):
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
        if self.regularize:
            dW += self.lambd * W / m

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
        if self.regularize:
            m = Y.shape[0]
            k = self.lambd / (2 * m)
            cost += k * sum(map(lambda w: np.sum(np.square(w)), self.W))
        return cost

    def compute_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.shape[1]

