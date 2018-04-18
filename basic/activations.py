#!/usr/bin/env python
"""
File: activations
Date: 4/14/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid of x
    Arguments:
    x -- A scalar or numpy array of any size.
    Return:
    s -- sigmoid(x)
    """
    return 1 / (1 + np.exp(-x))


def relu(x):
    return x * (x > 0)


def relu_backward(dA, Z):
    """
    Backward propagation for a single RELU unit.

    :param dA: post-activation gradient, of any shape
    :param Z: pre-activation value computed in forward prop
    :return: dZ -- Gradient of the cost with respect to Z
    """
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def sigmoid_backward(dA, Z):
    """
    Backward propagation for a single SIGMOID unit.

    :param dA: post-activation gradient, of any shape
    :param Z: pre-activation value computed in forward prop
    :return: dZ -- Gradient of the cost with respect to Z
    """
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ