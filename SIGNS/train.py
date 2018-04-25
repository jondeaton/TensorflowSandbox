#!/usr/bin/env python
"""
File: train
Date: 4/24/18 
Author: Jon Deaton (jdeaton@stanford.edu)

Adapted from Andrew Ng's deep learning Coursera Course

"""

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict


# Training set: 1080 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (180 pictures per number).
# Test set:      120 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 ( 20 pictures per number).


def main():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # Flatten the training and test images
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

    # Normalize image vectors
    X_train = X_train_flatten / 255.
    X_test = X_test_flatten / 255.

    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6)

    # Number of features and categories
    n_x, m = X_train.shape
    n_y = Y_train.shape[0]

    n_h = (25, 12, 6)
    n_h1, n_h2, n_h3 = n_h

    learning_rate = 0.0001
    num_epochs = 1500
    mini_batch_size = 32
    print_cost = True
    costs = []

    # computation graph
    ops.reset_default_graph()

    X = tf.placeholder(tf.float32, shape=(n_x, None))
    Y = tf.placeholder(tf.float32, shape=(n_y, None))

    W1 = tf.get_variable("W1", [n_h1, n_x], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [n_h1, 1], initializer=tf.zeros_initializer())

    W2 = tf.get_variable("W2", [n_h2, n_h1], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [n_h2, 1], initializer=tf.zeros_initializer())

    W3 = tf.get_variable("W3", [n_h3, n_h2], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [n_h3, 1], initializer=tf.zeros_initializer())

    Z1 = tf.matmul(W1, X) + b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(W3, A2) + b3

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y)
    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m / mini_batch_size)

            for mb_X, mb_y in random_mini_batches(X_train, Y_train,
                                                  mini_batch_size=mini_batch_size):

                _, c = sess.run([optimizer, cost], feed_dict={X: mb_X, Y: mb_y})

            if print_cost and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost and epoch % 5 == 0:
                costs.append(epoch_cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # lets save the parameters in a variable
    parameters = sess.run(parameters)
    print ("Parameters have been trained!")

    # Calculate the correct predictions
    correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
    print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))


if __name__ == "__main__":
    main()


