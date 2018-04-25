#!/usr/bin/env python
"""
File: train
Date: 4/24/18 
Author: Jon Deaton (jdeaton@stanford.edu)

Adapted from Andrew Ng's deep learning Coursera Course

"""

import os, sys
import argparse
import logging
import functools

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

logger = logging.getLogger()

tensorboard_dir = "tensorboard"


def lazy_property(function):
    # from: https://danijar.com/structuring-your-tensorflow-models/
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def define_scope(function):
    # from: https://danijar.com/structuring-your-tensorflow-models/
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class SignsModel:

    def __init__(self):
        self.data = None
        self.target = None

    @lazy_property
    def cost(self):
        pass

    @lazy_property
    def prediction(self):
        pass


def load_SNIGNS():
    """
    Loads the SIGNS training and test set

    Training set: 1080 pictures (64 by 64 pixels) of signs
    representing numbers from 0 to 5 (180 pictures per number).

    Test set:      120 pictures (64 by 64 pixels) of signs
     representing numbers from 0 to 5 ( 20 pictures per number).
    :return: X_train, Y_train, X_test, Y_test
    """

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

    return X_train, Y_train, X_test, Y_test


def train(X_train, Y_train, X_test, Y_test):
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

    logger.info("Creating computation graph...")
    # computation graph
    ops.reset_default_graph()

    X = tf.placeholder(tf.float32, shape=(n_x, None))
    Y = tf.placeholder(tf.float32, shape=(n_y, None))

    # First hidden layer
    with tf.name_scope('h1') as scope:
        W1 = tf.get_variable("W1", [n_h1, n_x],
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", [n_h1, 1],
                             initializer=tf.zeros_initializer())
        Z1 = tf.matmul(W1, X) + b1
        A1 = tf.nn.relu(Z1)

    # Second hidden layer
    with tf.name_scope('h2') as scope:
        W2 = tf.get_variable("W2", [n_h2, n_h1],
                             initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2", [n_h2, 1],
                             initializer=tf.zeros_initializer())
        Z2 = tf.matmul(W2, A1) + b2
        A2 = tf.nn.relu(Z2)

    with tf.name_scope('output') as scope:
        W3 = tf.get_variable("W3", [n_h3, n_h2],
                             initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable("b3", [n_h3, 1],
                             initializer=tf.zeros_initializer())
        Z3 = tf.matmul(W3, A2) + b3
        logits = tf.transpose(Z3)

    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Accuracy assessments
    correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.global_variables_initializer()

    logger.info("Training...")
    with tf.Session() as sess:
        sess.run(init)

        summary_writer = tf.summary.FileWriter(logdir=tensorboard_dir)

        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m / mini_batch_size)

            batches = random_mini_batches(X_train, Y_train, mini_batch_size=mini_batch_size)
            mb_cost = 0
            for mb_X, mb_y in batches:
                _, mb_cost = sess.run([optimizer, cost], feed_dict={X: mb_X, Y: mb_y})
                epoch_cost += mb_cost / num_minibatches

            costs.append(epoch_cost)

            # summarize to TensorBoard
            train_summary = tf.summary.scalar("training_accuracy", accuracy)
            test_summary = tf.summary.scalar("test_accuracy", accuracy)
            if epoch % 10 == 0:
                train_acc, train_summ = sess.run(
                    [accuracy, train_summary],
                    feed_dict={X: X_train, Y: Y_train})

                test_acc, test_summ = sess.run(
                    [accuracy, test_summary],
                    feed_dict={X: X_test, Y: Y_test})
                summary_writer.add_summary(train_summ, epoch)
                summary_writer.add_summary(test_summ, epoch)

            # Prints things to console
            if print_cost and epoch % 100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={X: X_train, Y: Y_train})
                test_accuracy = sess.run(accuracy, feed_dict={X: X_test, Y: Y_test})
                logger.info("Epoch %d:\tcost:\t%f,\t%f train,\t%f test" %
                            (epoch, epoch_cost, train_accuracy, test_accuracy))

        logger.info("Training complete.")

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def parse_args():
    """
    Parse the command line options for this file
    :return: An argparse object containing parsed arguments
    """

    parser = argparse.ArgumentParser(description="Train model to classify ASL digits",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    info_options_group = parser.add_argument_group("Info")
    info_options_group.add_argument("--job-dir", default=None, help="Job directory")
    info_options_group.add_argument("-cloud", "--cloud", action="store_true", help="Set true if running in cloud")

    io_options_group = parser.add_argument_group("I/O")
    io_options_group.add_argument("--virus-file", help="Virus k-mer counts")
    io_options_group.add_argument("--bacteria-file", help="Bacteria k-mer counts")

    options_group = parser.add_argument_group("General")
    options_group.add_argument("--simple", action="store_true", help="Run the simple way")
    options_group.add_argument("-i", "--iterations", type=int, default=1000, help="Number of iterations")
    options_group.add_argument('-p', '--pool-size', type=int, default=20, help="Thread-pool size")

    logging_options_group = parser.add_argument_group("Logging")
    logging_options_group.add_argument('--log', dest="log_level", default="WARNING", help="Logging level")
    logging_options_group.add_argument('--log-file', default="model.log", help="Log file")

    args = parser.parse_args()

    global logger
    logger = logging.getLogger('root')

    # Logging level configuration
    log_level = getattr(logging, args.log_level.upper())
    if not isinstance(log_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)

    log_formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')

    # For the log file...
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # For the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    logger.setLevel(log_level)

    return args


def main():
    args = parse_args()

    logger.info("Loading SIGNS data-set...")
    X_train, Y_train, X_test, Y_test = load_SNIGNS()
    logger.info("Data-set loaded.")

    train(X_train, Y_train, X_test, Y_test)


if __name__ == "__main__":
    main()


