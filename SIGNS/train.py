#!/usr/bin/env python
"""
File: train
Date: 4/24/18 
Author: Jon Deaton (jdeaton@stanford.edu)

Adapted from Andrew Ng's deep learning Coursera Course
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import argparse
import logging
import functools
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.python.framework import ops

from SIGNS.tf_utils import *
from SIGNS.cnn_utils import *

logger = logging.getLogger()

tensorboard_dir = "tensorboard/job_signs_%s" % time.time()
save_file = "./trained_model.ckpt-data"


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


def cnn_model(X, Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]

    # # conv1
    # with tf.variable_scope('conv1') as scope:
    #     kernel = _variable_with_weight_decay('weights',
    #                                          shape=[5, 5, 3, 64],
    #                                          stddev=5e-2,
    #                                          wd=None)
    #     conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    #     biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    #     pre_activation = tf.nn.bias_add(conv, biases)
    #     conv1 = tf.nn.relu(pre_activation, name=scope.name)
    #     _activation_summary(conv1)
    #
    # # pool1
    # pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
    #                        padding='SAME', name='pool1')
    # # norm1
    # norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                   name='norm1')
    #
    # # conv2
    # with tf.variable_scope('conv2') as scope:
    #     kernel = _variable_with_weight_decay('weights',
    #                                          shape=[5, 5, 64, 64],
    #                                          stddev=5e-2,
    #                                          wd=None)
    #     conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    #     biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    #     pre_activation = tf.nn.bias_add(conv, biases)
    #     conv2 = tf.nn.relu(pre_activation, name=scope.name)
    #     _activation_summary(conv2)
    #
    # # norm2
    # norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                   name='norm2')
    # # pool2
    # pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
    #                        strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    #
    # # local3
    # with tf.variable_scope('local3') as scope:
    #     # Move everything into depth so we can perform a single matrix multiply.
    #     reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
    #     dim = reshape.get_shape()[1].value
    #     weights = _variable_with_weight_decay('weights', shape=[dim, 384],
    #                                           stddev=0.04, wd=0.004)
    #     biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    #     local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    #     _activation_summary(local3)
    #
    # # local4
    # with tf.variable_scope('local4') as scope:
    #     weights = _variable_with_weight_decay('weights', shape=[384, 192],
    #                                           stddev=0.04, wd=0.004)
    #     biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    #     local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    #     _activation_summary(local4)
    #
    # # linear layer(WX + b),
    # # We don't apply softmax here because
    # # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # # and performs the softmax internally for efficiency.
    # with tf.variable_scope('softmax_linear') as scope:
    #     weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
    #                                           stddev=1 / 192.0, wd=None)
    #     biases = _variable_on_cpu('biases', [NUM_CLASSES],
    #                               tf.constant_initializer(0.0))
    #     softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    #     _activation_summary(softmax_linear)
    #
    # return softmax_linear


def highlevel_model(X, Y, n_h):
    """
    Create a high-level model to train

    :param X_train: Features for training data
    :param Y_train: Labels for training data
    :param X_test: Features for test data
    :param Y_test: Labels for test data
    :return: None
    """
    # Number of features and categories
    n_x = X.shape[0]
    n_y = Y.shape[0]
    n_h1, n_h2, n_h3 = n_h

    X_T = tf.transpose(X)

    with tf.variable_scope('hidden_1') as scope:
        h1 = tf.layers.dense(inputs=X_T, units=25, activation=tf.nn.relu, name=scope.name)

    with tf.variable_scope("hidden_2") as scope:
        h2 = tf.layers.dense(inputs=h1, units=12, activation=tf.nn.relu, name=scope.name)

    with tf.variable_scope("output") as scope:
        logits = tf.layers.dense(inputs=h2, units=6, activation=tf.nn.softmax, name=scope.name)

    return tf.transpose(logits)


def simple_model(X, Y, n_h):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    n_h1, n_h2, n_h3 = n_h

    def xavier():
        return tf.contrib.layers.xavier_initializer()

    def zeros():
        tf.zeros_initializer()

    # First hidden layer
    with tf.name_scope('h1') as scope:
        W1 = tf.get_variable("W1", [n_h1, n_x], initializer=xavier())
        b1 = tf.get_variable("b1", [n_h1, 1], initializer=zeros())
        Z1 = tf.matmul(W1, X) + b1
        A1 = tf.nn.relu(Z1)
        tf.summary.histogram("weights", W1)
        tf.summary.histogram("biases", b1)
        tf.summary.histogram("activations", A1)

    # Second hidden layer
    with tf.name_scope('h2') as scope:
        W2 = tf.get_variable("W2", [n_h2, n_h1], initializer=xavier())
        b2 = tf.get_variable("b2", [n_h2, 1], initializer=zeros())
        Z2 = tf.matmul(W2, A1) + b2
        A2 = tf.nn.relu(Z2)
        tf.summary.histogram("weights", W2)
        tf.summary.histogram("biases", b2)
        tf.summary.histogram("activations", A2)

    # Output layer
    with tf.name_scope('output') as scope:
        W3 = tf.get_variable("W3", [n_h3, n_h2], initializer=xavier())
        b3 = tf.get_variable("b3", [n_h3, 1], initializer=zeros())
        Z3 = tf.matmul(W3, A2) + b3
        logits = tf.transpose(Z3, name="logits")
        tf.summary.histogram("weights", W3)
        tf.summary.histogram("biases", b3)
        tf.summary.histogram("logits", logits)

    return logits


def train(X_train, Y_train, X_test, Y_test):
    """
    Train the model with simple construction

    :param X_train: Features for training data
    :param Y_train: Labels for training data
    :param X_test: Features for test data
    :param Y_test: Labels for test data
    :return: Array of costs
    """

    # Number of features and categories
    n_x, m = X_train.shape
    n_y = Y_train.shape[0]

    n_h = (25, 12, 6)

    logger.info("Creating computation graph...")
    ops.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(n_x, None))
    Y = tf.placeholder(tf.float32, shape=(n_y, None))

    logits = highlevel_model(X, Y, n_h)

    # labels = tf.transpose(Y)
    labels = Y
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=labels)
    cost = tf.reduce_mean(cross_entropy, name="cost")

    global_step = tf.Variable(0, name='global_step', trainable=False)
    sgd = tf.train.AdamOptimizer(learning_rate=learning_rate, name="gradient-descent")
    optimizer = sgd.minimize(cost, name='optimizer', global_step=global_step)

    # Accuracy assessments
    correct_prediction = tf.equal(tf.argmax(logits), tf.argmax(Y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.global_variables_initializer()

    logger.info("Training...")
    with tf.Session() as sess:
        sess.run(init)
        tf.summary.scalar('cost', cost)

        tf.summary.scalar("training_accuracy", accuracy)
        test_summary = tf.summary.scalar("Test accuracy", accuracy)
        merged_summary = tf.summary.merge_all()

        writer = tf.summary.FileWriter(logdir=tensorboard_dir)
        writer.add_graph(sess.graph)

        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m / mini_batch_size)

            batches = random_mini_batches(X_train, Y_train, mini_batch_size=mini_batch_size)
            mb_cost = 0
            for mb_X, mb_y in batches:
                _, mb_cost = sess.run([optimizer, cost], feed_dict={X: mb_X, Y: mb_y})
                epoch_cost += mb_cost / num_minibatches

            # Report progress to TensorBoard
            if epoch % 10 == 0:
                s = sess.run(merged_summary, feed_dict={X: X_train, Y: Y_train})
                writer.add_summary(s, epoch)

                test_acc, test_summ = sess.run([accuracy, test_summary], feed_dict={X: X_test, Y: Y_test})
                writer.add_summary(test_summ, epoch)

            # Report progress to console
            if epoch % 100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={X: X_train, Y: Y_train})
                test_accuracy = sess.run(accuracy, feed_dict={X: X_test, Y: Y_test})
                logger.info("Epoch %d:\tcost:\t%f,\t%f train,\t%f test" %
                            (epoch, epoch_cost, train_accuracy, test_accuracy))
        logger.info("Training complete.")

        logger.info("Saving model to: %s" % save_file)
        saver = tf.train.Saver()
        saver.save(sess, save_file, global_step=global_step)
        logger.info("Done saving model.")


def restore_model(save_file):
    """
    Restores the model that was saved to file after training
    :return: None
    """

    with tf.Session as sess:
        saver = tf.train.import_meta_graph(save_file)
        saver.restore(sess, tf.train.latest_checkpoint('./'))


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
    io_options_group.add_argument("--save-file", help="File to save trained model in")
    io_options_group.add_argument("--tensorboard", help="TensorBoard directory")

    hyper_params_group = parser.add_argument_group("Hyper-Parameters")
    hyper_params_group.add_argument("-l", "--learning-rate", type=float, default=0.0001, help="Learning rate")
    hyper_params_group.add_argument("-e", "--epochs", type=int, default=1500, help="Number of training epochs")
    hyper_params_group.add_argument("-mb", "--mini-batch", type=int, default=128, help="Mini-batch size")

    logging_options_group = parser.add_argument_group("Logging")
    logging_options_group.add_argument('--log', dest="log_level", default="WARNING", help="Logging level")
    logging_options_group.add_argument('--log-file', default="model.log", help="Log file")

    args = parser.parse_args()

    # Setup the logger
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

    if args.cloud:
        logger.info("Running on Google Cloud.")
    else:
        logger.debug("Running locally.")

    global learning_rate, num_epochs, mini_batch_size
    learning_rate = args.learning_rate
    num_epochs = args.epochs
    mini_batch_size = args.mini_batch

    logger.info("Loading SIGNS data-set...")
    X_train, Y_train, X_test, Y_test = load_SNIGNS()
    logger.info("Data-set loaded.")

    logger.info("Training...")

    logger.info("Learning rate: %s" % learning_rate)
    logger.info("Num epochs: %s" % num_epochs)
    logger.info("Mini-batch size: %s" % mini_batch_size)

    train(X_train, Y_train, X_test, Y_test)


if __name__ == "__main__":
    main()


