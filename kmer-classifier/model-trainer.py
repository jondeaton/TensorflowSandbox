#!/usr/bin/env python
"""
File: model-trainer
Date: 4/17/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
import numpy as np
import tensorflow as tf

# For reading TensorFlow things
from tensorflow.python.lib.io import file_io
import StringIO

# Module for loading k-mer count files into numpy
import viral_kmers

import logging
import argparse

logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)


def as_session(train_data, train_labels, eval_data, eval_labels, num_epochs=10000):

    with tf.Session() as sess:
        input_layer = tf.reshape(train_data, [-1, 256])

        # input_layer = tf.placeholder(tf.float64, shape=train_data.shape[0])

        hidden_layer = tf.layers.dense(inputs=input_layer, units=8)
        logits = tf.layers.dense(inputs=hidden_layer, units=1)

        # todo: am I using the correct loss function ?!
        # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=train_labels, logits=logits, name='cross_entropy_per_example')
        # cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")
        # loss = tf.add_n(cross_entropy_mean)

        loss = tf.losses.sigmoid_cross_entropy(train_labels.T, logits)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        init = tf.global_variables_initializer()

        def get_accuracy(features, labels):
            predictions = tf.to_int32(logits > 0.5)
            # todo: figure out how to get this to work...
            # accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
            # return sess.run(accuracy, feed_dict={input_layer: features})
            p = sess.run(predictions, feed_dict={input_layer: features})
            accuracy = np.sum(p == labels, dtype=float) / labels.shape[0]
            return accuracy

        sess.run(init)

        for i in range(num_epochs):
            sess.run(train_op, feed_dict={input_layer: train_data})
            if i % (num_epochs / 10) == 0:
                trained_result = sess.run(loss, feed_dict={input_layer: train_data})
                logger.info("Training loss: %f" % trained_result)

                # todo: figure out how to make this work for the evaluation data
                accuracy = get_accuracy(train_data, train_labels.T)
                logger.info("Eval accuracy: %s" % accuracy)


def model_fn(features, labels, mode):
    """
    A version of this model made using tf.estimator (easy version)

    :param features: Features of the dataset
    :param labels: Labels of the dataset
    :param mode: some tf.estimator.ModeKeys
    :return: tf.estimator.EstimatorSpec
    """

    input_layer = features['x']
    hidden_layer = tf.layers.dense(inputs=input_layer, units=8)
    logits = tf.layers.dense(inputs=hidden_layer, units=1)

    predictions = {
        "classes": tf.to_int32(logits > 0.5),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sigmoid_cross_entropy(labels, logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def run_with_training_wheels(train_data, train_labels, eval_data, eval_labels):
    """
    Run a simpler version of this analysis

    :param train_data:
    :param train_labels:
    :param eval_data:
    :param eval_labels:
    :return:
    """

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100000)

    logger.info("Creating model...")
    virus_classifier = tf.estimator.Estimator(model_fn=model_fn)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},
                                                        y=train_labels,
                                                        batch_size=300,
                                                        num_epochs=None,
                                                        shuffle=True)
    logger.info("Training model...")
    virus_classifier.train(input_fn=train_input_fn, steps=1000000, hooks=[logging_hook])

    logger.info("Evaluating model...")
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=10, shuffle=False)
    eval_results = virus_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


def parse_args():
    """
    Parse the command line options for this file
    :return: An argparse object containing parsed arguments
    """
    parser = argparse.ArgumentParser(description="Merge the Reddit data-set",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    info_options_group = parser.add_argument_group("Info")
    info_options_group.add_argument("--job-dir", default=None, help="Directory of the job")

    io_options_group = parser.add_argument_group("I/O Options")
    io_options_group.add_argument("--virus-file", help="Virus k-mer counts")
    io_options_group.add_argument("--bacteria-file", help="Bacteria k-mer counts")

    options_group = parser.add_argument_group("Options")
    options_group.add_argument("--simple", action="store_true", help="Run the simple way")
    options_group.add_argument('-p', '--pool-size', type=int, default=20, help="Thread-pool size")

    console_options_group = parser.add_argument_group("Console Options")
    console_options_group.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    console_options_group.add_argument('--debug', action='store_true', help='Debug Console')

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    elif args.verbose:
        logger.setLevel(logging.INFO)
        logging.basicConfig(format='[%(asctime)s][%(levelname)s][%(funcName)s] - %(message)s')
    else:
        logger.setLevel(logging.WARNING)
        logging.basicConfig(format='[log][%(levelname)s] - %(message)s')

    return args


def main():
    args = parse_args()

    logger.info("Loading k-mer data set...")

    if args.virus_file is not None and os.path.exists(args.virus_file):
        virus_file = StringIO.StringIO(file_io.read_file_to_string(args.virus_file))
    else:
        logger.info("Loading default virus k-mer file.")
        virus_file = None

    if args.bacteria_file is not None and os.path.exists(args.bacteria_file):
        bacteria_file = StringIO.StringIO(file_io.read_file_to_string(args.bacteria_file))
    else:
        logger.info("Loading default bacteria k-mer file.")
        bacteria_file = None

    virus_data = viral_kmers.load_dataset(virus_file=virus_file, bacteria_file=bacteria_file)

    train_data = virus_data.train.kmers
    train_labels = virus_data.train.labels

    eval_data = virus_data.eval.kmers
    eval_labels = virus_data.eval.labels
    logger.info("Loaded data set.")

    if args.simple:
        run_with_training_wheels(train_data, train_labels, eval_data, eval_labels)
    else:
        as_session(train_data.T, train_labels, eval_data.T, eval_labels)


if __name__ == "__main__":
    main()
