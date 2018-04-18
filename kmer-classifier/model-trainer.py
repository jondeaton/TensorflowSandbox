#!/usr/bin/env python
"""
File: model-trainer
Date: 4/17/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import numpy as np
import tensorflow as tf

import viral_kmers


def model_fn(features, labels, mode):

    layers = [features['x']]

    num_units = [100, 16, 8]
    for l in range(1, len(num_units)):
        layers.append(
            tf.layers.dense(inputs=layers[l - 1], units=num_units[l])
        )

    logits = tf.layers.dense(inputs=layers[-1], units=1)

    predictions = {
        "classes": tf.to_int32(logits > 0.5),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sigmoid_cross_entropy(labels, logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=5)

    virus_data = viral_kmers.load_dataset()

    train_data = virus_data.train.kmers
    train_labels = virus_data.train.labels

    eval_data = virus_data.eval.kmers
    eval_labels = virus_data.eval.labels

    virus_classifier = tf.estimator.Estimator(model_fn=model_fn,
                                              model_dir="/tmp/virus_deep_model")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},
                                                        y=train_labels,
                                                        batch_size=50,
                                                        num_epochs=None,
                                                        shuffle=True)

    virus_classifier.train(input_fn=train_input_fn,
                           steps=100000,
                           hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},
                                                       y=eval_labels,
                                                       num_epochs=5,
                                                       shuffle=False)

    eval_results = virus_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    main()
