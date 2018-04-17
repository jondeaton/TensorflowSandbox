#!/usr/bin/env python
"""
File: unblur
Date: 4/13/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import tensorflow as tf
import mahotas
import skimage.io as imgio
import numpy
import scipy.misc

def main():
    p_imgfile = 'monkey-02.jpg'
    p_imgscale = .33
    p_ksize = 4
    p_kiter = 3

    img_raw = mahotas.imread(p_imgfile)
    img_base = scipy.misc.imresize(img_raw, p_imgscale) / 255.0
    imgio.imsave('base.png', img_base)

    kernel = numpy.zeros([p_ksize, p_ksize, 3, 3])
    for c in range(0, 3):
        for xy in range(0, p_ksize):
            kernel[xy, xy, c, c] = 1.0 / p_ksize

    v_img = tf.Variable(tf.zeros(img_base.shape), name="Unblurred_Image")
    op_img_resize = tf.reshape(v_img, [-1, img_base.shape[0],
                                       img_base.shape[1], img_base.shape[2]])
    pl_kernel = tf.placeholder("float", shape=kernel.shape, name="Kernel")
    op_init = tf.initialize_all_variables()

    op_convolve = op_img_resize
    for blurStage in range(0, p_kiter):
        op_convolve = tf.nn.conv2d(op_convolve, pl_kernel,
                                   strides=[1, 1, 1, 1], padding='SAME')

    with tf.Session() as session:
        session.run(op_init)
        img_blurred = session.run(op_convolve, feed_dict={
            v_img: img_base, pl_kernel: kernel})

    imgio.imsave('blurred.png', img_blurred[0])

    pl_blurredImg = tf.placeholder("float", shape=img_blurred.shape)
    op_loss = tf.reduce_sum(tf.square(op_convolve - pl_blurredImg))
    op_optimize = tf.train.GradientDescentOptimizer(0.5).minimize(op_loss)

    def f_pixel(x):
        return 0 if x < 0 else 1 if x > 1 else x

    f_img = numpy.vectorize(f_pixel, otypes=[numpy.float])

    with tf.Session() as session:
        session.run(op_init)
        for epoch in range(0, 5):
            img_deblurred = f_img(session.run(v_img, feed_dict={
                pl_blurredImg: img_blurred, pl_kernel: kernel}))
            imgio.imsave("deblurred-%s.png" % epoch, img_deblurred)
            for iteration in range(0, 10):
                error = session.run([op_optimize, op_loss], feed_dict={
                    pl_blurredImg: img_blurred, pl_kernel: kernel})[1]
                print("%s/%s = %s" % (epoch, iteration, error))
        img_deblurred = f_img(session.run(v_img, feed_dict={
            pl_blurredImg: img_blurred, pl_kernel: kernel}))

    imgio.imsave("deblurred-final.png", img_deblurred)


if __name__ == "__main__":
    main()
