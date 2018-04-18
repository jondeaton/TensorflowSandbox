#!/usr/bin/env python
"""
File: cifar
Date: 4/17/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


class CIFAR(object):

    def __init__(self, root):
        self.root = os.path.expanduser(root)

    def batches(self):


