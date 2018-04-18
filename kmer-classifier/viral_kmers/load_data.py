#!/usr/bin/env python
"""
File: load_dataset
Date: 4/17/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
import numpy as np


class DataSetSlice(object):
    def __init__(self, kmers, labels):
        self.kmers = kmers
        self.labels = labels


class DataSet(object):
    def __init__(self, virus_file, bacteria_file):
        virus_file = os.path.expanduser(virus_file)
        bacteria_file = os.path.expanduser(bacteria_file)
        self.kmers, self.labels = self._load(virus_file, bacteria_file)
        self._setup_slices()

    def _load(self, virus_file, bacteria_file):
        def extract_X(file):
            content = np.loadtxt(file, delimiter=',', dtype=str)
            counts = content[:, 1:].astype(float)
            for i in range(counts.shape[0]):
                counts[i, :] /= np.sum(counts[i, :])
            return counts.T

        self.X_vir = extract_X(virus_file)
        self.X_bac = extract_X(bacteria_file)
        X = np.concatenate((self.X_vir, self.X_bac), axis=1)

        num_vir = self.X_vir.shape[1]
        num_bac = self.X_bac.shape[1]
        self.Y_vir = np.ones((1, num_vir))
        self.Y_bac = np.zeros((1, num_bac))
        y = np.concatenate((self.Y_vir, self.Y_bac), axis=1).astype(int)
        return X, y

    def _setup_slices(self):
        m = self.kmers.shape[1]

        idx = np.arange(m)
        np.random.shuffle(idx)

        eval_size = m // 10
        eval_idx = idx[:eval_size]
        train_idx = idx[eval_size:]

        self.train = DataSetSlice(self.kmers[:, train_idx].T,
                                  self.labels[:, train_idx].T)

        self.eval = DataSetSlice(self.kmers[:, eval_idx].T,
                                 self.labels[:, eval_idx].T)


def load_dataset(virus_file="viral_kmers/virus.kmer", bacteria_file="viral_kmers/bacteria.kmer"):
    return DataSet(virus_file, bacteria_file)
