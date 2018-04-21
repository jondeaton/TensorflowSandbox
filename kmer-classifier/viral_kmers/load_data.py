#!/usr/bin/env python
"""
File: load_dataset
Date: 4/17/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
import numpy as np

# Get the path to the data files
dir = os.path.dirname(__file__)
default_virus_file = os.path.join(dir, "virus.kmer")
default_bacteria_file = os.path.join(dir, "bacteria.kmer")

def extract_kmers(file, column_wise=False):
    """
    Reads a k-mer file into a numpy array

    :param file: Path to the file or StringIO containing the k-mer counts
    :param column_wise: If true, then each k-mer set will be a column
    :return: Numpy array with each set of k-mers as a row or column
    """
    content = np.loadtxt(file, delimiter=',', dtype=str)

    counts = content[:, 1:].astype(float)
    for i in range(counts.shape[0]):
        counts[i, :] /= np.sum(counts[i, :])
    return counts.T if column_wise else counts


class DataSetSlice(object):
    def __init__(self, kmers, labels):
        self.kmers = kmers
        self.labels = labels


class DataSet(object):
    def __init__(self, virus_file, bacteria_file):
        self.kmers, self.labels = self._load(virus_file, bacteria_file)
        self._setup_slices()

    def _load(self, virus_file, bacteria_file):
        self.X_vir = extract_kmers(virus_file, column_wise=True)
        self.X_bac = extract_kmers(bacteria_file, column_wise=True)
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

        self.train = DataSetSlice(self.kmers[:, train_idx],
                                  self.labels[:, train_idx])

        self.eval = DataSetSlice(self.kmers[:, eval_idx],
                                 self.labels[:, eval_idx])


def load_dataset(virus_file=default_virus_file, bacteria_file=default_bacteria_file):
    """
    Loads the k-mer data set

    :param virus_file: The file containing the virus k-mer counts
    :param bacteria_file: The file containing the bacteria k-mer counts
    :return: A DataSet object containing the loaded data
    """
    virus_file = default_bacteria_file if virus_file is None else virus_file
    bacteria_file = default_bacteria_file if bacteria_file is None else bacteria_file
    return DataSet(virus_file, bacteria_file)
