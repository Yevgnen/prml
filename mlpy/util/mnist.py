#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import array
import struct

import numpy as np


def loadMNISTImages(file_name):
    image_file = open(file_name, 'rb')
    head1 = image_file.read(4)
    head2 = image_file.read(4)
    head3 = image_file.read(4)
    head4 = image_file.read(4)

    num_examples = struct.unpack('>I', head2)[0]
    num_rows = struct.unpack('>I', head3)[0]
    num_cols = struct.unpack('>I', head4)[0]
    dataset = np.zeros((num_rows * num_cols, num_examples))
    images_raw = array.array('B', image_file.read())
    image_file.close()

    for i in range(num_examples):
        limit1 = num_rows * num_cols * i
        limit2 = num_rows * num_cols * (i + 1)
        dataset[:, i] = images_raw[limit1:limit2]

    return dataset.T / 255

def loadMNISTLabels(file_name):
    label_file = open(file_name, 'rb')
    head1 = label_file.read(4)
    head2 = label_file.read(4)

    num_examples = struct.unpack('>I', head2)[0]
    labels = np.zeros(num_examples, dtype=np.int)
    labels_raw = array.array('b', label_file.read())
    label_file.close()

    labels[:] = labels_raw[:]

    return labels
