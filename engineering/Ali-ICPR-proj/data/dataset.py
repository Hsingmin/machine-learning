# -*- coding: utf-8 -*-
#
# -*- version:
#		python 3.5.2
#		tensorflow 1.4.0 
#		numpy 1.13.1
# -*- author: Hsingmin Lee
#
# dataset.py -- import data.dataset to get train/validate-dataset and
# test dataset .

import glob
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

"""
# Train-dataset directory .
INPUT_DATA = 'D:/engineering-data/Ali-ICPR-data/train_slice'

# Validation data percentage .
VALIDATION_PERCENTAGE = 10
# Test data percentage .
TEST_PERCENTAGE = 10

# Network arguments setting .
BATCH_SIZE = 100
"""

class Dataset(object):
    def __init__(self, data_path, validation_percentage, test_percentage):
        self.data_path = data_path
        self.validation_percentage = validation_percentage
        self.test_percentage = test_percentage
        self.test = []
        self.validation = []
        self.train = []

    def split(self):
        img_list = []
        for rootdir, subdirs, filenames in os.walk(self.data_path):
            for filename in filenames:
                img_list.append(os.path.join(rootdir, filename))

        for img in img_list:
            chance = np.random.randint(100)
            if chance < self.test_percentage:
                self.test.append(img)
            elif chance < self.test_percentage + self.validation_percentage:
                self.validation.append(img)
            else:
                self.train.append(img)

    def get_train(self):
        return self.train
    def get_test(self):
        return self.test
    def get_validation(self):
        return self.validation

class AlignedBatch(object):
    def __init__(self, height=32, width=256):
        self.height = height
        self.width = width

    def __call__(self, batch):
        aligned_batch = []
        self.width = max(np.shape(sample)[1] for sample in batch)
        for sample in batch:
            padding = np.array([[0]*(self.width-np.shape(sample)[1])]*np.shape(sample)[0])
            padding_sample = np.concatenate((sample, padding), axis=1)
            aligned_batch.append(padding_sample)
        return aligned_batch

"""
def main(argv=None):
    dt = Dataset(INPUT_DATA, VALIDATION_PERCENTAGE, TEST_PERCENTAGE)
    dt.split()
    test_dataset = dt.get_test()
    print(test_dataset)
if __name__ == '__main__':
    tf.app.run()
"""













