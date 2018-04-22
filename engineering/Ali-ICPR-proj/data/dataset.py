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

# Train-dataset directory .
INPUT_DATA = 'D:/engineering-data/Ali-ICPR-data/train_slice'

# Validation data percentage .
VALIDATION_PERCENTAGE = 10
# Test data percentage .
TEST_PERCENTAGE = 10

# Network arguments setting .
BATCH_SIZE = 100

class Dataset(object):
    def __init__(self, data_path, validation_percentage, testing_percentage):
        self.data_path = data_path
        self.validation_percentage = validation_percentage
        self.testing_percentage = testing_percentage
        self.testing_dataset = []
        self.validation_dataset = []
        self.train_dataset = []

    def split_dataset(self):
        img_list = []
        for rootdir, subdirs, filenames in os.walk(INPUT_DATA):
            for filename in filenames:
                img_list.append(os.path.join(rootdir, filename))

        for img in img_list:
            chance = np.random.randint(100)
            if chance < self.testing_percentage:
                self.testing_dataset.append(img)
            elif chance < self.testing_percentage + self.validation_percentage:
                self.validation_dataset.append(img)
            else:
                self.train_dataset.append(img)

    def get_train_dataset(self):
        return self.train_dataset
    def get_testing_dataset(self):
        return self.testing_dataset
    def get_validation_dataset(self):
        return self.validation_dataset

def main(argv=None):
    dt = Dataset(INPUT_DATA, VALIDATION_PERCENTAGE, TEST_PERCENTAGE)
    dt.split_dataset()
    test_dataset = dt.get_testing_dataset()
    print(test_dataset)
if __name__ == '__main__':
    tf.app.run()














