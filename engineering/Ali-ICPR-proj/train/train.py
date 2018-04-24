# -*- coding: utf-8 -*-
#
# -*- version:
#		python 3.5.2
#		tensorflow 1.4.0 
#		numpy 1.13.1
# -*- author: Hsingmin Lee
#
# train.py -- Train ocr model with dataset provided by data.dataset.Dataset .

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from PIL import Image
import sys
sys.path.append(r"D:\python_work\machine-learning\engineering\Ali-ICPR-proj")
import data.dataset as dataset
import ocr.model as om

# Train-dataset directory .
INPUT_DATA = 'D:/engineering-data/Ali-ICPR-data/train_slice'

# Validation data percentage .
VALIDATION_PERCENTAGE = 10
# Test data percentage .
TEST_PERCENTAGE = 10

# Network arguments setting .
BATCH_SIZE = 32

# Network arguments setting .
LEARNING_RATE = 0.01
STEPS = 4000

def load_dataset(path, validation_percentage, test_percentage):
    dt = dataset.Dataset(path, validation_percentage, test_percentage)
    dt.split()
    train_dataset = dt.get_train()
    # validation_dataset = dt.get_validation()
    # test_dataset = dt.get_test()
    return train_dataset

# Batch generator for ocr model training.
def batch_generator():
    # train_dataset = load_dataset(INPUT_DATA, VALIDATION_PERCENTAGE, TEST_PERCENTAGE)
    while True:
        X_train, y_train = batch_loader()
        yield (X_train, y_train)


# Batch provider for ocr model training.
def batch_loader():
    train_dataset = load_dataset(INPUT_DATA, VALIDATION_PERCENTAGE, TEST_PERCENTAGE)
    batch_list = []
    label_list = []
    X_train = []
    y_train = []

    for i in range(BATCH_SIZE):
        img_dir = train_dataset[np.random.randint(len(train_dataset))]
        batch_list.append(img_dir)
        label_list.append(os.path.basename(img_dir).split('.')[0])

    for img_dir in batch_list:
        image_raw = Image.open(img_dir)
        # image = np.array(image_raw.convert('RGB'))
        alligned_height = 32
        bimage = image_raw.convert('L')
        scale = bimage.size[1]*1.0/alligned_height
        width = int(bimage.size[0]/scale)
        image = bimage.resize((width, alligned_height))
        # image.save(os.path.join('./to', os.path.basename(img_dir)))
        image = np.array(image)
        # print(image)
        X_train.append(image)
    y_train = np.array(label_list)

    aligned_batch = dataset.AlignedBatch(alligned_height, 256)
    X_train = np.array(aligned_batch(X_train))
    return X_train, y_train

# Train ocr model .
def main(argv=None):
    aligned_height = 32
    characters = om.keys.alphabet[:]
    nclass = len(characters)
    model, basemodel = om.get_model(aligned_height, nclass)
    # Get train batch .
    # Train model input:
    #   input = Input(name='the_input', shape=(height, None, 1))
    #   labels = Input(name='the_labels', shape=[None,], dtype='float32')
    #   input_length = Input(name='input_length', shape=[1], dtype='int64')
    #   label_length = Input(name='label_length', shape=[1], dtype='int64')
    for s in range(STEPS):
        X_batch, y_batch = batch_loader()
        # print(np.shape(X_batch))
        # print(np.shape(y_batch))
        





if __name__ == '__main__':
	tf.app.run()















