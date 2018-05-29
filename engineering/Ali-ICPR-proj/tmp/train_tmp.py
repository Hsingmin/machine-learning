# -*- coding: utf-8 -*-
#
# -*- version:
#		python 3.5.2
#		tensorflow 1.4.0 
#		numpy 1.13.1
#       Keras 2.1.2 Using Tensorflow backend
# -*- author: Hsingmin Lee
#
# train.py -- Train ocr model with dataset provided by data.dataset.Dataset .

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import sys
sys.path.append(r"D:\python_work\machine-learning\engineering\Ali-ICPR-proj")
import data.dataset as dd
import ocr.model as om
import codecs
# Train-dataset directory .
INPUT_DATA = 'D:/engineering-data/Ali-ICPR-data/alligned_slice'

# Validation data percentage .
VALIDATION_PERCENTAGE = 10
# Test data percentage .
TEST_PERCENTAGE = 10

# Network arguments setting .
# BATCH_SIZE = 32
BATCH_SIZE = 32
# Network arguments setting .
LEARNING_RATE = 0.01
STEPS = 4

characters = om.keys.alphabet[:]

TRUNCATED_WIDTH = 256
N_LEN = 10
EPOCH = 10

def load_dataset(category, path, validation_percentage, test_percentage):
    dt = dd.Dataset(path, validation_percentage, test_percentage)
    dt.split()
    if category == 'train':
        train_dataset = dt.get_train()
        return train_dataset
    elif category == 'validation':
        validation_dataset = dt.get_validation()
        return validation_dataset
    else:
        test_dataset = dt.get_test()
        return test_dataset

# Batch generator for ocr model training.
def batch_generator(category):
    # dataset = load_dataset(category, INPUT_DATA, VALIDATION_PERCENTAGE, TEST_PERCENTAGE)
    while True:
        X_batch, y_batch = batch_loader()
        yield (X_batch, y_batch)

# Batch provider for ocr model training.
def batch_loader(category=None):
    if category == None:
        return
    dataset = load_dataset(category, INPUT_DATA, VALIDATION_PERCENTAGE, TEST_PERCENTAGE)
    batch_list = []
    label_list = []
    X_batch = []
    y_batch = []

    for i in range(BATCH_SIZE):
        img_dir = dataset[np.random.randint(len(dataset))]
        batch_list.append(img_dir)
        label_list.append(os.path.basename(img_dir).split('.')[0])

    for img_dir in batch_list:
        image_raw = Image.open(img_dir)
        # image = np.array(image_raw.convert('RGB'))
        alligned_height = 64
        bimage = image_raw.convert('L')
        scale = bimage.size[1]*1.0/alligned_height
        width = int(bimage.size[0]/scale)
        image = bimage.resize((width, alligned_height))
        # image.save(os.path.join('./to', os.path.basename(img_dir)))
        image = np.array(image)
        # print(image)
        X_batch.append(image)

    # aligned_batch = dd.AlignedBatch(alligned_height, TRUNCATED_WIDTH)
    # X_batch = np.array(aligned_batch(X_batch))
    # print(label_list)
    X_batch = np.array(X_batch)
    aligned_onehot = dd.AlignedOnehot(N_LEN, characters)
    y_batch = np.array(aligned_onehot(label_list))

    return X_batch, y_batch

def input_allocate(X_batch, y_batch):
    X_batch = X_batch.reshape((BATCH_SIZE, 64, -1, 1))
    batch_size = X_batch.shape[0]
    input_length = int(X_batch.shape[2]/4)-2
    label_length = y_batch.shape[1]
    X, Y = [X_batch, y_batch, np.ones(batch_size)*input_length,
            np.ones(batch_size)*label_length], np.ones(batch_size)
    return X, Y

# Train ocr model .
def main(argv=None):
    aligned_height = 64
    # nclass = len(characters)
    # global_loss = 1000
    # model, basemodel = om.get_model(aligned_height, nclass)

    # Train model input:
    #   input = Input(name='the_input', shape=(height, None, 1))
    #   labels = Input(name='the_labels', shape=[None,], dtype='float32')
    #   input_length = Input(name='input_length', shape=[1], dtype='int64')
    #   label_length = Input(name='label_length', shape=[1], dtype='int64')
    for s in range(STEPS):
        X_batch, y_batch = batch_loader(category='train')
        print(X_batch.shape)
        print("---------------------------------------")
        print(y_batch.shape)
        X, Y = input_allocate(X_batch, y_batch)
        # model.train_on_batch(X, Y)
        # result = om.predict(X, basemodel)
        # result = pred_model.predict(X)
        print("---------------------------------------")
        # print(X)
        print("---------------------------------------")
        # print(Y)
if __name__ == '__main__':
	main(argv=None)















