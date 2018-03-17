
# -*- coding: utf-8 -*-
# -*- version: 
#	python 3.5.2
#	tensorflow 1.4.0
#	numpy 1.13.1
# -*-
#
# -*- author: Hsingmin Lee -*-
# image_to_tfrecord.py 
# Covert images into tfrecord files for batching into model training .
#
import tensorflow as tf
import numpy as np
import os

# Cast image features into tfrecord features .
def _int64_feature(value):
	return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _float_feature(value):
	return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))

def convert_to_tfrecord(image, n_classes, label_index, tfrecord):
	image_raw = tf.gfile.FastGFile(image, 'rb').read()
	
	image_label = np.zeros(n_classes, dtype=np.float32)
	image_label[label_index] = 1

	length = image_data.shape[0]
	width = image_data.shape[1]
	channel = image_data.shape[2]
	
	writer = tf.python_io.TFRecordWriter(tfrecord)
	# Convert into Example Protocol Buffer .
	example = tf.train.Example(features=tf.train.Feature(features={
		'length': _int64_feature(length),
		'width': _int64_feature(width),
		'channel': _int64_feature(channel),
		'image_raw': _bytes_feature(image_raw),
		'label': _int64_feature(label)
		}))

	writer.write(example.SerializedToString(example))

	writer.close()

def parse_from_tfrecord(example):
	return tf.parse_single_example(
			example, features={
				'length': tf.FixedLenFeature([], tf.int64),
				'width': tf.FixedLenFeature([], tf.int64),
				'channel': tf.FixedLenFeature([], tf.int64),
				'image_raw': tf.FixedLenFeature([], tf.string),
				'label': tf.FixedLenFeature([], tf.int64)})






























