
# -*- coding: utf-8 -*-
# -*- version:
#	python 3.5.2
#	Tensorflow 1.4.0
#	numpy 1.13.1
# -*- ----------------------------------------- -*-
# tfrecord_writer.py -- Tensorflow provides TFRecord file to store
# different type of data .
# 
# Data in TFRecord file stored in tf.train.Example Protocol Buffer .

'''
message Example{
	Features features = 1;		
};

message Features{
	map<string, Feature> feature = 1;		
};

message Feature{
	oneof kind{
		BytesList bytes_list = 1;
		FloatList float_list = 2;
		Int64List int64_list = 3;
	}		
};
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# Produce int64 type feature .
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Produce bytes type feature .
def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Produce float type feature .
def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

mnist = input_data.read_data_sets("./data", dtype=tf.uint8, one_hot=True)
# Image data .
images = mnist.train.images
# Real labels of image as one feature to store into TFRecord file .
labels = mnist.train.labels
# Image resolution as one feature to store into Example .
pixels = images.shape[1]
num_examples = mnist.train.num_examples

# Path to store TFRecord file .
filename = "./to/output.tfrecords"

# Creater a writer to write TFRecord file .
writer = tf.python_io.TFRecordWriter(filename)

# Write all images into Example Protocol Buffer .
# Features including:
# 	'pixels': 	typeof _int64_feature	valueof pixels
#	'labels': 	typeof _int64_feature	valueof labels
#	'image_raw':	typeof _bytes_feature	valueof image_raw
# 
for index in range(num_examples):
	# Converse image matrix into a string .
	image_raw = images[index].tostring()
	# Converse one sample into Example Protocol Buffer and fill with all information .
	example = tf.train.Example(features=tf.train.Features(feature={
			'pixels': _int64_feature(pixels),
			'labels': _int64_feature(np.argmax(labels[index])),
			'image_raw': _bytes_feature(image_raw)}))

	# Write one Example into TFRecord .
	writer.write(example.SerializeToString())

# Close writer file handler .
writer.close()
	







































