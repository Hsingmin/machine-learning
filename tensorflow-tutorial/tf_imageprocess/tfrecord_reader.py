
# -*- coding: utf-8 -*-
# -*- version:
#		python 3.5.2
#		Tensorflow 1.4.0
# -*- -------------------------------- -*-
# 
# tfrecord_reader.py -- Read data from TFRecord file .
import tensorflow as tf

# Create a reader to get samples stored in TFRecord file .
reader = tf.TFRecordReader()

# Create a list to maintain input file queue .
filename_queue = tf.train.string_input_producer(["./to/output.tfrecords"])

# Reade one example from TFRecord file .
# read_up_to() can be used to get multiple examples .
_, serialized_example = reader.read(filename_queue)

# Parse readed example .
# parse_example() can be used to parse mutiple examples . 
#
# Tensorflow provides two methods to parse TFRecord file features field :
# 	tf.FixedLenFeature() to get Tensor as result .
#	tf.VarLenFeature() to get SparseTensor processing sparse data . 
features = tf.parse_single_example(serialized_example,
		features={'image_raw': tf.FixedLenFeature([], tf.string),
			  'pixels': tf.FixedLenFeature([], tf.int64),
			  'labels': tf.FixedLenFeature([], tf.int64)})

# Parse string to corresponding pixel array with tf.decode_raw() method .
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['labels'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()
# Process data starting multithread .
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 
for i in range(10):
	image, label, pixel = sess.run([images, labels, pixels]) 



































