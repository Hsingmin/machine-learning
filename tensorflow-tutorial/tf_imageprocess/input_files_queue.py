
# -*- coding: utf-8 -*-
# -*- version:
#		python 3.5.2
#		tensorflow 1.4.0
# -*- ----------------------------------- -*-
#
# input_files_queue.py

import tensorflow as tf
import os

# Get input file queue with tf.train.match.filename_once() function .
# files = tf.train.match_filenames_once(['./data/data.tfrecords-00000-of-00002',\
# 				       './data/data.tfrecords-00001-of-00002'])
files = tf.train.match_filenames_once('./data/data.tfrecords-*')


# Create input files queue with tf.train.string_input_producer() ,
# in which files is file list got from tf.train.match_filename_once() ,
# shuffle set to be False here to avoid messing up the order ,
# but it would be set to True in normal engineering project .
#
# tf.train.string_input_producer(string_tensor, num_epochs=None, ...)
# string_tensor must be a list [path1, path2, ...]
filename_queue = tf.train.string_input_producer(files, shuffle=False)

# Read and parse an example .
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example,
		serialized_example,
		features={
		'i': tf.FixedLenFeature([], tf.int64),
		'j': tf.FixedLenFeature([], tf.int64)})


with tf.Session() as sess:
	# Attempting to use uninitialized value matching_filenames 
	# when calling tf.train.match_filenames_once() because in
	# tf.train.match_filenames_once(pattern) argument pattern is
	# a local variable .
	# Initialize both global variable and local variable in initializer . 
	init = (tf.global_variables_initializer(), tf.local_variables_initializer())
	sess.run(init)

	print(sess.run(files))

	# Coordinator and start all threads . 
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	
	# Multiple executions of operations to get data .
	for i in range(6):
		print(sess.run([features['i'], features['j']]))
	coord.request_stop()
	coord.join(threads)


































