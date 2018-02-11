
# -*- coding: utf-8 -*-
# -*- version:
#		python 3.5.2
# 		tensorflow 1.4.0
# -*- ------------------------------------------------ -*-
#
# data_batching.py -- 
# 	Batch multiple input data example 
#	for training nueral network model .

import tensorflow as tf

# Read and parse TFRecord file to get example ,
# i as example's feature vector , and j as label .

filenames = tf.train.match_filenames_once("./data/data.tfrecords-*")
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
		serialized_example,
		features={'i': tf.FixedLenFeature([], tf.int64),
		          'j': tf.FixedLenFeature([], tf.int64)})

example, label = features['i'], features['j']

# Example quantity in one batch .
batch_size = 3

# Queue size to group examples .
capacity = 1000 + 3 * batch_size

# Group example with tf.train.batch() , 
# [example, label] as grouped elements ,
# batch_size defines example quantity in one batch ,
# capacity defines queue size , and when queue length reach the capacity ,
# Tensorflow would pause pushing operation , once the length less than capacity ,
# pushing operation would restart .
example_batch, label_batch = tf.train.batch([example, label], 
		batch_size=batch_size, 
		capacity=capacity)

shuffle_example_batch, shuffle_label_batch = tf.train.shuffle_batch(
		[example, label], batch_size=batch_size,
		capacity=capacity, min_after_dequeue=30)

with tf.Session() as sess:
	init = (tf.global_variables_initializer(),
		tf.local_variables_initializer())
	sess.run(init)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	# Get grouped examples ,
	# and it would input into the network in real engineering problem .
	for i in range(2):
		cur_example_batch, cur_label_batch = sess.run(
				[example_batch, label_batch])
		print(cur_example_batch, cur_label_batch)

	print('---------- Using tf.train.shuffle_batch() -------------')
	
	for i in range(2):
		cur_example_batch, cur_label_batch = sess.run(
				[shuffle_example_batch, shuffle_example_batch])
		print(cur_example_batch, cur_label_batch)

	coord.request_stop()
	coord.join(threads)









































