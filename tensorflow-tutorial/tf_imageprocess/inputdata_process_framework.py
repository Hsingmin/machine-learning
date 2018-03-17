
# -*- coding: utf-8 -*-
# -*- version:
#		python 3.5.2
#		tensorflow 1.4.0
# -*- ------------------------------------ -*-
#
# inputdata_process_framework.py 

import tensorflow as tf
import image_preprocess

# Create file list and file queue .
# Before that , store training data into TFRecord file with unified format .
filenames = tf.train.match_filenames_once("./data/file_pattern-*")
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

# Get and parse image data from specified TFRecord files in queue .
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
		serialized_example,
		features={
		'image': tf.FixedLenFeature([], tf.string),
		'label': tf.FixedLenFeature([], tf.int64),
		'height': tf.FixedLenFeature([], tf.int64),
		'width': tf.FixedLenFeature([], tf.int64),
		'channels': tf.FixedLenFeature([], tf.int64)})

image, label = features['image'], features['label']
height, width = features['height'], features['width']
channels = features['channels']

# Parse pixel matrix from raw image and set image size .
decoded_image = tf.decode_raw(image, tf.uint8)
decoded_image.set_shape([height, width, channels])

# Set input image size .
image_size = 299
# Preprocess images with 
distorted_image = image_preprocess.preprocess_for_train(decoded_image, image_size, None)

#
min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch(
		[distorted_image, label], batch_size=batch_size,
		capacity=capacity, min_after_dequeue=min_after_dequeue)

# Define network structure and optimization process ,
# image_batch as input for input layer , image_label give real classify results . 
logit = inference(image_batch)
loss = calc_loss(logit, label_batch)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#
with tf.Session() as sess:
	# Variables initilaization and threads start .
	init = (tf.global_variables_initializer(),
		tf.local_variables_initializer())
	sess.run(init)

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	# Train network .
	for i in range(TRAINING_ROUNDS):
		sess.run(train_step)

	# Stop all threads .
	coord.request_stop()
	# Wait for all threads quiting . 
	coord.join(threads)















































