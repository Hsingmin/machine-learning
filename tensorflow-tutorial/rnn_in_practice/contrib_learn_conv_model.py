
# contrib_learn_conv_model.py -- Create CNN model with tf.contrib.learn

import tensorflow as tf
from sklearn import metrics

# Achieve convolutional layers with API in tf.contrib.layers 
layers = tf.contrib.layers
learn = tf.contrib.learn

# Self-defined model with specified arguments :
# 	input : features of input examples
# 	target : real label of input examples
#	mode : train or test

def conv_model(input, target, mode):
	# Converse real label into one-hot incode .
	target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)
	# Set network structure by conversing input into 3-D matrix ,
	# the first dimension as the examples in one batch .
	# input_layer [batch_size, length=28, width=28, deepth=1]
	network = tf.reshape(input, [-1, 28, 28, 1])

	# Create convolutional kernel with size=5*5 . 
	network = layers.convolution2d(network, 32, kernel_size=[5,5],
			activation_fn=tf.nn.relu)

	# Creatd max-pool layer size=2*2 , strides=2*2 .
	network = tf.nn.max_pool(network, ksize=[1,2,2,1], 
			strides=[1,2,2,1], padding='SAME')
	
	# Conv_layer2 with kernel size=5*5 .
	network = layers.convolution2d(network, 64, kernel_size=[5,5],
			activation_fn=tf.nn.relu)
	# Max-pool layer2 with size=2*2, strides=2*2 .
	network = tf.nn.max_pool(network, ksize=[1,2,2,1],
			strides=[1,2,2,1], padding='SAME')

	# Reshape CNN output layer into a vector for 
	# full-connected layer with size=7*7*64 .
	network = tf.reshape(network, [-1, 7*7*64])
	# Add dropout into network when training .
	network = layers.dropout(
			layers.fully_connected(network, 500, activation_fn=tf.nn.relu),
			keep_prob=0.5,
			is_training=(mode == tf.contrib.learn.ModeKeys.TRAIN))
	# Predicted result of final fully-connected layer .
	logits = layers.fully_connected(network, 10, activation_fn=None)
	# Loss function .
	loss = tf.losses.softmax_cross_entropy(target, logits)

	# Optimizer and train_op .
	train_op = layers.optimize_loss(
			loss,
			tf.contrib.framework.get_global_step(),
			optimizer='SGD',
			learning_rate=0.01)

	return tf.arg_max(logits, 1), loss, train_op

# Load datasets .
mnist = learn.datasets.load_dataset('mnist')
# Train CNN model on datasets .
classifier = learn.Estimator(model_fn=conv_model)
classifier.fit(mnist.train.images, mnist.train.labels, batch_size=100, steps=20000)
# Calculate accuracy on test dataset .
score = metrics.accuracy_score(mnist.test.labels, list(classifier.predict(mnist.test.images)))
print('Accuracy: {0:f}'.format(score))












































