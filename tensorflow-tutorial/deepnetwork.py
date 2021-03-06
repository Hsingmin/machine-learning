
__author__ = 'Lamdar'

# deepnetwork.py -- Add nonlinear activate function to nodes in network .
# 
# tensorflow.nn.relu() tensorflow.nn.sigmoid() tensorflow.nn.tanh()
#
# A = tf.nn.relu(tf.matmul(x, W1) + biases1)
# y = tf.nn.relu(tf.matmul(A, W2) + biases2)
#
# Measures the distance between two probabilities through Cross-Entropy,
# which is often used in Classification-Problem .

import tensorflow as tf

from numpy.random import RandomState 

# Softmax Regression handles the original output to become a Probablity Distribution .
# Cross Entropy measures the distance between output and real classification .
#
# cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1)))
# tensorflow.clip_by_value() to limite the value at a specified scale .
# tensorflow.log() to calculate log(element) one by one in a matrix .
# In Tersorflow, tensorflow.matmul() calculates matrix multiplication ,
# operator * calculates the elements multiplication one by one .
# And summary of the multiplication here is equivalent to its mean value for matrix's length keeps 
# the same, so use tensorflow.reduce_mean() .
# 
# Tensroflow provides API tensorflow.nn.softmax_cross_entropy_with_logits(y, y_) to package softmax
# and cross-entropy, logits calculation for programmers .
# Tensorflow provides API tensorflow.nn.sparse_softmax_cross_entropy_with_logits(y, y_) to accelerate
# the calculation when there is only one real output in classification problem .
#
# The Regression Problem often output just one value rather than many labels of classification ,
# tensorflow.reduce_mean(tf.square(y, y_)) to calculate the MSE(Mean Squre Error) of the model's output .

# Users can define loss function according to the model and the engineering problem .
# Tensorflow provides tensorflow.where() to select the specified element in a matrix .

batch_size = 8

# Two features set to be inputs .
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
# Just one output for Regression problem .
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# A simple weighted summary to propagate forward .
W1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, W1)

# Define Loss-Function .
loss_less = 10
loss_more = 1

# tensorflow.select() is deprecated in higher version, use tensorflow.where() instead .
# loss = tf.reduce_sum(tf.select(tf.greater(y, y_), (y-y_)*loss_more, (y_-y)*loss_less))
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y-y_)*loss_more, (y_-y)*loss_less))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# Get a simulation dataset .
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

# Add noise into the label of dataset .
Y = [[x1 + x2 + rdm.rand()/10.0-0.05] for (x1, x2) in X]

# Model Training .
# Different Loss-Function may produce different model, which is the key point .
#
# Error-Backforward-Propagation Algorithm and Gradient-Descent Algorithm , 
# both the most important category for parameters tunning in deeplearning . 
with tf.Session() as sess:
	init_ops = tf.global_variables_initializer()
	sess.run(init_ops)
	STEPS = 5000
	for i in range(STEPS):
		start = (i * batch_size) % dataset_size
		end = min(start + batch_size, dataset_size)
		sess.run(train_step, feed_dict={x: X[start : end], y_: Y[start : end]})

		print(sess.run(W1))

'''
# Step of network training .
batch_size = N

# Select a group of data as a batch for back-forward propagation .
x = tf.placeholder(tf.float32, shape=(batch_size, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(batch_size, 1), name='y-input')

# Define structure of network and loss-function .
loss = ...
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# Train network .
with tf.Session() as sess:
	# Parameters initialization .
	...
	# Update parameters iterarily .
	for i in range(STEPS):
		# Randomly pick dataset with size of batch_size from train-data to get better optimization.
		current_X, current_Y = ...
		sess.run(train_step, feed_dict={x: current_X, y_: current_Y})
'''

# Exponential Decay to set learning rate .
# decayed_learning_rate will be used in every training step , with learning_rate 
# is the original learning rate set by user , decay_rate is the decay factor set for 
# decay progress , decay_steps is the decay speed .
# decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
#
# learning_rate will be transformed to be a staircase function

global_step = tf.Variable(0)

# Produce learning_rate with exponential_decay() .
# Original learning_rate = 0.1, learning_rate will multiply 0.96 every 100 train iterations .
learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase = True)

# Tensorflow provides API to define L2 regularization function .
'''
w = tf.Variable(tf.random_normal([2, 1], stddev = 1, seed = 1))
y = tf.matmul(x, w)
loss = tf.reduce_mean(tf.square(y_ - y)) + tf.contrib.layers.l2_regularizer(lambda)(w)
'''

# Give a sample to use tensorflow.contrib.layers.l1_regularizer() and tensorflow.contrib.layers.l2_regularizer()
weights = tf.constant([[1.0, -2.0], [-3.0, 4.0]])
with tf.Session() as sess:
	print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))
	print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))





























