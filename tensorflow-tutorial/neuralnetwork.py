
__author__ = 'Lamdar'

# neuralnetwork.py -- Get a neural network through tensorflow .
#
# Forward Propagation Algorithm .
# Nueral network parameters and tensorflow variables .
# Training model through tensorflow .

import tensorflow as tf

# import numpy to get simulation data set .
from numpy.random import RandomState

# Set batch size to be 8 .
batch_size = 8

# Define variable W1 and W2 to store layer1 weights matrix and 
# layer2 weights matrix .
#
# Set random seed to get the same output running the model everytime .
#
# tensorflow.random_normal() returns dtype=float32 as default .
#
# w2 = tf.Variable(tf.random_normal([3, 1], dtype = float64, stddev = 1, name = "w2"))
# w1.assign(w2) 
# TypeError : type float64 does no match type float32 . 
#
# tf.assign(W1, W2)
# ValueError : Shape(2, 3) and (3, 1) are not compatible .
# tf.assign(W1, W2, validate_shape = False) can be executable .
W1 = tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1))
W2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))

# Define input vector to be a constant temporarily . 
# x = tf.constant([[0.7, 0.9]])
#
# Define placeholder to store input data .
#
# We often divide training dataset to be small batch,
# but input all test data at one time when test the model .
x = tf.placeholder(tf.float32, shape = (None, 2), name = "x-input")
y_ = tf.placeholder(tf.float32, shape = (None, 1), name = "y-input")

# Forward propagation to get output.
a = tf.matmul(x, W1)
y = tf.matmul(a, W2)

# Define Loss-Function : Cross-Entropy .
cross_entropy = -tf.reduce_mean(\
		y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

# Define Learning-rate : 0.001
learning_rate = 0.001
# Define Error-Backward-Propagation Algorithm :
# tensorflow.train.GradientDescentOptimizer()
# tensorflow.train.AdamOptimizer()
# tensorflow.train.MomentumOptimizer()
#
# We can optimize all variables in collection GraphKeys.TRAINABLE_VARIABLES
# in tensorflow.Session(train_step)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# Produce a simulation dataset shape=(128, 2) .
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

# Simulate the dataset label, (x1+x2)<1 to be negative sample, otherwise positive .
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

# Create Session to run tensorflow .
with tf.Session() as sess:
	init_ops = tf.global_variables_initializer()
	sess.run(init_ops)

	print(sess.run(W1))
	print(sess.run(W2))

	# Training .
	STEPS = 5000
	for i in range(STEPS):
		# Pick batch_size samples to train model .
		start = (i * batch_size) % dataset_size
		end = min(start + batch_size, dataset_size)

		# Train model on one batch .
		sess.run(train_step, feed_dict = {x: X[start : end], y_: Y[start : end]})

		if i % 1000 == 0:
			total_cross_entropy = sess.run(cross_entropy, feed_dict = {x: X, y_: Y})
			print("After %d Training steps, cross-entropy on whole dataset is %g" %(i, total_cross_entropy))

	print(sess.run(W1))
	print(sess.run(W2))























