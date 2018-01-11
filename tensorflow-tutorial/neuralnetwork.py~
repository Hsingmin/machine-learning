
__author__ = 'Lamdar'

# neuralnetwork.py -- Get a neural network through tensorflow .
#
# Forward Propagation Algorithm .
# Nueral network parameters and tensorflow variables .
# Training model through tensorflow .

import tensorflow as tf

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
x = tf.placeholder(tf.float32, shape = (3, 2), name = "input")

# Forward propagation to get output.
a = tf.matmul(x, W1)
y = tf.matmul(a, W2)

init_ops = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init_ops)
	print(sess.run(y, feed_dict = {x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))

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
























