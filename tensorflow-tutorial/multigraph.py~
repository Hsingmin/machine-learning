
__author__ = 'Hsingmin'

# multigraph.py -- Tensorflow support multi Graph test.

import tensorflow as tf

# Define Graph g1 .
g1 = tf.Graph()
with g1.as_default():
	# Define variable "v" in Graph g1,
	# Set g1.v = 0 .
	#
	# tensorflow.get_variable("bias", initializer=tf.zeros_initializer(shape=[out_channels])) 
	# has been already removed in higher version,
	# use tensorflow.get_variable("bias", shape=[out_channels], initializer=tf.zeros_initializer()) instead .
	# v = tf.get_variable("v", initializer = tf.zeros_initializer(shape = [1]))
	v = tf.get_variable("v", shape = [1], initializer=tf.zeros_initializer())

# Define another Graph g2 .
g2 = tf.Graph()
with g2.as_default():
	# Define variable "v" in Graph g2 ,
	# Set g2.v = 1 .
	# v = tf.get_variable("v", initializer = tf.ones_initializer(shape = [1]))
	v = tf.get_variable("v", shape = [1], initializer = tf.ones_initializer())

# Get variable v in Graph g1 .
with tf.Session(graph = g1) as sess:
	tf.global_variables_initializer().run()
	with tf.variable_scope("", reuse = True):
		print(sess.run(tf.get_variable("v")))

# Output: [0.] 

# Get variable v in Graph g2 .
with tf.Session(graph = g2) as sess:
	tf.global_variables_initializer().run()
	with tf.variable_scope("", reuse = True):
		print(sess.run(tf.get_variable("v")))
	
# Output: [1.]

# In TensorFlow, tensor just store a reference to a operation rather than a result .
a = tf.constant([1.0, 2.0], name = "a")
b = tf.constant([3.0, 4.0], name = "b")
result = tf.add(a, b, name = "add")
print(result)

# Output: Tensor("add:0", shape=(2,), dtype=float32)
#
# Get result value of the calculation in tensorflow.Session.run(result) .
#
# Output: [4. 6.]
with tf.Session() as sess:
	print(sess.run(result))

# Device Mapping: No known device.
config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = True)
sess1 = tf.InteractiveSession(config = config)
sess2 = tf.Session(config = config)





















