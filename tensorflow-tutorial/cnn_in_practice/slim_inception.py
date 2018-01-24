
# slim_inception.py
# Inception-v3 model acheived by tensorflow-slim toolkit .

# Comparision between tensorflow raw API and tensorflow-slim toolkit
# to acheive a convolutional layer .
with tf.variable_scope(scope_name):
	weights = tf.get_variable("weight", \
			[CONV_SIZE, CONV_SIZE, NUM_CHANNELS, CONV_DEEP],\
			initializer=tf.truncated_normal_intializer(stddev=0.1))
	biases = tf.get_variable("bias",\
			[CONV_DEEP], intializer=tf.constant_initializer(0.0))
	conv = tf.nn.conv2d(former_relu, weights, stride=[1, STRIDE, STRIDE, 1], padding='SAME')
	relu = tf.nn.relu(tf.nn.bias_add(conv, biases))

# Get convolutional layer with tensorflow slim toolkit .
#
# Arguments (input_tensor, filter_deep, [filter_size, filter_size], 
# [filter_stride, filter_stride], padding, activate_function, variable_scope, ...)

import tensorflow.contrib.slim as slim
net = slim.conv2d(input_tensor, filter_deep, [filter_size, filter_size])

# Set default arguments value .
# Function in the list will take default value stride=1 , padding='SAME' ,
# that would be replaced by set value in following code given by developer . 
with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],\
			stride=1, padding='SAME'):
	# Inception-v3 model network structure code need to be added here .
	#
	# Produce Inception structure assuming former layer output net .
	# 
	net = former_relu

	# Declare a variable_scope for one inception module .
	with tf.variable_scope('Mixed_7c'):
		# Declare a variable_scope for one branch in inception .
		with tf.variable_scope('Branch_0'):
			# Get a filter with size=1*1 , deep=320
			brach_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')

		# The second branch in Inception that is also an Inception .
		with tf.variable_scope('Branch_1'):
			branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')

			# Concat matrix in deepth dimension , 
			# tf.concat(dimension) allocate dimension to be the dimension 
			# developer would like to concat .
			branch_1 = tf.concat(3, [slim.conv2d(branch_1, 384, [1,3], scope='Conv2d_0b_1x3'),\
						 slim.conv2d(branch_1, 384, [3,1], scope='Conv2d_0b_3x1')])

		with tf.variable_scope('Branch_2'):
			branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
			branch_2 = slim.conv2d(branch2, 384, [3, 3], scope='Conv2d_0b_3x3')
			branch_2 = tf.concat(3, [slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),\
						 slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0c_3x1')])

		with tf.variable_scope('Branch_3'):
			branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
			branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')

		# Concat branch0 ~ branch3 to net .
		net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])



















































