
# convolutional_forward_propagation.py
# Convolutioanl layer forward propagation .

# Creatd filter weights and biases in matrix
# [filter_length, filter_width, current_layer_depth, next_layer_depth]
# in which next_layer_depth also as the filter depth .
#

import tensorflow as tf

filter_weight = tf.get_variable('weights', [5, 5, 3, 16],\
		initializer=tf.truncated_normal_initializer(stddev=0.1))

# Different position on current layer shares same biase ,
# biases matrix just owns one dimension , filter depth . 
biases = tf.get_variable('biases', [16],\
		intializer=tf.constant_initializer(0.1))


# Forward propagation with tensorflow.nn.conv2d() , 
# input: current layer node matrix , [batch, length, width, depth] .
# filter_weight: convolution weight , [filter_length, filter_width, current_layer_depth, filter_depth] .
# strides: strides on different dimension , [1, stride_on_length, stride_on_width, 1] .
# paddind: SAME -- zero_padding , VALID -- no_padding .
conv = tf.nn.conv2d(input, filter_weight, strides=[1, 1, 1, 1], padding='SAME')

# Add bias to all nodes on convolutional layer ,
# add the same bias to all elements in convolutional layer ,
# using API tensorflow.nn.bias_add() rather than raw add() .
bias = tf.nn.bias_add(conv, biases)
# Nonlinearize output with ReLU function . 
actived_conv = tf.nn.relu(bias)


# Maximum pool forward propagation with tensorflow.nn.max_pool() ,
# actived_conv: current conv-layer matrix , [batch, length, width, depth] .
# ksize: pool filter size , [1, pool_length, pool_width, 1] .
# strides: pool strides , [1, length_strides, width_strides, 1] .
# padding: SAME -- zero_padding , VALID -- no zero_padding .
pool = tf.nn.max_pool(actived_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


# Get a 4-dimension matrix as the input .
#
x = tf.placeholder(tf.float32,\
		[BATCH_SIZE, mnist_inference.IMAGE_SIZE,\
		mnist_inference.IMAGE_SIZE,\
		mnist_inference.NUM_CHANNELS], name='x-input')
y = tf.placeholder(tf.float32,)

# Reshape training data to be a 4-dimension matrix .
reshaped_xs = np.reshape(xs, (BATCH_SIZE,\
			mnist_inference.IMAGE_SIZE,\
			mnist_inference.IMAGE_SIZE,\
			mnist_inference.NUM_CHANNELS))











































