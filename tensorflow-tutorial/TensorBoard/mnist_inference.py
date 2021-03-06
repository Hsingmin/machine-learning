
# mnist_inferenct.py -- Define forward propagation progress and arguments of network .
import tensorflow as tf

# Neural Network Arguments .
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

# 
def get_weight_variable(shape, regularizer):
	weights = tf.get_variable("weights", shape,\
				initializer=tf.truncated_normal_initializer(stddev=0.1))

	# Add regularizer into collection 'losses' .
	if regularizer != None:
		tf.add_to_collection('losses', regularizer(weights))

	return weights

# Forward Propagation of Neural Network .
def inference(input_tensor, regularizer):
	# Layer1 arguments and forward propagation declare.
	with tf.variable_scope('layer1'):
		# No differece between tf.get_variable() and tf.Variable() 
		# for just one call in the same program whether training or test . 
		# If called more than once , parameter reuse needed assigned to be True 
		# after the first calling .
		weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
		biases = tf.get_variable("biases", [LAYER1_NODE],\
				initializer = tf.constant_initializer(0.0))
		layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
	with tf.variable_scope('layer2'):
		# Declare weights and biases in scope 'layer2'
		weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
		biases = tf.get_variable("biases", [OUTPUT_NODE],\
				initializer = tf.constant_initializer(0.0))
		
		# Output layer without ReLU .
		# layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
		layer2 = tf.matmul(layer1, weights) + biases

	return layer2








































