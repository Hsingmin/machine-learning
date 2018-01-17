
# mnistsimple.py -- minist dataset recognation in practice with tensorflow .

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

'''
# Load MNIST datasets
mnist = input_data.read_data_sets(".", one_hot=True)

print('Training dataset size : ', mnist.train.num_examples)

print('Validating dataset size : ', mnist.validation.num_examples)

print('Testing dataset size : ', mnist.test.num_examples)

print("Example training data : ", mnist.train.images[0])

print("Example training data labels : ", mnist.train.labels[0])

batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
# Pick batch_size training data from training-datasets .
print("X shape : ", xs.shape)
print("Y shape : ", ys.shape)
'''

# MNIST dataset constant .
INPUT_NODE = 784	# Input layer nodes also image pixel .
OUTPUT_NODE = 10	# Output layer nodes also image label .

# Set neural network parameters .
LAYER1_NODE = 500	# Hidden layer1 with 500 nodes .

BATCH_SIZE = 100	# Data batch size in one training step .

LEARNING_RATE_BASE = 0.8	# Base learning rate .
LEARNING_RATE_DECAY = 0.99	# Decay of learning rate .

REGULARIZATION_RATE = 0.0001	# Regularizer factor .
TRAINING_STEPS = 30000		# Total training steps .
MOVING_AVERAGE_DECAY = 0.99	# Moving average decay rate .

# Assistant function , calculate forward propagation value with inputs and parameters ,
# set a 3-layers full-connected network with ReLU activate function .
#
# Add hidden layer to get multi-layer structure , and ReLU to get nonlinear character ,
# average value calculating class referenced as a parameter to support using MovingAverage
# model when testing .
#
# Modified with namescope tensorflow.variable_scope() and tensorflow.get_variable() .
'''
def interface(input_tensor, avg_class, reuse=False):
	# Define variables and forward propagation of layer1 .
	with tf.variable_scope('layer1', reuse=reuse):
		# Create new variables or use those already created based on value of reuse .
		# Next time change parameter reuse=True , and no necessary transfer all parameters
		# in again .
		weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE],\
					initializer=tf.truncated_normal_initializer(stddev=0.1))
		biases = tf.get_variable("biases", [LAYER1_NODE],\
					initializer=tf.constant_initializer(0.0))
		# Use current parameters value to calculate layer1 output when no average class provided .
		if avg_class == None:
			layer1 = tf.nn.relu(tf.matmul(input_tensor, weights)+biases)
		else:
			layer1 = tf.nn.relu(tf.matmul(input_tensor,\
						avg_class.average(weights))+avg_class.average(biases))

	with tf.variable_scope('layer2', reuse=reuse):
		weights = tf.get_variable('weights', [LAYER1_NODE, OUTPUT_NODE],\
					initializer=tf.truncated_normal_initializer(stddev=0.1))
		biases = tf.get_variable("biases", [OUTPUT_NODE],\
					initializer=tf.constant_initializer(0.0))
		
		if avg_class == None:
			layer2 = tf.nn.relu(tf.matmul(layer1, weights)+biases)
		else:
			layer2 = tf.nn.relu(tf.matmul(layer1,\
						avg_class.average(weights))+avg_class.average(biases))
	return layer2
'''

def interface(input_tensor, avg_class, weights1, biases1, weights2, biases2):
	# Use current parameter value when no average class provided .
	if avg_class == None:
		layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)

		# Calculate the forward progpagation value of output layer .
		#
		# Softmax function will be used to calculate loss function , 
		# so activation is not put into use here .
		# We compare all outputs to pick the result of network , therefore 
		# it is not necessary to get the accurate value of output with softmax function .
		return tf.matmul(layer1, weights2) + biases2
	else:
		# Calculate variables' MovingAverage value .
		# And then calculate the forward propagation value .
		layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))

		return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

# Training model .
def train(mnist):
	x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
	y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
	
	
	# Produce hidden layer parameters .
	weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
	biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

	# Produce output layer parameters .
	weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
	biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
	

	# Calculate forward propagation value with current parameters , when aver_class == None .
	# Calculate forward propagation value with current parameters in namespace , create parameters
	# the first time with reuse=AUTO_REUSE .
	y = interface(x, None, weights1, biases1, weights2, biases2)
	# y = interface(x, avg_class=None)
	

	# Define global_step , storing the training steps that will not be used to calculate
	# MovingAverage , so allocated to be untrainable .
	global_step = tf.Variable(0, trainable=False)

	# Set MOVING_AVERAGE_DECAY to initialize aver_class to accelerate early training progress .
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

	# Apply MovingAverage to all variables of network ,
	# 
	# tensorflow.trainable_variables() returns elements in collection GraphKeys.TRAINABLE_VARIABLES ,
	# which not set to be trainable=False .
	variables_averages_op = variable_averages.apply(tf.trainable_variables())

	# Calculate forward propagation value with MovingAverage .
	#
	# For no change to variables' value and insteadly store them in shadow_varible ,
	# so call average function tensorflow.train.ExponentialMovingAverage() explicitly .
	# 
	# Calls interface() function using namescope already created by setting reuse=True .
	average_y = interface(x, variable_averages, weights1, biases1, weights2, biases2)
	# average_y = interface(x, variable_averages, reuse=True)

	# Calculate Cross-Entropy as the loss-function using 
	# tensorflow.sparse_softmax_cross_entropy_with_logits() function 
	# it will accelerate computing in dual-classification problem and calculate cross-entropy-loss
	# in multi-classfication problem .
	# 
	# parameter1 : forward propagation value withput softmax layer operation ;
	# parameter2 : the real label of datasets , use tf.argmax() to provide the category index .
	# tensorflow.argmax(tensor, axis) to find index corresponding the max value by row when axis=1 ,
	# and index corresponding the max value by column .
	# tensorflow.nn.sparse_softmax_cross_entropy_with_logits(logits=..., labels=...) must called with
	# named arguments 'logits' and 'labels' .
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
	
	# Calculate the mean cross-entropy in current batch .
	cross_entropy_mean = tf.reduce_mean(cross_entropy)

	# Calculate L2-Reularizer loss-function .
	regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
	# Calculate regularized loss of this model ignoring biases .
	regularization = regularizer(weights1) + regularizer(weights2)
	# Total loss function .
	loss = cross_entropy_mean + regularization

	# Set exponential decay learning rate .
	learning_rate = tf.train.exponential_decay(\
			LEARNING_RATE_BASE,			# Base learing rate decaying with training progress .
			global_step,				# Current iteration steps .
			mnist.train.num_examples/BATCH_SIZE,	# Iterations when all datasets inputed into model .
			LEARNING_RATE_DECAY)			# Learing rate decay rate .

	# Optimizer algorithm to minimize loss-function .
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
	
	# Tensorflow provides tf.control_dependencies() and tf.group() to complete many operations at one time ,
	# every time whole datasets inputed into model , parameters and the moving average value must be updated .
	#
	# train_op = tf.group([train_step, variables_averages_op])
	with tf.control_dependencies([train_step, variables_averages_op]):
		# tensorflow.no_op() returns a operation placeholder and does nothing .
		train_op = tf.no_op(name='train')			# Train operation placeholder .

	# Check the prediction of forward propagation value of network manipulating MovingAverage model .
	#
	# tensorflow.argmax(average_y, 1) :
	# average_y is a batch_size*10 array , asix = 1 means pick the index corresponding the maximum
	# value by column , which is the predicted label .
	# Then, tensorflow.average(average_y, 1) returns a array length of batch_size ,
	# tensorflow(tf.array(), tf.array()) returns an array like [True, False , ...] , in which
	# equal element assigned to be True , otherwise to be False .
	correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

	# Calculate accuracy by casting boolean value into float32 .
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# Create session and train model .
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
	
		# Prepare validation dataset , and determine stop condition and judge the result
		# during training progress .
		validation_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

		# Prepare testing dataset that cannot be used during training process ,
		# just used to judge model performance .
		test_feed = {x: mnist.test.images, y_: mnist.test.labels}

		# Train network iteratively .
		for i in range(TRAINING_STEPS):
			if i % 1000 == 0:
				# Calculate MovingAverage model's accuracy using validation datasets .
				#
				# Here input the whole validate dataset at one time rather than dividing
				# it into different batches for the dataset is not very large ,
				# but when network structure is complex and validate dataset is large ,
				# this may cause memory error .
				validate_acc = sess.run(accuracy, feed_dict=validation_feed)
				# Calculate MovingAverage model's accuracy on test dataset every 1000 steps .
				test_acc = sess.run(accuracy, feed_dict=test_feed)
				print("After %d training steps, validation accuracy using average model is %g, \
					test accuracy using average model is %g " %(i, validate_acc, test_acc))

			# Produce a batch for this training step . 
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			sess.run(train_op, feed_dict={x: xs, y_: ys})
		
		# Calculate the final accuracy of model on test dataset .
		final_test_acc = sess.run(accuracy, feed_dict=test_feed)
		print("After %d training steps, final test accuracy using average model is %g " %(TRAINING_STEPS, final_test_acc))

# The main function .
def main(argv = None):
	
	# Load MNIST datasets
	mnist = input_data.read_data_sets("./data", one_hot=True)
	train(mnist)

# main function entry tensorflow provided .
if __name__ == '__main__':
	tf.app.run()







































