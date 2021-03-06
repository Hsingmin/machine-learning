
# mnist_train.py -- Training Neural Network .
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Load constants and forward propagation function in mnist_inference.py
import mnist_inference
import csv

# Network Arguments
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 3000		# MARK
MOVING_AVERAGE_DECAY = 0.99

# Model saved path .
MODEL_SAVE_PATH = './model/'
MODEL_NAME = "model.ckpt"
CSV_NAME = "parameter.csv"

def train(mnist):
	# Define inputs placeholder .
	x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
	y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-output')

	# Regularizer defination .
	regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
	
	# Forward propagation by calling mnist_inference.inference()
	y = mnist_inference.inference(x, regularizer)
	# Record global training steps .
	global_step = tf.Variable(0, trainable=False)

	# Define loss-function , learning-rate, moving-average and training-steps .
	#
	# Apply MovingAverage to all trainable variables to get a robust model . 
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())

	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	# Add cross-entropy and regularizer saved in collection .
	loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

	learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, \
							global_step,\
							mnist.train.num_examples/BATCH_SIZE,\
							LEARNING_RATE_DECAY)
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
	
	# Update both variables and the moving-averages .
	# tensorflow.no_op() is a operation placeholder .
	with tf.control_dependencies([train_step, variables_averages_op]):
		train_op = tf.no_op(name='train')

	# Initialize persistence class .
	saver = tf.train.Saver()
	
	#layer1_weights, layer1_biases = mnist_inference.get_model_parameters('layer1')
	#layer2_weights, layer2_biases = mnist_inference.get_model_parameters('layer2')
	
	#csv_out = open(os.path.join(MODEL_SAVE_PATH, CSV_NAME), 'w', newline='')
	#csv_writer = csv.writer(csv_out, dialect='excel')
	with tf.Session() as sess:
		tf.global_variables_initializer().run()

		# Raw training process , no validating and testing .
		for i in range(TRAINING_STEPS):
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			_, loss_value, step = sess.run([train_op, loss, global_step],\
							feed_dict={x: xs, y_: ys})

			# Save model every 1000 steps .
			if i % 1000 == 0:
				# Calculate current loss on training dataset .
				print("After %d training steps, loss on training batch is %g ." %(step, loss_value))

				# Save current model .
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

		#l1_weights, l1_biases = sess.run([layer1_weights, layer1_biases])
		#csv_writer.writerow(l1_weights)
		#csv_writer.writerow(l1_biases)
		
def main(argv=None):
	mnist = input_data.read_data_sets("./data/", one_hot=True)
	train(mnist)

if __name__ == '__main__':
	tf.app.run()
		





































