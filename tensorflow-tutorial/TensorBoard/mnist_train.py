
# mnist_train.py -- Training Neural Network with name socope and TensorBoard Visualization .
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Load constants and forward propagation function in mnist_inference.py
import mnist_inference

# Network Arguments
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# Model saved path .
MODEL_SAVE_PATH = './model/'
MODEL_NAME = "model.ckpt"

def train(mnist):
	# Define computations processing input data in name scope "input" .
	with tf.name_scope('input'):
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
	# Define MovingAverage in name scope "moving_average" and apply 
	# to all trainable variables to get a robust model . 
	with tf.name_scope('moving_average'):
		variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
		variables_averages_op = variable_averages.apply(tf.trainable_variables())

	# Define loss function computation in name scope "loss_function" .
	with tf.name_scope("loss_function"):
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
		cross_entropy_mean = tf.reduce_mean(cross_entropy)
		# Add cross-entropy and regularizer saved in collection .
		loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

	# Create logout writer .
	train_writer = tf.summary.FileWriter("./mnist_to/log", tf.get_default_graph())

	# Define learning_rate , optimizer and train operations in name scope "train_step" .
	with tf.name_scope("train_step"):
		learning_rate = tf.train.exponential_decay(
							LEARNING_RATE_BASE,
							global_step,
							mnist.train.num_examples / BATCH_SIZE,
							LEARNING_RATE_DECAY,
							staircase=True)
		train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
	
		# Update both variables and the moving-averages .
		# tensorflow.no_op() is a operation placeholder .
		with tf.control_dependencies([train_step, variables_averages_op]):
			train_op = tf.no_op(name='train')

		# Initialize persistence class .
		saver = tf.train.Saver()
	
		with tf.Session() as sess:
			tf.global_variables_initializer().run()

			# Raw training process , no validating and testing .
			for i in range(TRAINING_STEPS):
				xs, ys = mnist.train.next_batch(BATCH_SIZE)
				
				# Save model every 1000 steps .
				if i % 1000 == 0:

					# Configure information need to be recorded when running .
					run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
					# ProtoBuffer file recording run information .
					run_metadata = tf.RunMetadata()
					
					_, loss_value, step = sess.run([train_op, loss, global_step],
								feed_dict={x: xs, y_: ys},
								options=run_options,
								run_metadata=run_metadata)
					
					# Write information of nodes running into log file .
					train_writer.add_run_metadata(run_metadata, 'step%03d' % i)

					# Calculate current loss on training dataset .
					print("After %d training steps, loss on training batch is %g ." %(step, loss_value))
					
					# Save current model .
					saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
			
				else:
					_, loss_value, step = sess.run([train_op, loss, global_step],
								feed_dict={x: xs, y_: ys})

	train_writer.close()

def main(argv=None):
	mnist = input_data.read_data_sets("./data/", one_hot=True)
	train(mnist)

if __name__ == '__main__':
	tf.app.run()
		





































