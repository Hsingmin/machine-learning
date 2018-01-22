
# conv_mnist_eval.py -- Testing on MovingAverage model with LeNet-5 structure.
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
# Load variables and functions defined in conv_mnist_inference.py and conv_mnist_train.py
import conv_mnist_inference
import conv_mnist_train

# Load latest model to get the accuracy on testing dataset .
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
	with tf.Graph().as_default() as g:
		# Define the format of inputs and outputs .
		x = tf.placeholder(tf.float32, [None, conv_mnist_inference.IMAGE_SIZE,\
						      conv_mnist_inference.IMAGE_SIZE,\
						      conv_mnist_inference.NUM_CHANNELS],\
						      name="x-input")
		y_ = tf.placeholder(tf.float32, [None, conv_mnist_inference.OUTPUT_NODE], name="y-input")

		# Reshape input validate dataset .
		xs = np.reshape(mnist.validation.images,\
				(None, conv_mnist_inference.IMAGE_SIZE,\
				       conv_mnist_inference.IMAGE_SIZE, \
				       conv_mnist_inference.NUM_CHANNELS))
		ys = mnist.validation.labels

		validate_feed = {x: xs, y_: ys}

		# Calculate forward-propagation value .
		#
		# Ignoring regularization loss by setting regularizer=None
		y = conv_mnist_inference.inference(x, False, None)

		# Calculate accuracy on validation datasets .
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		# Loading MovingAverage Model by renaming variables .
		variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		while True:
			with tf.Session() as sess:
				# Get latest model saved in ckpt file with checkpoint . 
				ckpt = tf.train.get_checkpoint_state(conv_mnist_train.MODEL_SAVE_PATH)
				if ckpt and ckpt.model_checkpoint_path:
					# Load model .
					saver.restore(sess, ckpt.model_checkpoint_path)
					# Get steps model saved .
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

					accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
					print("After %s training steps, validation accuracy=%g " %(global_step, accuracy_score))
				else:
					print("No checkpoint file found .")
					return
			time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
	mnist = input_data.read_data_sets("./data/", one_hot=True)
	evaluate(mnist)

if __name__ == '__main__':
	tf.app.run()






















































