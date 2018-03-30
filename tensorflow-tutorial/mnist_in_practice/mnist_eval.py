
# mnist_eval.py -- Testing on MovingAverage model .
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# Load variables and functions defined in mnist_inference.py and mnist_train.py
import mnist_inference
import mnist_train

from tensorflow.python.framework import graph_util 

# Load latest model to get the accuracy on testing dataset .
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
	with tf.Graph().as_default() as g:
		# Define the format of inputs and outputs .
		x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name="x-input")
		y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="y-input")
		validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

		# Calculate forward-propagation value .
		#
		# Ignoring regularization loss by setting regularizer=None
		y = mnist_inference.inference(x, None)

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
				ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
				if ckpt and ckpt.model_checkpoint_path:
					# Load model .
					saver.restore(sess, ckpt.model_checkpoint_path)
					# Get steps model saved .
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

					accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
					print("After %s training steps, validation accuracy=%g " %(global_step, accuracy_score))

				#	saver.export_meta_graph("./model/model.ckpt.meta.json", as_text=True)
				'''
				new_saver = tf.train.import_meta_graph('./model/model.ckpt-28001.meta')
				for var in tf.trainable_variables():
					print(var.name)
					new_saver.restore(sess, ckpt.model_checkpoint_path)
					all_vars = tf.trainable_variables()
					for v in all_vars:
						v_4d = np.array(sess.run(v))
						print(v_4d)
				'''
				else:
					print("No checkpoint file found .")
					return
			time.sleep(EVAL_INTERVAL_SECS)

				
def main(argv=None):
	mnist = input_data.read_data_sets("./data/", one_hot=True)
	evaluate(mnist)

if __name__ == '__main__':
	tf.app.run()






















































