
# tfb_run_logout.py -- Logout Tensorflow running information into TensorBoard .

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SUMMARY_DIR = "./monitor_to/log"
BATCH_SIZE = 100
TRAIN_STEPS = 30000

# Produce variable monitoring information and define operations producing 
# monitoring information log .
#
# Signature 'var' as the numeric tensor name need to be recorded , 
# Signature 'name' as the node given in visualization 
# which should be named the same as variable name .  
def variable_summaries(var, name):
	# Place operations producing monitoring information into
	# name space 'summaries' .
	with tf.name_scope('summaries'):
		# Record variable values' distribution in tensor .
		#
		# tf.histogram() will return a Summary protocol buffer 
		# for given name and var . 
		# Corresponding Graph would be shown in HISTOGRAMS bar
		# after Summary writen into Tensorflow log file .
		#
		# tf.histogram_summary() would executed only in sess.run() ,
		# and produced output Summary Protocol Buffer .
		tf.summary.histogram(name, var)

		# Calculate variable mean value , and define operation producing
		# mean value information log .
		# 
		# Label of variable mean value named as 'mean/'+name,
		# in which mean as name scope .
		# Monitoring items in the same name scope will placed in the same bar ,
		# and name indicating the variable of current item monitoring . 
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean/' + name, mean)

		# Calculate the stddev of variables and define operation logout .
		stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev/' + name, stddev)


# Get full-connected layer of network .
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
	# Place network layer into name scope .
	with tf.name_scope(layer_name):
		# Declare weights on network edge ,
		# call variable_summaries() to produce weights 
		# information motitoring log .
		with tf.name_scope('weight'):
			weights = tf.Variable(tf.truncated_normal(
				[input_dim, output_dim], stddev=0.1))
			variable_summaries(weights, layer_name + '/weights')

		# Declare biases on network , and produce monitoring log .
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
			variable_summaries(biases, layer_name + '/biases')

		# Forward propagation , produce monitoring log .
		with tf.name_scope('Wx_plus_b'):
			preactivate = tf.matmul(input_tensor, weights) + biases
			# Record output before activation .
			tf.summary.histogram(layer_name + '/pre_activations',
					preactivate)
		
		# act() set to be tf.nn.relu() defaultly and changed 
		# in signature of nn_layer()
		activations = act(preactivate, name='activation')
		
		# Record the distribution of nodes after activation .
		#
		# For layer1 , ReLU activation used ,
		# for layer2 , no activation used .
		tf.summary.histogram(layer_name + '/activations', activations)
		
		return activations

#
def main(_):
	mnist = input_data.read_data_sets("./data", one_hot=True)

	# Define inputs .
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32, [None, 784], name='x-input')
		y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

	# Reshape input tensor into image pixel matrix ,
	# write image information into log .
	with tf.name_scope('input_reshape'):
		image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
			
		# tf.image_summary() has been removed in TensorFlow 1.0 
		# use tf.Summary.Image() instead .
		tf.summary.image('input', image_shaped_input)

	hidden1 = nn_layer(x, 784, 500, 'layer1')
	y = nn_layer(hidden1, 500, 10, 'layer2', act=tf.identity)

	# Calculate cross-entropy and produce cross-entropy monitoring log .
	with tf.name_scope('cross_entropy'):
		cross_entropy = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
		tf.summary.scalar('cross_entropy', cross_entropy)

	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
		
	
	# Calculate the classify accuracy and produce monitoring log .
	#
	# It would be accuracy on batch data when training , 
	# and would be accuracy on test data when test .
	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_prediction'):
			correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(
					tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('accuracy', accuracy)

	# Merge all summaries and executed in once time in sess.run(merged) .
	# tf.contrib.merge_all_summaries() has been deprecated in TensorFlow 1.0 ,
	# use tf.summary.merge_all() instead .
	merged = tf.summary.merge_all()

	with tf.Session() as sess:
		#
		summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
		init = (tf.global_variables_initializer(),
			tf.local_variables_initializer())
		sess.run(init)

		for i in range(TRAIN_STEPS):
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			# Operations of producing logs and train_ops .
			summary, _, accurace = sess.run([merged, train_step, accuracy], 
					feed_dict={x: xs, y_: ys})

			# Write log into Summary file .
			summary_writer.add_summary(summary, i)

			if i % 1000 == 0:
				print('After %d steps , accuracy on training dataset is %.1f%% ' % (i, accurace * 100))
	
	summary_writer.close()

if __name__ == '__main__':
	tf.app.run()










































