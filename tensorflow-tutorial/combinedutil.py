
# combinedutil.py -- Convert variables and the values in graph into constants 

import tensorflow as tf
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init_op)

	# Extract GraphDef from current Graph .
	# We can execute the calculating operations of forward propagation with the GraphDef .
	graph_def = tf.get_default_graph().as_graph_def()

	# Convert variables and the value into constants , and delete unneccesary nodes in Graph ,
	# parameter ['add'] set the node to extract .
	#
	output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])

	# Save extracted model to file .
	with tf.gfile.GFile("./model/combined_model.pb", "wb") as f:
		f.write(output_graph_def.SerializeToString())







































