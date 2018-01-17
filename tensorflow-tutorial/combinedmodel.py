
# combinedmodel.py -- Calculate operation "add" with model provided by file combined_model.pb .

import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
	model_filename = "./model/combined_model.pb"
	
	# Read model file and parse into GraphDef Protocol Buffer .
	with gfile.FastGFile(model_filename, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	
	# Load Graph saved in graph_def into current Graph , return_elements=["add:0"]
	# as the returned tensor name .
	#
	# Output : [array([3.], dtype=float32)]
	# tensor ["add:0"] including variable v1 , variable v2 , operation "add" 
	# and the result of operation .
	result = tf.import_graph_def(graph_def, return_elements=["add:0"])
	print(sess.run(result))






















































