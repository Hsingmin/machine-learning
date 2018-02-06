
# tfb_logout.py -- Use TensorBoard to Visualize the model training process .

import tensorflow as tf

# A simple Compute Graph of add operation .
'''
with tf.name_scope("input1"):
	input1 = tf.constant([1.0, 2.0, 3.0], name="input1")

with tf.name_scope("input2"):
	input2 = tf.Variable(tf.random_uniform([3], name="input2"))

output = tf.add_n([input1, input2], name="add")

# Achieve a writer to logout ,
# add current compute graph into log .
writer = tf.summary.FileWriter("./to/log", tf.get_default_graph())
writer.close()
'''


with tf.variable_scope("foo"):
	# Get variable "foo/bar" in variable scope "foo" .
	a = tf.get_variable("bar", [1])
	print(a.name)				# print foo/bar:0

with tf.variable_scope("bar"):
	# Get variable "bar/bar" in variable scope "bar" 
	# variable "bar/bar" would not conflict with "foo/bar" .
	b = tf.get_variable("bar", [1])
	print(b.name)				# print bar/bar:0
	v = tf.Variable([1])
	print(v.name)

with tf.name_scope("a"):
	# It would get variable "a/Variable" with tf.Variable() .
	a = tf.Variable([1])
	print(a.name)				# print "a/Variable:0"

	# It would get variable "b" without "a/" as its prefix when using 
	# tf.get_variable() , for variable b not in scope "a" .
	a = tf.get_variable("a", [1])
	print(a.name)

with tf.name_scope("b"):
	# Variable get by tf.get_variable() would not influenced by name scope ,
	# it would raise error when tied to get variable name "a" for it already
	# declared .
		
	# b = tf.get_variable("a", [1])
	
	# print ValueError: Variable bar already exists, disallowed . Did you mean
	# to set reuse=True in VariableScope? Originally defined at: ...
	b = tf.get_variable("b", [1])
	print(b.name)



































