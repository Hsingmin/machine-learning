
# variablemanage.py -- variable create, get and share between function .

import tensorflow as tf

'''
v1 = tf.get_variable("v", shape=[1], initializer=tf.constant_initializer(1.0))
v2 = tf.Variable(tf.constant(1.0, shape=[1]), name="v")

with tf.variable_scope("foo"):
	v = tf.get_variable("v", shape=[1], initializer=tf.constant_initializer(1.0))

# Variable foo/v already exists, disallowed . Did you mean to set reuse=True in VarScope ?
# with tf.variable_scope("foo"):
# 	v = tf.get_variable("v", shape=[1])

# Set reuse=True in variable_scope, then tf.get_variable() will get variable already define .
with tf.variable_scope("foo", reuse=True):
	v1 = tf.get_variable("v", shape=[1])
	print(v == v1)		# Output : True , v1 is the same variable in tensorflow .

# Variable var/v does not exist, disallowed . Did you mean to set reuse=tf.AUTO_REUSE in VarScope ?
with tf.variable_scope("bar", reuse=tf.AUTO_REUSE):
	v = tf.get_variable("v", shape=[1])

# Variable var/v already exist, disallowed . Did you mean to set reuse=tf.True in VarScope ?
with tf.variable_scope("bar", reuse=True):
	v = tf.get_variable("v", shape=[1])

# tensorflow.variable_scope() can be called nestly .
with tf.variable_scope("root"):
	# Get reuse value in variable_scope by tf.get_variable_scope().reuse
	print(tf.get_variable_scope().reuse)			# Output : False , the outest reuse .

	with tf.variable_scope("foo", reuse=True):		# Create a new scope with reuse=True
		print(tf.get_variable_scope().reuse)		# Output : True

		with tf.variable_scope("bar"):			# Create a new nested scope and reuse will left be outter defaultly . 
			print(tf.get_variable_scope().reuse)	# Output : True

	print(tf.get_variable_scope().reuse)			# Output : False , for exiting from the scope reuse=True
'''


v1 = tf.get_variable("v", [1])
print(v1.name)							# Output : v:0 , "v" as variable name , 0 means v1 is the first element .

with tf.variable_scope("foo"):
	v2 = tf.get_variable("v", [1])
	print(v2.name)						# Output : /foo/v:0 with namespace as the prefix .

with tf.variable_scope("foo"):
	with tf.variable_scope("bar"):
		v3 = tf.get_variable("v", [1])			
		print(v3.name)					# Output : /foo/bar/v:0 with nested namespace as the prefix .

	v4 = tf.get_variable("v1", [1])
	print(v4.name)						# Output : /foo/v1:0 exit from /bar/ namespace 

	# Variable foo/v already exists , disallowed . 
	# Do you mean to set reuse=True or set reuse=tf.AUTO_REUSE in Variable_scope ? 
	# v4_ = tf.get_variable("v", [1])
	# print(v4_.name)					# Output : supposed to be /foo/v:1


# Create a new namescope with name="" and reuse=True
with tf.variable_scope("", reuse=True):
	v5 = tf.get_variable("foo/bar/v", [1])			# Get variable from other namespace with its full path .
	print(v5 == v3)						# Output : True

	v6 = tf.get_variable("foo/v1", [1])
	print(v6 == v4)						# Output : True










































