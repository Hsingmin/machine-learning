
# cgVariableDemo.py Computate Graph Variable use demo.

import tensorflow as tf

# Create a Variable, that will be initialized to the scalar value 0.
state = tf.Variable(0, name="counter")


# Create an op to add one to 'state'.

one = tf.constant(1)
# Tensorflow.add() and Tensorflow.assign() both executed after Session.run() method calling.
new_value = tf.add(state, one)
# Reassign variable 'state'.
update = tf.assign(state, new_value)

# Variables must be initialized by running an 'init' op after having
# launched the graph. We firstly have to add the 'init' op to the graph.
init_op = tf.global_variables_initializer()

# Launch the graph and run the ops.
with tf.Session() as sess:
	# Run the 'init' op.
	# Initialize all variables of the graph.
	sess.run(init_op)
	# Print the initial value of all variables 'state'
	print(sess.run(state))
	# Run the op that updates 'state' and print 'state'.
	for _ in range(3):
		sess.run(update)
		print(sess.run(state))
# output:
# 0
# 1
# 2
# 3

input1 = tf.constant(3.)
input2 = tf.constant(2.)
input3 = tf.constant(5.)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

# output all results one time when all ops finished in Session.run() method.
with tf.Session() as sess:
	result = sess.run([mul, intermed])
	print(result)

# output:
# [array([21.], dtype=float32), array([7.], dtype=float32)]

# Create placeholder of dtype, and fill them in Session.run() method.
pinput1 = tf.placeholder(tf.float32)
pinput2 = tf.placeholder(tf.float32)
poutput = tf.multiply(pinput1, pinput2)

with tf.Session() as sess:
	print(sess.run([poutput], feed_dict={pinput1:[7.], pinput2:[2.]}))

# output:
# [array([14.], dtype=float32)]




























