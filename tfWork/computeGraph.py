
# computeGraph.py -- first demo for compute graph

import tensorflow as tf


# terrible codes from tensorflow in Action WTF
'''
b = tf.Variable(tf.zeros([100])) # vector with 100 demension
W = tf.Variable(tf.random_uniform([784,100], -1, 1)) # 784*100 random matrix
x = tf.placeholder(name="x") # placeholder
relu = tf.nn.relu(tf.matmul(W,x)+b) # ReLU(Wx+b) 
C = [...] # calculate Cost
s = tf.Session()
for step in range(0,10):
	# input = ...construct 100-D input array... # input 100-D vector
	input = tf.Variable(tf.uniform_random([1, 100], -1, 1))
	result = s.run(C, feed_dict={x: input}) # get Cost, give it to input x
	print(step, result)
'''

# Create a Constant op that produces a 1x2 matrix. 
# The op is added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])

# Create another Constant the produces a 2x1 matrix.
matrix2 = tf.constant([[2.], [2.]])

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', presents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2)

# Launch the default graph.
# sess = tf.Session()

# To run the matmul op we call the session 'run()' method, passing
# 'product' which represents the ouptput of matmul op. This indicates
# to the call that we want to get the output of the matmul op back.
#
# All inputs needed by the op are run automatically by threes ops in the
# graph: the two constants and matmul.
#
# The call 'run(product)' thus causes the execution of three ops in the
# graph: the two constants and matmul.
#
# The output of the op is returned in 'result' as a numpy 'ndarray' object.

# Sessions must be closed when task is done, use 'with' keyword to program.
# Execute Graph computing use Session Session.run() Session.close()

with tf.Session() as sess:
	result = sess.run([product])
	print(result)
	# ==>[[12.]]

# Close the Session when task is done.
# sess.close()

# All ops of tensorflow will be executed on device (CPUs, GPUs).
# When executed on GPU, the GPU0 used default.
#
# If GPU1 is intended to be used, program like this.
'''
with tf.Session() as sess:
	with tf.device("/gpu:1"):
		matrix1 = tf.constant([[3., 3.]])
		matrix2 = tf.constant([[2.], [2.]])
		product = tf.matmul(matrix1, matrix2)
		result = sess.run(product)
		print(result)
'''

# Enter an interactive Tensorflow Session. 
intsess = tf.InteractiveSession()

x = tf.Variable([1., 2.])
a = tf.constant([3., 3.])

# Initialize 'x' using the run() method of its initializer op.
x.initializer.run()

# Add an op to substract 'a' from 'x', run and print the result.
# tensorflow.mul() tensorflow.sub() tensorflow.neg() has been deprecated,
# use tensorflow.multiply() tensorflow.subtract() tensorflow.negative() instead.
sub = tf.subtract(x, a)
print(sub.eval())
# ==>[-2., -1.]

# Close the Session when we'v done.
intsess.close()






















