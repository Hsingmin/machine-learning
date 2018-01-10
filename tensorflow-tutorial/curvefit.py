
# curvefit.py -- first demo for tensorflow on Google-api
#
# Simplify the lib-name.
import tensorflow as tf
import numpy as np

# Produce 100 group data, in function of y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype("float32")
y_data = x_data * 0.1 + 0.3

# Get weight value 'w' and bias value 'b'
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize MSE-Mean Square Error.
loss = tf.reduce_mean(tf.square(y - y_data))

# Learn method: Gradient Descent Optimizer.
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Initialize tensorflow parameters.
init = tf.global_variables_initializer()

# Creat Session to run tensorflow.
sess = tf.Session()
sess.run(init)

# Get iterable values of W and b.
#
# range() in python3.x is xrange() in python2.x, it is in fact range() 
# is removed in python3.x .
for step in range(201):
	sess.run(train)
	if step % 20 == 0:
		print(step, sess.run(W), sess.run(b))

# The best output is W ~ 0.1, b ~0.3































