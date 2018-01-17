
# persistence.py -- Persistence model with tensorflow.train.Saver class .

import tensorflow as tf

'''
# Declare two variables and add them .
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()

# Declare tensorflow.train.Saver class to persistent model .
saver = tf.train.Saver()

with tf.Session() as sess:

	# sess.run(init_op)
	# saver.save(sess, "./model/model.ckpt")
	
	# Load model and calculate result with variables saved in model .
	saver.restore(sess, "./model/model.ckpt")
	print(sess.run(result))

# Load persistence graph .
#
# File model.ckpt.meta saves neural network structure ,
# file model.ckpt saves variables value ,
# file checkpoint keeps a list including all model in the same path .
saver = tf.train.import_meta_graph("./model/model.ckpt.meta")

with tf.Session() as sess:
	saver.restore(sess, "./model/model.ckpt")
	# Get tensor by tensor name .
	print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))

	# Output : [.3]

# Declare variables with different name from saved model .
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="other-v2")

saver = tf.train.Saver()
# saver = tf.train.Saver({"v1":v1, "v2":v2})

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init_op)
	
	print(v1)
	print(v2)

	print(sess.run(v1))
	print(sess.run(v2))
'''

v = tf.Variable(0, dtype=tf.float32, name="v")
# There is only one variable v .
# Output : "v:0"
#
# tensorflow.all_variables() has been deprecated since 2017-03 ,
# use tensorflow.global_variables() instead .
for variables in tf.global_variables():
	print(variables.name)

# Moving average model applied to all variables .
ema = tf.train.ExponentialMovingAverage(0.99)

print(ema.variables_to_restore())

# maintain_averages_op = ema.apply(tf.global_variables())

# Tensorflow will produce a shadow variable v/ExponentialMoving Average .
# Output : "v:0" and "v/ExponentialMoving Average"
for variables in tf.global_variables():
	print(variables.name)

saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:

	'''
	init_op = tf.global_variables_initializer()
	sess.run(init_op)

	sess.run(tf.assign(v, 10))
	sess.run(maintain_averages_op)

	# Tensorflow will save both "v:0" and v/ExponentialMoving Average into model.ckpt
	saver.save(sess, "./model/model.ckpt")

	print(sess.run([v, ema.average(v)]))
	# Output : [10.0, 0.099999905]
	'''
	
	saver.restore(sess, "./model/model.ckpt")
	print(sess.run(v))
'''
# Create a new variable , named as "v"
v = tf.Variable(0, dtype=tf.float32, name="v")
# v1 = tf.Variable(0, dtype=tf.float32, name="v1")
# Assign moving-average value to variable v by rename "v/ExponentialMovingAverage" to v .
saver = tf.train.Saver({"v/ExponentialMovingAverage":v})
with tf.Session() as sess:
	saver.restore(sess, "./model/model.ckpt")
	print(sess.run(v))
'''































