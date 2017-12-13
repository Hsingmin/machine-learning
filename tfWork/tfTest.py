
# tfTest.py

import tensorflow as tf

'''
g1 = tf.Graph()
with g1.as_default():
	v = tf.get_variable("v", initializer=tf.zeros_initializer(shape=[1]))

g2 = tf.Graph()
with g2.as_default():
	v = tf.get_variable("v", initializer=tf.ones_initializer(shape=[1]))

with tf.Session(graph=g1) as sess:
	tf.initialize_all_variables().run()
	with tf.variable_scope("", reuse=True):
		print(sess.run(tf.get_variable("v")))

with tf.Session(graph=g2) as sess:
	tf.initialize_all_variables().run()
	with tf.variable_scope("", reuse=True):
		print(sess.run(tf.get_variable("v")))
'''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("++++++++++++++ softmax regression accuracy : +++++++++++++++")
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))





















