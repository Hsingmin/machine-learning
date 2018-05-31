
# tf_graph.py

import tensorflow as tf

"""
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
"""
"""
g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v", shape=[1], initializer=tf.zeros_initializer(dtype=tf.float64))
    u = tf.Variable(tf.random_normal([1], stddev=0.01), name="u")
    # u = tf.get_variable("u", shape=[1], initializer=tf.random_normal_initializer(stddev=0.01))
    result1 = v + u

with tf.Session(graph=g1) as sess:
    sess.run([tf.global_variables_initializer(),
              tf.local_variables_initializer()])
    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        print(sess.run(tf.get_variable("u")))
        print(sess.run(tf.get_variable("v")))
        print(result1.eval(session=sess))

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v", shape=[1], initializer=tf.ones_initializer(dtype=tf.float64))
    u = tf.Variable(tf.random_normal([1], stddev=0.01), name="u")
    # u = tf.get_variable("u", shape=[1], initializer=tf.random_normal_initializer(stddev=0.01))
    result2 = v + u

with tf.Session(graph=g2) as sess:
    sess.run([tf.global_variables_initializer(),
              tf.local_variables_initializer()])
    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        print(sess.run(tf.get_variable("u")))
        print(sess.run(tf.get_variable("v")))
        print(result2.eval(session=sess))
"""
"""
with tf.name_scope('name_scope_x'):
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
    var3 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    var4 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name, sess.run(var1))    # var1:0 [0.20366716]
    print(var3.name, sess.run(var3))    # name_scope_x/var2:0 [2.]
    print(var4.name, sess.run(var4))    # name_scope_x/var2_1:0 [2.]
"""
"""
with tf.name_scope('name_scope_1'):
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
    var2 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initilaizer())
    print(var1.name, sess.run(var1))
    print(var2.name, sess.run(var2))
"""

with tf.variable_scope('variable_scope_y') as scope:
    var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
    scope.reuse_variables()
    var1_reuse = tf.get_variable(name='var1')
    var2 = tf.Variable(initial_value=[2.], name='var2', dtype=tf.float32)
    var2_reuse = tf.Variable(initial_value=[2.], name='var2', dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name, sess.run(var1))
    print(var1_reuse.name, sess.run(var1_reuse))
    print(var2.name, sess.run(var2))
    print(var2_reuse.name, sess.run(var2_reuse))














