
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
g1 = tf.Graph()
with g1.as_default():
    with tf.variable_scope("g1_scope", reuse=tf.AUTO_REUSE) as scope:
        v = tf.get_variable("v", shape=[1], initializer=tf.zeros_initializer(dtype=tf.float32))
        w = tf.Variable(initial_value=[2.], name='w', dtype=tf.float32)
        # w = tf.get_variable("w", shape=[1], initializer=tf.random_normal_initializer(stddev=0.01))
        u = tf.get_variable("u", shape=[1], initializer=tf.random_normal_initializer(stddev=0.01, dtype=tf.float32))
        result1 = v + u + w

with tf.Session(graph=g1) as sess:
    sess.run([tf.global_variables_initializer(),
              tf.local_variables_initializer()])
    print("In graph1 ================= ")
    print(sess.run(v))
    print(sess.run(u))
    print(sess.run(w))
    print(result1.eval(session=sess))

g2 = tf.Graph()
with g2.as_default():
    with tf.variable_scope("g2_scope", reuse=tf.AUTO_REUSE) as scope:
        v = tf.get_variable("v", shape=[1], initializer=tf.ones_initializer(dtype=tf.float32))
        w = tf.Variable(initial_value=[2.], name='w', dtype=tf.float32)
        # w = tf.get_variable("w", shape=[1], initializer=tf.random_normal_initializer(stddev=0.01))
        u = tf.get_variable("u", shape=[1], initializer=tf.random_normal_initializer(stddev=0.01, dtype=tf.float32))
        result2 = v + u + w

with tf.Session(graph=g2) as sess:
    sess.run([tf.global_variables_initializer(),
              tf.local_variables_initializer()])
    print("In graph2 ================= ")
    print(sess.run(v))
    print(sess.run(u))
    print(sess.run(w))
    print(result2.eval(session=sess))
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
# 在tf.name_scope下，tf.get_variable()创建的变量名不受name_scope的影响，并且在
# 未指定共享变量时，重名会报错，tf.Variable()则会检测有没有变量重名，若有则进行处理
# 若使用tf.get_variable()创建变量，且没有设置共享变量，重名会报错
# 因此，需要共享变量时，使用tf.variable_scope()
"""
g1 = tf.Graph()

with g1.as_default():
    with tf.variable_scope('variable_scope_x') as scope:
        var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
        scope.reuse_variables()
        var1_reuse = tf.get_variable(name='var1')
        var2 = tf.Variable(initial_value=[2.], name='var2', dtype=tf.float32)
        var2_reuse = tf.Variable(initial_value=[2.], name='var2', dtype=tf.float32)
        result = var1+var1_reuse+var2+var2_reuse


with tf.Session(graph=g1) as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name, sess.run(var1))
    print(var1_reuse.name, sess.run(var1_reuse))
    print(var2.name, sess.run(var2))
    print(var2_reuse.name, sess.run(var2_reuse))
    print("In graph1, get result = ", result.eval(session=sess))

g2 = tf.Graph()

with g2.as_default():
    with tf.variable_scope('variable_scope_y') as scope:
        var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32)
        scope.reuse_variables()
        var1_reuse = tf.get_variable(name='var1')
        var2 = tf.Variable(initial_value=[2.], name='var2', dtype=tf.float32)
        var2_reuse = tf.Variable(initial_value=[2.], name='var2', dtype=tf.float32)
        result = var1+var1_reuse+var2+var2_reuse


with tf.Session(graph=g2) as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name, sess.run(var1))
    print(var1_reuse.name, sess.run(var1_reuse))
    print(var2.name, sess.run(var2))
    print(var2_reuse.name, sess.run(var2_reuse))
    print("In graph2, get result = ", result.eval(session=sess))
"""





