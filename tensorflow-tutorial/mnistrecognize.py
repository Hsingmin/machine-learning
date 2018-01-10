
__author__ = 'lamdar'
# mnistrecognize.py - Tensorflow helloworld demo on Google main page.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Get MNIST_data sets from ./dir .
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
print('Download datasets done!')

# Describe computing progress. 
# Create a 2-D tensor to store the whole datasets.
#
# x is a holder, and assigned to required value when used.
x = tf.placeholder(tf.float32, [None, 784])

# Variable W as the weight and b as the bias, initialize to be zeros.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax regression model, y = softmax(x * W + b) .
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Pick the Cross-Entropy as Loss function.
#
# Set y_ holder to store the real label of image.
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# Gradient Descent Optimizer to minimize the Cross-Entropy.
#
# Backward-Propagation Algorithm.
# Learning rate is set to be 0.01, 
# tensorflow provides very simple api, we can focus on the model 
# rather than its implementation. 
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# In tensorflow 0.1.0 or higher version, 
# tensorflow.initialize_all_variables() is deprecated,
# use tensorflow.global_variables_initializer() instead.
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# 1000 iterations to train the model.
# In every iteration, pick 100 samples randomly, which is called batch.
#
# Fill the batch into placeholder x and y_ as the input of model training.
# Radom Gradient Descent.
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Get prediction value and real value of image classification.
# 
# tensorflow.arg_max() is deprecated and will removed in future version,
# use tensorflow.argmax() instead.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# Get classification accuracy, cast boolean value [True, False ...] 
# to float [1, 0, ...] .
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Test the model on test-datasets.
print("Accuracy on Test-Dataset: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))








































