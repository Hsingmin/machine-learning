
# embedding_demo.py

import tensorflow as tf
import numpy as np


embedding = tf.Variable(tf.reshape(tf.range(1, 26, name="embedding"), shape=[5, 5]))
input_ids = tf.placeholder(dtype=tf.int32, shape=[None])
input_embedding = tf.nn.embedding_lookup(embedding, input_ids)

with tf.Session() as sess:
	init = (tf.global_variables_initializer(),\
		tf.local_variables_initializer())
	sess.run(init)
	print("embedding matrix : ")
	print(sess.run(embedding))
	print("input_embedding matrix : ")
	print(sess.run(input_embedding, feed_dict={input_ids: [1, 2, 3, 0, 3, 2, 1]}))

























