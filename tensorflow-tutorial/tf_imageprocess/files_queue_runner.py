
# files_queue_runner.py

import tensorflow as tf
import numpy as np

# 1000 4-D vectors with random element in range of 1~10 
data = 10 * np.random.randn(1000, 4) + 1
# 1000 random boolean numbers . 
target = np.random.randint(0, 2, size=1000)

# Create data input queue .
queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.int32])

# Enqueue push operation .
enqueue_op = queue.enqueue_many([data, target])
# Queue pop operation .
data_sample, label_sample = queue.dequeue()

# Create QueueRunner including 4 threads to execute push operation .
qr = tf.train.QueueRunner(queue, [enqueue_op]*4)

with tf.Session() as sess:
	coord = tf.train.Coordinator()
	# Start threads managed by QueueRunner qr to push queue and feed to main thread .
	enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
	
	# Main thread to consume 100 data .
	for step in range(100):
		if coord.should_stop():
			break
		data_batch, label_batch = sess.run([data_sample, label_sample])
		print(data_batch, label_batch)
	coord.request_stop()
	coord.join(enqueue_threads)
































